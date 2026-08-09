import argparse
import csv
import math
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd

# The objective of this script:
# - Read from all three WDPA shapefile parts (0, 1, 2)
# - Merge them into one GeoDataFrame
# - Filter for marine protected areas in category Ia or Ib
# - Save to a single GeoPackage for downstream use

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WDPA_DIR = REPO_ROOT / "data" / "raw" / "WDPA"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "wdpa_marine_ia_ib.gpkg"
DEFAULT_INDEX_OUT = REPO_ROOT / "data" / "processed" / "mpa_index.csv"

# The geometry-free companion to the GeoPackage. `gfw_mpa_sweep.py` needs only the MPA
# *identifiers* to drive its per-region API calls, and it runs in CI where a geopandas /
# GDAL install would be dead weight — so the sweep reads this CSV with stdlib `csv` and
# the GeoPackage stays the source of truth for anything spatial. Committed (unlike the
# gitignored raw WDPA shapefiles) so a fresh clone and the nightly cron both have it.
INDEX_COLUMNS = ("WDPA_PID", "WDPAID", "NAME", "ISO3", "IUCN_CAT", "MARINE", "GIS_M_AREA")

# Paths to shapefiles (3 parts).
SHP_PATHS = [
    WDPA_DIR / "WDPA_Sep2025_Public_shp_0" / "WDPA_Sep2025_Public_shp-polygons.shp",
    WDPA_DIR / "WDPA_Sep2025_Public_shp_1" / "WDPA_Sep2025_Public_shp-polygons.shp",
    WDPA_DIR / "WDPA_Sep2025_Public_shp_2" / "WDPA_Sep2025_Public_shp-polygons.shp",
]

REQUIRED_COLUMNS = ("MARINE", "IUCN_CAT")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Filter the WDPA shapefiles to marine Ia/Ib MPAs -> GeoPackage."
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output GeoPackage path (default {DEFAULT_OUT}).")
    p.add_argument("--index-out", type=Path, default=DEFAULT_INDEX_OUT,
                   help=f"Geometry-free MPA index CSV, read by gfw_mpa_sweep.py "
                        f"(default {DEFAULT_INDEX_OUT}).")
    p.add_argument("--no-index", action="store_true",
                   help="Skip writing the MPA index CSV.")
    return p.parse_args(argv)


def check_inputs(shp_paths):
    """Raise a clear FileNotFoundError if any shapefile part is missing.

    The WDPA data is gitignored (too large), so a fresh clone will not have it;
    fail with a pointer instead of a deep geopandas/GDAL traceback."""
    missing = [p for p in shp_paths if not p.exists()]
    if missing:
        listed = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Missing WDPA shapefile part(s):\n  " + listed +
            "\n\nThe raw WDPA shapefiles are gitignored. Download the WDPA "
            "September 2025 public shapefile (protectedplanet.net) and extract the "
            "three split parts under data/raw/WDPA/ (see "
            "data/raw/WDPA/Shapefile_splitting_README.txt)."
        )


def load_wdpa(shp_paths):
    """Read and concatenate the split WDPA shapefiles into one GeoDataFrame.

    on_invalid="ignore" skips WDPA's self-intersecting polygons."""
    frames = []
    for p in shp_paths:
        print(f"Reading {p}")
        gdf = gpd.read_file(p, on_invalid="ignore")
        print(f"  {len(gdf)} features, CRS={gdf.crs}")
        frames.append(gdf)
    wdpa = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    print(f"Total features after merge: {len(wdpa)}")
    return wdpa


def filter_marine_ia_ib(wdpa):
    """Keep fully-marine/mixed (MARINE in {1,2}) MPAs in IUCN category Ia/Ib, in WGS84."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in wdpa.columns]
    if missing_cols:
        raise KeyError(
            f"WDPA data is missing expected column(s) {missing_cols}; "
            f"present columns: {sorted(wdpa.columns)}"
        )
    wdpa = wdpa.copy()
    wdpa["MARINE"] = pd.to_numeric(wdpa["MARINE"], errors="coerce")
    wdpa["IUCN_CAT"] = wdpa["IUCN_CAT"].astype(str).str.strip()

    mpas = wdpa[
        (wdpa["MARINE"].isin([1, 2])) &          # include fully marine or mixed
        (wdpa["IUCN_CAT"].isin(["Ia", "Ib"]))
    ]
    print(f"Filtered MPAs (Ia/Ib): {len(mpas)} of {len(wdpa)} features")
    return mpas.to_crs("EPSG:4326")


def write_gpkg_atomic(mpas, out_file):
    """Write the GeoPackage atomically: to a sibling .tmp.gpkg, then rename over
    the target. A GPKG is a single SQLite file, so the rename is atomic and a
    crash mid-write never corrupts an existing output. Cleaned up on failure."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_name(out_file.name + ".tmp.gpkg")
    try:
        if tmp.exists():
            tmp.unlink()
        mpas.to_file(tmp, driver="GPKG")
        tmp.replace(out_file)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return out_file


def _clean_cell(value):
    """Normalize one attribute for CSV: NaN/None -> "", floats that are whole -> int.

    WDPA carries WDPAID as a float (``1.0``), and a region id must go out over the wire
    as ``1``, not ``1.0`` — GFW would not resolve the latter."""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return repr(value)
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none") else text


def mpa_index_rows(records):
    """Project MPA attribute records onto INDEX_COLUMNS.

    Takes an iterable of mappings (``gdf.to_dict("records")``) rather than a GeoDataFrame
    so the projection is testable without geopandas. Rows with no usable WDPA_PID are
    dropped — a region id is the whole point of this file, and a blank one would become a
    guaranteed API failure at sweep time."""
    rows = []
    for rec in records:
        row = {col: _clean_cell(rec.get(col)) for col in INDEX_COLUMNS}
        if not row["WDPA_PID"]:
            continue
        rows.append(row)
    return rows


def write_index_atomic(rows, out_file):
    """Write the MPA index CSV atomically (sibling .tmp -> rename), UTF-8, LF endings.

    ``newline=""`` + explicit ``lineterminator`` keep the file byte-identical across
    Windows and the Linux CI runner, so committing it never produces a spurious diff."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_name(out_file.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(INDEX_COLUMNS), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(out_file)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return out_file


def main(argv=None):
    args = parse_args(argv)
    check_inputs(SHP_PATHS)
    wdpa = load_wdpa(SHP_PATHS)
    mpas = filter_marine_ia_ib(wdpa)
    if mpas.empty:
        raise SystemExit(
            "No marine Ia/Ib MPAs matched the filter — refusing to write an empty "
            "GeoPackage. Check the WDPA MARINE/IUCN_CAT values."
        )
    out_file = write_gpkg_atomic(mpas, args.out)
    print(f"Saved {len(mpas)} filtered polygons -> {out_file}")

    if not args.no_index:
        rows = mpa_index_rows(mpas.drop(columns="geometry").to_dict("records"))
        index_file = write_index_atomic(rows, args.index_out)
        dropped = len(mpas) - len(rows)
        note = f" ({dropped} dropped for a missing WDPA_PID)" if dropped else ""
        print(f"Saved {len(rows)} MPA index rows{note} -> {index_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
