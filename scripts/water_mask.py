"""Land/water mask for a sat_fetch run — the precision filter (improvement plan O1 / SAR S3).

Detections on land (coastline, surf, buildings, bright bare ground) are the dominant optical
false-positive class (see docs/IMPROVEMENT_PLAN.md). This builds a 10 m land/water mask over a
run's AOI from **ESA WorldCover** (free on the same Planetary Computer STAC as the imagery, no
credentials — class 80 = permanent water) and caches it as `water_mask.tif` in the run dir, so
`detect_boats.py` can drop land detections offline (no repeated network call).

Shared by BOTH detectors: the optical `best.pt` and the SAR `best_sar.pt` (SAR has the same
bright-coast clutter problem). Standalone:

    python scripts/water_mask.py --run data/raw/sentinel2/<run>/ [--insecure-tls]

Importable: `build_water_mask(run_dir, ...)` writes the cache; `sample_water(mask_path, lons, lats)`
returns a per-point water bool (out-of-mask -> water, the open-ocean default, so we never drop a
detection for lack of coverage).
"""
import argparse
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge

SCRIPT_DIR = Path(__file__).resolve().parent
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
WATER_CLASS = 80   # ESA WorldCover: permanent water bodies
NODATA = 0         # treat as water (open ocean beyond classified land) — never drop for no-data
MASK_NAME = "water_mask.tif"


def _latest_worldcover_items(catalog, bbox):
    """WorldCover items covering bbox, filtered to the newest product version (2021 v200)."""
    items = list(catalog.search(collections=["esa-worldcover"], bbox=list(bbox)).items())
    if not items:
        raise RuntimeError(f"No esa-worldcover items over bbox={bbox}. Check lon/lat order / coverage.")
    def ver(it):
        return it.properties.get("esa_worldcover:product_version") or it.properties.get("start_datetime", "")
    newest = max(ver(it) for it in items)
    return [it for it in items if ver(it) == newest]


def build_water_mask(run_dir, insecure_tls=False, overwrite=False, stac_url=STAC_URL):
    """Fetch + cache a 1=water/0=land uint8 GeoTIFF (EPSG:4326) over the run's AOI. Returns its path."""
    import planetary_computer
    from pystac_client import Client
    from providers._raster import gdal_env

    run_dir = Path(run_dir)
    out = run_dir / MASK_NAME
    if out.exists() and not overwrite:
        print(f"  water mask exists (use --overwrite to refetch): {out}")
        return out

    import json
    manifest = json.loads((run_dir / "manifest.json").read_text())
    bbox = manifest.get("aoi_bbox_wgs84")
    if not bbox or len(bbox) != 4:
        raise RuntimeError(f"manifest.json has no aoi_bbox_wgs84: {run_dir}")

    catalog = Client.open(stac_url, modifier=planetary_computer.sign_inplace)
    items = _latest_worldcover_items(catalog, bbox)
    minlon, minlat, maxlon, maxlat = bbox
    with rasterio.Env(**gdal_env(insecure_tls)):
        srcs = [rasterio.open(it.assets["map"].href) for it in items]
        try:
            # WorldCover is EPSG:4326; merge mosaics the (possibly multiple) 3-deg tiles over the AOI.
            mosaic, out_tf = merge(srcs, bounds=(minlon, minlat, maxlon, maxlat))
        finally:
            for s in srcs:
                s.close()

    lulc = mosaic[0]
    water = ((lulc == WATER_CLASS) | (lulc == NODATA)).astype(np.uint8)
    profile = dict(driver="GTiff", height=water.shape[0], width=water.shape[1], count=1,
                   dtype="uint8", crs="EPSG:4326", transform=out_tf, compress="deflate", nodata=None)
    tmp = out.with_suffix(".tif.tmp")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(water, 1)
    tmp.replace(out)
    frac = 100.0 * water.mean()
    print(f"  wrote {out}  ({water.shape[1]}x{water.shape[0]} @10m, {frac:.1f}% water, "
          f"{len(items)} WorldCover tile(s))")
    return out


class WaterSampler:
    """Loads a cached water_mask.tif once and samples lon/lat points against it."""
    def __init__(self, mask_path):
        with rasterio.open(mask_path) as src:
            self.arr = src.read(1)
            self.inv = ~src.transform
            self.h, self.w = self.arr.shape

    def is_water(self, lon, lat):
        c, r = self.inv * (lon, lat)
        c, r = int(c), int(r)
        if 0 <= r < self.h and 0 <= c < self.w:
            return bool(self.arr[r, c])
        return True  # outside the AOI mask -> open-ocean default, keep the detection


def sample_water(mask_path, lons, lats):
    s = WaterSampler(mask_path)
    return [s.is_water(lo, la) for lo, la in zip(lons, lats)]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build/cache a 10 m land/water mask for a sat_fetch run dir.")
    p.add_argument("--run", type=Path, required=True, help="A sat_fetch run directory (has manifest.json).")
    p.add_argument("--insecure-tls", action="store_true",
                   default=str(os.getenv("SAT_INSECURE_TLS", "")).strip().lower() in ("1", "true", "yes", "on"),
                   help="Skip GDAL TLS verification on the COG read (TLS-intercepting proxy/AV only).")
    p.add_argument("--overwrite", action="store_true", help="Refetch even if water_mask.tif exists.")
    p.add_argument("--stac-url", default=STAC_URL)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.insecure_tls:
        print("WARNING: --insecure-tls set - GDAL will NOT verify TLS certs on the WorldCover read.")
    build_water_mask(args.run.resolve(), args.insecure_tls, args.overwrite, args.stac_url)


if __name__ == "__main__":
    raise SystemExit(main())
