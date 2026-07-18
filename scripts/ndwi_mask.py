"""NDWI open-water false-positive filter for a sat_fetch run — improvement plan O2 (open-ocean scope).

The dominant optical false-positive class over open ocean is ON-WATER false alarms (Step 0, 2026-07-16:
~84% of all FPs) — the detector firing on swell lines, sun-glint, current boundaries and textured water
where there is no vessel. NIR separates them physically: water absorbs NIR strongly (dark → high NDWI),
while a hull / wake reflects NIR (NDWI dips). So NDWI tells "an object is here" from "just water".

This re-reads the run's chosen scene's Green (B03) + NIR (B08) over the SAME AOI/resolution as the RGB
chips (through the same `providers/pc.py` `read_aoi`, so the result is pixel-aligned with the chip grid),
computes NDWI = (green - nir) / (green + nir), and caches it as `ndwi.tif` (float32, scene UTM) in the run
dir. `detect_boats.py --open-water` then samples each detection's box and drops the ones that sit on pure
water — offline, no repeated network call.

This is NOT a land mask — that is `water_mask.py` / `--water-only`, which drops detections on coast/rock.
This targets on-WATER false alarms, the open-ocean precision problem.

⚠️ CALIBRATION RESULT (2026-07-18, Alonnisos small-vessel holdout, 63 TP / 20 FP with human verdicts):
**the premise does NOT hold on small-vessel scenes — keep --open-water OFF there.** The on-water FPs are
not on *blank* water; they sit on glint / foam / swell / reef-edge features that reflect NIR strongly
(FP min-NDWI median -0.122), while the 1-5 px real boats are sub-pixel-mixed with water (TP min-NDWI
median -0.027 — i.e. *weaker* NIR contrast than the FPs). So no threshold separates them: dropping FPs
also drops the real boats (at -0.10 → 7/20 FP but 60/63 TP gone, precision 0.76→0.19). The tool is kept
(correct, tested, opt-in, and `min_ndwi` is a useful per-detection diagnostic) but it is a *validated
non-win* for small-vessel precision — that lever stays O3 (same-region hard negatives) + O7 (AIS fusion).
It may still help on big-ship/anchorage scenes where FPs are genuine blank-water hits, but big-ship
precision is already ~0.97 so the ceiling there is small. **Always calibrate per-scene against reviewed
verdicts before enabling.** See docs/IMPROVEMENT_PLAN.md O2.

Standalone:

    python scripts/ndwi_mask.py --run data/raw/sentinel2/<run>/ [--insecure-tls]

Importable: `build_ndwi(run_dir, ...)` writes the cache; `NdwiSampler(path)` samples a detection's WGS84
bbox and returns the MINIMUM NDWI within it (the most vessel-like pixel). A box whose minimum NDWI is
still >= the water threshold is uniformly water → an on-water FP; any box containing a non-water pixel is
kept, so the filter costs ~no recall when a real (even faint) target is present.

Reflectance note: NDWI is a normalized ratio, so a common multiplicative scale cancels; but the S2
processing-baseline >= 04.00 additive BOA offset does not cancel in the denominator, so absolute NDWI can
be biased scene-to-scene. The water-vs-object *separation* still holds (objects reflect more NIR than
water), so the threshold is meant to be calibrated on the holdouts (conf_sweep-style), not trusted as a
physical constant. Default 0.2 is conservative (drops only boxes that are confidently, uniformly water).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

SCRIPT_DIR = Path(__file__).resolve().parent
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
MASK_NAME = "ndwi.tif"
DEFAULT_WATER_THRESH = 0.2   # a pixel with NDWI >= this is "water"; see the reflectance note above


def build_ndwi(run_dir, insecure_tls=False, overwrite=False, stac_url=STAC_URL):
    """Fetch + cache a float32 NDWI GeoTIFF (scene UTM, pixel-aligned with the chips) over the run's AOI.

    Returns the cache path. Reuses the run's chosen scene (manifest ``chosen_item.id``) and the same
    ``read_aoi`` the chips came from, so ``ndwi.tif`` lines up with the chip grid one-to-one.
    """
    from providers.pc import PlanetaryComputerS2Provider

    run_dir = Path(run_dir)
    out = run_dir / MASK_NAME
    if out.exists() and not overwrite:
        print(f"  ndwi mask exists (use --overwrite to refetch): {out}")
        return out

    manifest = json.loads((run_dir / "manifest.json").read_text())
    bbox = manifest.get("aoi_bbox_wgs84")
    if not bbox or len(bbox) != 4:
        raise RuntimeError(f"manifest.json has no aoi_bbox_wgs84: {run_dir}")
    item_id = (manifest.get("chosen_item") or {}).get("id")
    if not item_id:
        raise RuntimeError(f"manifest.json has no chosen_item.id: {run_dir}")
    collection = manifest.get("collection", "sentinel-2-l2a")
    resolution = float(manifest.get("resolution_m") or 10.0)

    prov = PlanetaryComputerS2Provider()
    if collection != prov.default_collection:
        # NDWI needs S2 optical bands; a SAR run has no B03/B08.
        raise RuntimeError(f"NDWI needs a Sentinel-2 optical run; this run is collection={collection!r}.")

    catalog = prov.open_catalog(stac_url, None)
    items = list(catalog.search(collections=[collection], ids=[item_id]).items())
    if not items:
        raise RuntimeError(f"Scene {item_id} not found in {collection} — it may have aged out of the catalog.")
    item = items[0]

    # Same windowed read as the chips, but Green + NIR instead of RGB → pixel-aligned NDWI.
    stack, transform, crs, valid = prov.read_aoi(item, tuple(bbox), "B03,B08", resolution, insecure_tls)
    green = stack[0].astype(np.float32)
    nir = stack[1].astype(np.float32)
    denom = green + nir
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = np.where(denom > 0, (green - nir) / denom, np.nan).astype(np.float32)
    ndwi[~valid] = np.nan

    profile = dict(driver="GTiff", height=ndwi.shape[0], width=ndwi.shape[1], count=1,
                   dtype="float32", crs=crs, transform=transform, nodata=float("nan"),
                   compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    tmp = out.with_suffix(".tif.tmp")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(ndwi, 1)
    tmp.replace(out)

    finite = ndwi[np.isfinite(ndwi)]
    med = float(np.median(finite)) if finite.size else float("nan")
    print(f"  wrote {out}  ({ndwi.shape[1]}x{ndwi.shape[0]} @{resolution:g}m, "
          f"median NDWI {med:+.3f}, {100.0 * finite.size / ndwi.size:.0f}% valid)")
    return out


class NdwiSampler:
    """Loads a cached ndwi.tif once and samples detection boxes against it.

    All sampling is by WGS84 bbox (detections carry ``bbox_wgs84``), projected into the raster's UTM CRS.
    Returns the MINIMUM NDWI inside a box — the single most vessel-like (least-watery) pixel — so a box is
    judged an on-water FP only when *every* pixel in it is water.
    """

    def __init__(self, mask_path):
        with rasterio.open(mask_path) as src:
            self.arr = src.read(1)
            self.crs = src.crs
            self.inv = ~src.transform
            self.h, self.w = self.arr.shape

    def min_ndwi_bbox(self, bbox_wgs84):
        """Minimum finite NDWI inside the box, or None if the box has no valid coverage (→ never drop)."""
        minlon, minlat, maxlon, maxlat = bbox_wgs84
        xs, ys = warp_transform("EPSG:4326", self.crs,
                                [minlon, maxlon, minlon, maxlon],
                                [minlat, maxlat, maxlat, minlat])
        cols, rows = [], []
        for x, y in zip(xs, ys):
            c, r = self.inv * (x, y)
            cols.append(c)
            rows.append(r)
        c0, c1 = max(0, int(np.floor(min(cols)))), min(self.w, int(np.ceil(max(cols))) + 1)
        r0, r1 = max(0, int(np.floor(min(rows)))), min(self.h, int(np.ceil(max(rows))) + 1)
        if c0 >= c1 or r0 >= r1:
            return None  # box falls outside the NDWI raster → no evidence, keep the detection
        win = self.arr[r0:r1, c0:c1]
        finite = win[np.isfinite(win)]
        if finite.size == 0:
            return None
        return round(float(finite.min()), 4)

    def is_open_water_fp(self, bbox_wgs84, thresh=DEFAULT_WATER_THRESH):
        """True iff the whole box is water (min NDWI >= thresh) — i.e. a detection to drop under --open-water."""
        m = self.min_ndwi_bbox(bbox_wgs84)
        return m is not None and m >= thresh


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build/cache a 10 m NDWI raster for a sat_fetch run dir "
                                            "(open-water false-positive filter; detect_boats.py --open-water).")
    p.add_argument("--run", type=Path, required=True, help="A sat_fetch run directory (has manifest.json).")
    p.add_argument("--insecure-tls", action="store_true",
                   default=str(os.getenv("SAT_INSECURE_TLS", "")).strip().lower() in ("1", "true", "yes", "on"),
                   help="Skip GDAL TLS verification on the COG read (TLS-intercepting proxy/AV only).")
    p.add_argument("--overwrite", action="store_true", help="Refetch even if ndwi.tif exists.")
    p.add_argument("--stac-url", default=STAC_URL)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.insecure_tls:
        print("WARNING: --insecure-tls set - GDAL will NOT verify TLS certs on the B03/B08 read.")
    build_ndwi(args.run.resolve(), args.insecure_tls, args.overwrite, args.stac_url)


if __name__ == "__main__":
    raise SystemExit(main())
