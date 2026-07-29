import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from dotenv import load_dotenv

# The objective of this script (pipeline stage 2 - "scan"):
# - Read a sat_fetch.py run directory (manifest.json + georeferenced RGB chips)
# - Run the trained YOLOv8 boat detector on each chip
# - Map every box from chip pixels -> scene UTM -> WGS84 (each chip GeoTIFF is
#   self-describing, so rasterio gives us its CRS + affine directly)
# - Deduplicate boats that appear in the 64 px chip overlaps by merging detections
#   whose centroids are within --merge-dist METRES (IoU is numerically unstable for
#   1-5 px vessels, so we merge in metric space, not pixel/degree space)
# - Optionally keep only detections inside the MPA polygon (point-in-polygon)
# - Write detections.json (+ detections.geojson for QGIS) into the run dir
#
# The model is the in-repo YOLOv8n boat detector (data/training/weights/best.pt), single class
# 'boat', trained + deployed at imgsz 1024 (upscaling the 640 chips is the dominant small-object
# recall lever). Domain-gap caveat: base weights came from oblique photos, not top-down 10 m
# imagery, so first-pass detections are CANDIDATES to review in stage 3, not answers.
# See docs/PIPELINE.md and docs/STATUS.md.

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_GPKG = REPO_ROOT / "data" / "processed" / "wdpa_marine_ia_ib.gpkg"
# The in-repo promoted model — the active-learning loop trains and promotes here. (Originally the
# sibling C:\gtest\BoatDetection best.pt; AntiAngler is now canonical.)
DEFAULT_WEIGHTS = REPO_ROOT / "data" / "training" / "weights" / "best.pt"

load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")


def env_or(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def truthy(val):
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the YOLOv8 boat detector over a sat_fetch run dir and geolocate detections.")
    p.add_argument("--run", type=Path, default=env_or("DET_RUN", None),
                   help="A sat_fetch run directory (contains manifest.json + chips/).")
    p.add_argument("--weights", type=Path, default=Path(env_or("DET_WEIGHTS", str(DEFAULT_WEIGHTS))),
                   help="YOLOv8 .pt weights. Defaults to the in-repo promoted model "
                        "(data/training/weights/best.pt).")
    p.add_argument("--conf", type=float, default=float(env_or("DET_CONF", 0.25)),
                   help="Confidence threshold. Default 0.25 = the finetune-diverse model's recall-leaning "
                        "operating point (2026-07-26 sweep: recall up on all 4 holdouts vs the old 0.20 "
                        "deploy — Alonnisos wins outright, Kornati is statistical parity; stage 3 / "
                        "AIS-mismatch filters the extra FPs). Re-sweep with conf_sweep.py after each retrain.")
    p.add_argument("--imgsz", type=int, default=int(env_or("DET_IMGSZ", 1024)),
                   help="Model inference size. Default 1024: the promoted model is trained at 1024, and "
                        "upscaling the 640 chips (1-5 px vessels -> 2-10 px) is the dominant recall lever.")
    p.add_argument("--augment", action="store_true", default=truthy(env_or("DET_AUGMENT", "")),
                   help="Test-time augmentation (multi-scale + flips). BIG-SHIP AOIs ONLY: with --imgsz 1280 "
                        "it lifts large-ship recall (Port Said 0.69->0.81) at unchanged precision/FP. Do NOT use "
                        "on small-vessel/cluttered scenes - it amplifies water-texture false alarms (Alonnisos "
                        "precision 0.40->0.34, Singapore FP/chip 2.5->5.75). ~4x slower. See docs/STATUS.md.")
    p.add_argument("--merge-dist", type=float, default=float(env_or("DET_MERGE_DIST", 30.0)),
                   help="Merge detections whose centroids are within this many METRES (cross-chip dedup).")
    p.add_argument("--inside-mpa-only", action="store_true", default=truthy(env_or("DET_INSIDE_MPA_ONLY", "")),
                   help="Drop detections whose centroid falls outside the MPA polygon (WDPA runs only).")
    p.add_argument("--water-only", action="store_true", default=truthy(env_or("DET_WATER_ONLY", "")),
                   help="Drop detections on land (coast/rocks/islets) using the run's cached water_mask.tif "
                        "(build it with scripts/water_mask.py). Improvement plan O1 — kills the confident "
                        "rock/islet false positives at ~0 recall cost. On-water FPs need hard negatives, not this.")
    p.add_argument("--open-water", action="store_true", default=truthy(env_or("DET_OPEN_WATER", "")),
                   help="Drop detections whose whole box is water (min NDWI >= --ndwi-thresh) using the run's "
                        "cached ndwi.tif (build it with scripts/ndwi_mask.py). Improvement plan O2. Distinct "
                        "from --water-only (that drops LAND). NOTE - CALIBRATED NON-WIN on small-vessel scenes "
                        "(Alonnisos 2026-07-18): the on-water FPs sit on glint/foam/swell that reflect NIR "
                        "MORE than the tiny real boats, so no threshold drops FPs without killing recall — "
                        "leave OFF unless you have per-scene reviewed verdicts showing it separates. See "
                        "ndwi_mask.py. Every detection still gets a min_ndwi diagnostic field regardless.")
    p.add_argument("--ndwi-thresh", type=float, default=float(env_or("DET_NDWI_THRESH", 0.2)),
                   help="Water threshold for --open-water: a pixel with NDWI >= this counts as water, and a "
                        "detection is dropped only when its minimum NDWI is still >= this (uniformly water). "
                        "Higher = more conservative (fewer drops). Default 0.2; calibrate on the holdouts.")
    p.add_argument("--gpkg", type=Path, default=Path(env_or("DET_GPKG", str(DEFAULT_GPKG))),
                   help="MPA GeoPackage for the point-in-polygon test.")
    p.add_argument("--class-name", default=env_or("DET_CLASS_NAME", "boat"),
                   help="Label written for each detection.")
    p.add_argument("--device", default=env_or("DET_DEVICE", None),
                   help="Torch device passed to Ultralytics (e.g. 'cpu', '0'). Default: auto.")
    return p.parse_args()


def load_manifest(run_dir):
    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        raise RuntimeError(f"No manifest.json in {run_dir} - is this a sat_fetch run dir?")
    return json.loads(mpath.read_text())


def load_model(weights, class_name):
    if not Path(weights).exists():
        raise RuntimeError(f"Weights not found: {weights}. Point --weights at a YOLOv8 .pt — "
                           "the promoted models live in data/training/weights/ (best.pt optical, "
                           "best_sar.pt for Sentinel-1 runs); train one with scripts/train.py.")
    from ultralytics import YOLO  # heavy dep (pulls torch); imported lazily
    return YOLO(str(weights))


def detect_chip(model, chip_path, conf, imgsz, device, augment=False):
    """Run the detector on one chip. Returns (list of {conf, bbox_pixel, _utm, centroid_wgs84,
    bbox_wgs84}, ok) where _utm is the centroid in the chip's UTM CRS (for metric dedup)."""
    from PIL import Image  # Ultralytics reads a PIL image as RGB (a numpy array would be read as BGR)

    with rasterio.open(chip_path) as src:
        rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0))  # (H, W, 3) uint8, true RGB
        transform, crs = src.transform, src.crs

    res = model.predict(Image.fromarray(rgb, "RGB"), imgsz=imgsz, conf=conf,
                        augment=augment, device=device, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
    confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.empty((0,))
    if len(boxes) == 0:
        return []

    # Map each box: chip pixels -> UTM (via the chip's own affine) -> WGS84.
    dets = []
    us, vs = [], []          # UTM coords to batch-project: [c, minx, miny, maxx, maxy] per box
    for (x1, y1, x2, y2) in boxes:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        (ucx, ucy) = transform * (cx, cy)
        (u1, v1) = transform * (x1, y1)
        (u2, v2) = transform * (x2, y2)
        us += [ucx, u1, u2]
        vs += [ucy, v1, v2]
        dets.append({"bbox_pixel": [float(x1), float(y1), float(x2), float(y2)], "_utm": (ucx, ucy)})
    lons, lats = warp_transform(crs, "EPSG:4326", us, vs)
    for i, d in enumerate(dets):
        clon, clat = lons[3 * i], lats[3 * i]
        blon = [lons[3 * i + 1], lons[3 * i + 2]]
        blat = [lats[3 * i + 1], lats[3 * i + 2]]
        d["confidence"] = round(float(confs[i]), 4)
        d["centroid_wgs84"] = [round(clon, 7), round(clat, 7)]
        d["bbox_wgs84"] = [round(min(blon), 7), round(min(blat), 7),
                           round(max(blon), 7), round(max(blat), 7)]
    return dets


def dedupe(dets, merge_dist):
    """Greedy metric dedup: highest-confidence first, drop any later detection whose UTM centroid is
    within merge_dist metres of one already kept (the boat straddled a chip overlap seam)."""
    kept = []
    d2 = merge_dist * merge_dist
    for d in sorted(dets, key=lambda x: x["confidence"], reverse=True):
        x, y = d["_utm"]
        if all((x - k["_utm"][0]) ** 2 + (y - k["_utm"][1]) ** 2 > d2 for k in kept):
            kept.append(d)
    return kept


def load_mpa_polygon(gpkg, wdpaid):
    import geopandas as gpd
    if not Path(gpkg).exists():
        raise RuntimeError(f"GeoPackage not found: {gpkg}")
    gdf = gpd.read_file(gpkg)
    sel = gdf[gdf["WDPAID"] == float(wdpaid)]
    if sel.empty:
        return None
    return sel.geometry.union_all() if hasattr(sel.geometry, "union_all") else sel.geometry.unary_union


def write_atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def build_geojson(detections):
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": d["centroid_wgs84"]},
             "properties": {"detection_id": d["detection_id"], "chip": d["chip"],
                            "confidence": d["confidence"], "inside_mpa": d["inside_mpa"],
                            "on_water": d.get("on_water"), "min_ndwi": d.get("min_ndwi")}}
            for d in detections
        ],
    }


def main():
    args = parse_args()
    if args.run is None:
        raise RuntimeError("Provide a run dir: --run data/raw/sentinel2/<run>/")
    run_dir = args.run.resolve()
    manifest = load_manifest(run_dir)

    print("DETECT:")
    print(f"  run={run_dir}")
    print(f"  weights={args.weights}")
    print(f"  conf={args.conf}  imgsz={args.imgsz}  augment={args.augment}  merge-dist={args.merge_dist}m")
    print(f"  chips={len(manifest['chips'])}  scene_crs={manifest['scene_crs']}")

    model = load_model(args.weights, args.class_name)

    all_dets = []
    for chip in manifest["chips"]:
        chip_dets = detect_chip(model, run_dir / chip["filename"], args.conf, args.imgsz,
                                args.device, augment=args.augment)
        for d in chip_dets:
            d["chip"] = chip["filename"]
            d["class"] = args.class_name
        all_dets.extend(chip_dets)
    print(f"  raw detections: {len(all_dets)}")

    deduped = dedupe(all_dets, args.merge_dist)
    print(f"  after cross-chip dedup: {len(deduped)}")

    # Point-in-MPA (WDPA runs only). Always tag inside_mpa; --inside-mpa-only filters.
    wdpa = manifest.get("wdpa")
    poly = None
    if wdpa and wdpa.get("WDPAID") is not None:
        poly = load_mpa_polygon(args.gpkg, wdpa["WDPAID"])
    if poly is not None:
        from shapely.geometry import Point
        for d in deduped:
            d["inside_mpa"] = bool(poly.contains(Point(*d["centroid_wgs84"])))
        if args.inside_mpa_only:
            before = len(deduped)
            deduped = [d for d in deduped if d["inside_mpa"]]
            print(f"  inside-MPA filter: {len(deduped)}/{before} kept")
    else:
        for d in deduped:
            d["inside_mpa"] = None

    # Water/land mask (improvement plan O1). The cached water_mask.tif (scripts/water_mask.py,
    # ESA WorldCover 10 m) marks each detection on_water; --water-only drops land ones — the
    # confident coast/rock/islet FP class. NOTE: Step 0 (2026-07-16) found land is only ~16% of
    # FPs; the dominant ~84% are ON-WATER false alarms (empty-water/swell/shallows on small-vessel
    # scenes) which this filter does NOT touch — those need same-region hard negatives (O3).
    mask_path = run_dir / "water_mask.tif"
    if mask_path.exists():
        from water_mask import WaterSampler
        sampler = WaterSampler(mask_path)
        for d in deduped:
            lon, lat = d["centroid_wgs84"]
            d["on_water"] = sampler.is_water(lon, lat)
        if args.water_only:
            before = len(deduped)
            deduped = [d for d in deduped if d["on_water"]]
            print(f"  water-only filter: {len(deduped)}/{before} kept (dropped {before - len(deduped)} on land)")
    else:
        for d in deduped:
            d["on_water"] = None
        if args.water_only:
            print(f"  water-only requested but no water_mask.tif in {run_dir.name} "
                  f"(build it: python scripts/water_mask.py --run {run_dir}) — skipping filter")

    # NDWI open-water filter (improvement plan O2). The cached ndwi.tif (scripts/ndwi_mask.py, from the
    # scene's own B03/B08) tags each detection with the minimum NDWI in its box; --open-water drops the
    # ones whose box is uniformly water (min NDWI >= --ndwi-thresh) — the on-WATER false-alarm class that
    # --water-only (land) does not touch. A box containing any non-water pixel is kept (~no recall cost).
    ndwi_path = run_dir / "ndwi.tif"
    if ndwi_path.exists():
        from ndwi_mask import NdwiSampler
        nd = NdwiSampler(ndwi_path)
        for d in deduped:
            d["min_ndwi"] = nd.min_ndwi_bbox(d["bbox_wgs84"])
        if args.open_water:
            before = len(deduped)
            deduped = [d for d in deduped if not nd.is_open_water_fp(d["bbox_wgs84"], args.ndwi_thresh)]
            print(f"  open-water NDWI filter: {len(deduped)}/{before} kept "
                  f"(dropped {before - len(deduped)} on pure water, thresh {args.ndwi_thresh:+g})")
    else:
        for d in deduped:
            d["min_ndwi"] = None
        if args.open_water:
            print(f"  open-water requested but no ndwi.tif in {run_dir.name} "
                  f"(build it: python scripts/ndwi_mask.py --run {run_dir}) — skipping filter")

    deduped.sort(key=lambda d: d["confidence"], reverse=True)
    for i, d in enumerate(deduped):
        d["detection_id"] = f"det_{i:05d}"
        d.pop("_utm", None)

    chosen = manifest.get("chosen_item", {})
    detections_doc = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": run_dir.name,
        "weights": str(args.weights),
        "model": "yolov8",
        "conf_threshold": args.conf,
        "merge_dist_m": args.merge_dist,
        "scene_crs": manifest["scene_crs"],
        "scene": {"item_id": chosen.get("id"), "datetime": chosen.get("datetime"),
                  "eo:cloud_cover": chosen.get("eo:cloud_cover"), "aoi_cloud": chosen.get("aoi_cloud")},
        "wdpa": wdpa,
        "count": len(deduped),
        "detections": deduped,
    }
    write_atomic(run_dir / "detections.json", json.dumps(detections_doc, indent=2))
    write_atomic(run_dir / "detections.geojson", json.dumps(build_geojson(deduped), indent=2))
    print(f"\nWrote {len(deduped)} detections -> {run_dir / 'detections.json'} (+ .geojson)")


if __name__ == "__main__":
    main()
