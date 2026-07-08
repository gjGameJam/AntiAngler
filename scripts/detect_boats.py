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
# The model contract lives in C:\gtest\BoatDetection (YOLOv8n, single class 'boat',
# imgsz 640). Domain-gap caveat: it was trained on oblique photos, not top-down 10 m
# imagery, so first-pass detections are low-confidence CANDIDATES to review in stage 3,
# not answers. See docs/PIPELINE.md.

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_GPKG = REPO_ROOT / "data" / "processed" / "wdpa_marine_ia_ib.gpkg"
# BoatDetection is a sibling repo of AntiAngler (both under C:\gtest by default).
DEFAULT_WEIGHTS = REPO_ROOT.parent / "BoatDetection" / "runs" / "train" / "boat_detection" / "weights" / "best.pt"

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
                   help="YOLOv8 .pt weights. Defaults to the sibling BoatDetection best.pt.")
    p.add_argument("--conf", type=float, default=float(env_or("DET_CONF", 0.25)),
                   help="Confidence threshold. Keep low (~0.25) for recall; stage 3 filters false positives.")
    p.add_argument("--imgsz", type=int, default=int(env_or("DET_IMGSZ", 640)),
                   help="Model input size; must match the chip size (640).")
    p.add_argument("--merge-dist", type=float, default=float(env_or("DET_MERGE_DIST", 30.0)),
                   help="Merge detections whose centroids are within this many METRES (cross-chip dedup).")
    p.add_argument("--inside-mpa-only", action="store_true", default=truthy(env_or("DET_INSIDE_MPA_ONLY", "")),
                   help="Drop detections whose centroid falls outside the MPA polygon (WDPA runs only).")
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
        raise RuntimeError(f"Weights not found: {weights}. Point --weights at your YOLOv8 .pt "
                           "(see C:\\gtest\\BoatDetection).")
    from ultralytics import YOLO  # heavy dep (pulls torch); imported lazily
    return YOLO(str(weights))


def detect_chip(model, chip_path, conf, imgsz, device):
    """Run the detector on one chip. Returns (list of {conf, bbox_pixel, _utm, centroid_wgs84,
    bbox_wgs84}, ok) where _utm is the centroid in the chip's UTM CRS (for metric dedup)."""
    from PIL import Image  # Ultralytics reads a PIL image as RGB (a numpy array would be read as BGR)

    with rasterio.open(chip_path) as src:
        rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0))  # (H, W, 3) uint8, true RGB
        transform, crs = src.transform, src.crs

    res = model.predict(Image.fromarray(rgb, "RGB"), imgsz=imgsz, conf=conf,
                        device=device, verbose=False)[0]
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
                            "confidence": d["confidence"], "inside_mpa": d["inside_mpa"]}}
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
    print(f"  conf={args.conf}  imgsz={args.imgsz}  merge-dist={args.merge_dist}m")
    print(f"  chips={len(manifest['chips'])}  scene_crs={manifest['scene_crs']}")

    model = load_model(args.weights, args.class_name)

    all_dets = []
    for chip in manifest["chips"]:
        chip_dets = detect_chip(model, run_dir / chip["filename"], args.conf, args.imgsz, args.device)
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
