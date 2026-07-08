import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from dotenv import load_dotenv

# The objective of this script (pipeline stage 3 - "refine"):
# - Read a sat_fetch run dir's manifest.json + reviews.json (written by review_server.py)
# - For every FULLY REVIEWED chip, emit a YOLO training example:
#     images/<run>__<chip>.png  +  labels/<run>__<chip>.txt
#   Label lines: one per box the human kept (verdict correct|missed), class 0, normalized
#   `cls cx cy w h`. Boxes marked incorrect are dropped. A reviewed chip with no kept boxes
#   becomes an EMPTY label file - a hard negative, which is what teaches the detector to stop
#   firing on whitecaps/rocks/cloud edges.
#
# Only chips the reviewer marked reviewed are exported: YOLO needs COMPLETE labels (every boat
# in a training image boxed), and the reviewed flag is that guarantee. Feed the output dir to
# BoatDetection/scripts/helpers/splitData.py, then fine-tune from best.pt. See docs/PIPELINE.md.

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "training_exports"

load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")


def env_or(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def parse_args():
    p = argparse.ArgumentParser(
        description="Export reviewed chips to a YOLO training set for detector fine-tuning.")
    p.add_argument("--run", type=Path, default=env_or("EXPORT_RUN", None),
                   help="A sat_fetch run dir with manifest.json + reviews.json.")
    p.add_argument("--out", type=Path, default=Path(env_or("EXPORT_OUT", str(DEFAULT_OUT))),
                   help="Root for training exports; a per-tag subdir is created under it.")
    p.add_argument("--tag", default=env_or("EXPORT_TAG", None),
                   help="Subdir name to accumulate into. Default: today's UTC date (YYYY-MM-DD).")
    p.add_argument("--class-id", type=int, default=int(env_or("EXPORT_CLASS_ID", 0)),
                   help="YOLO class id written for every kept box (BoatDetection is single-class 0).")
    return p.parse_args()


def chip_to_png(tif_path, png_path):
    with rasterio.open(tif_path) as src:
        arr = np.transpose(src.read([1, 2, 3]), (1, 2, 0))  # (H,W,3) uint8, pre-stretched
    Image.fromarray(arr, "RGB").save(png_path, format="PNG")


def to_yolo_line(box, tile, class_id):
    """Normalized `cls cx cy w h`, clamped to the tile so an edge-drawn box stays valid [0,1]."""
    x1, y1, x2, y2 = box["bbox_pixel"]
    x1, x2 = sorted((max(0.0, min(tile, x1)), max(0.0, min(tile, x2))))
    y1, y2 = sorted((max(0.0, min(tile, y1)), max(0.0, min(tile, y2))))
    cx, cy = (x1 + x2) / 2 / tile, (y1 + y2) / 2 / tile
    w, h = (x2 - x1) / tile, (y2 - y1) / tile
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def write_atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def main():
    args = parse_args()
    if args.run is None:
        raise RuntimeError("Provide a run dir: --run data/raw/sentinel2/<run>/")
    run_dir = args.run.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    reviews_path = run_dir / "reviews.json"
    if not reviews_path.exists():
        raise RuntimeError(f"No reviews.json in {run_dir} - review the run first (review_server.py).")
    reviews = json.loads(reviews_path.read_text())
    tile = int(manifest.get("tile_size", 640))

    tag = args.tag or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = args.out / tag
    img_dir, lbl_dir = out_dir / "images", out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    kept_verdicts = {"correct", "missed"}
    n_pos = n_neg = n_boxes = 0
    for filename, entry in reviews.get("chips", {}).items():
        if not entry.get("reviewed"):
            continue  # complete-labels rule: only fully-reviewed chips are trainable
        stem = f"{run_dir.name}__{Path(filename).stem}"
        boxes = [b for b in entry.get("boxes", []) if b.get("verdict") in kept_verdicts]
        chip_to_png(run_dir / filename, img_dir / f"{stem}.png")
        write_atomic(lbl_dir / f"{stem}.txt",
                     "\n".join(to_yolo_line(b, tile, args.class_id) for b in boxes) + ("\n" if boxes else ""))
        n_boxes += len(boxes)
        if boxes:
            n_pos += 1
        else:
            n_neg += 1

    total = n_pos + n_neg
    if total == 0:
        print(f"No reviewed chips in {reviews_path} - nothing exported. Mark chips reviewed first.")
        return

    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_run": run_dir.name, "tile_size": tile, "class_map": {str(args.class_id): "boat"},
        "reviewed_chips": total, "positive_chips": n_pos, "negative_chips": n_neg, "total_boxes": n_boxes,
    }
    # Append this run's provenance to a cumulative log so a tag dir that accumulates many runs
    # stays auditable (which runs contributed, when, how many boxes).
    log_path = out_dir / "export_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"exports": []}
    log["exports"] = [e for e in log["exports"] if e.get("source_run") != run_dir.name] + [summary]
    write_atomic(log_path, json.dumps(log, indent=2))

    print("EXPORT:")
    print(f"  source={run_dir.name}")
    print(f"  reviewed chips: {total}  (positive {n_pos}, negative {n_neg})  total boxes: {n_boxes}")
    print(f"  -> {out_dir}  (images/ + labels/)")
    print("\nNext: point BoatDetection/scripts/helpers/splitData.py at this dir, then fine-tune")
    print("      train.py from best.pt (see docs/PIPELINE.md).")


if __name__ == "__main__":
    main()
