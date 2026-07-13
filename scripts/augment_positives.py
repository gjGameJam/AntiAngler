"""Fix the negative-heavy training set by synthesizing positive chips — no new imagery/review.

The reviewed data is ~11 boat-bearing chips vs ~58 empty-water chips (strict MPAs are empty by
design), which biases the detector conservative and suppresses recall. This script attacks that
imbalance directly, on EXISTING labels, via water-gated copy-paste augmentation
(Ghiasi et al. copy-paste; Kisantal et al. small-object augmentation):

  * STAMPS: crop every labeled boat instance from the reviewed positive chips.
  * CANVASES: the reviewed hard-negative chips — i.e. the real MPA water/reef backgrounds we deploy on.
  * PASTE each boat (isolated by a brightness mask, so we paste the vessel, not its background square)
    onto water-gated locations (skip nodata + the brightest ~15% = foam/glint/sand/reef-crest), giving
    synthetic positives IN the deployment domain. Plus optional oversampling of the real positive chips.

Output is written as a new training-export tag so scripts/build_dataset.py layers it idempotently.

LEAKAGE GUARD: the held-out scene (default 'fallbacktest') is excluded from BOTH stamps and canvases,
so the eval holdout stays unseen. Synthetic chip names contain '__' (AL marker) and never the holdout
filter, so build_dataset's scene-aware reassignment routes them to TRAIN.

Usage:  venv/Scripts/python.exe scripts/augment_positives.py            # 80 synth chips, oversample real x2
        venv/Scripts/python.exe scripts/augment_positives.py --num 120 --oversample-real 3
"""
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
EXPORTS = REPO / "data" / "processed" / "training_exports"
TILE = 640


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exports", type=Path, default=EXPORTS, help="Training-exports root to read/write.")
    p.add_argument("--tag", default="augmented", help="Output tag subdir (accumulated into).")
    p.add_argument("--num", type=int, default=80, help="Number of synthetic positive chips to generate.")
    p.add_argument("--min-boats", type=int, default=1)
    p.add_argument("--max-boats", type=int, default=5)
    p.add_argument("--oversample-real", type=int, default=2,
                   help="Also copy each real positive chip N times (with random flips). 0 disables.")
    p.add_argument("--exclude", default="fallbacktest",
                   help="Scene name-filter to EXCLUDE from stamps+canvases (the eval holdout). Prevents leakage.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tile", type=int, default=TILE)
    return p.parse_args()


def read_pairs(exports, exclude, skip_tag):
    """Yield (img_path, label_path) for every export chip except the excluded scene and the output tag."""
    for tag_dir in sorted(exports.glob("*")):
        if not tag_dir.is_dir() or tag_dir.name == skip_tag:
            continue
        for img in sorted((tag_dir / "images").glob("*.png")):
            if exclude and exclude in img.name:
                continue
            yield img, tag_dir / "labels" / f"{img.stem}.txt"


def load_boxes(label_path, tile):
    """-> list of (x1,y1,x2,y2) pixel boxes."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = (float(v) * tile for v in p[1:5])
        out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def make_stamp(arr, box, pad=2):
    """Crop a boat patch + a brightness mask isolating the vessel from its water margin."""
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
    x2, y2 = min(w, int(round(x2)) + pad), min(h, int(round(y2)) + pad)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    patch = arr[y1:y2, x1:x2].astype(np.float32)
    gray = patch.mean(axis=2)
    # border ring = surrounding water; the vessel is what stands out brighter than it
    ring = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    thr = float(np.median(ring) + 1.0 * (np.std(ring) + 1e-3))
    mask = gray > thr
    if mask.sum() < 1:  # low-contrast vessel: fall back to the inner box
        mask = np.zeros_like(gray, dtype=bool)
        mask[pad:-pad or None, pad:-pad or None] = True
    return patch.astype(np.uint8), mask, (x2 - x1), (y2 - y1)


def water_valid_map(canvas):
    """Per-pixel bool: plausible open/lagoon water — not nodata, not the brightest ~15% (foam/sand/reef)."""
    gray = canvas.mean(axis=2)
    nonblack = gray > 6
    if nonblack.sum() == 0:
        return np.zeros_like(gray, dtype=bool)
    hi = np.percentile(gray[nonblack], 85)
    return nonblack & (gray < hi)


def try_place(valid, cx, cy, bw, bh):
    """True if the boat footprint at (cx,cy) sits mostly on valid water and within bounds."""
    h, w = valid.shape
    x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
    x2, y2 = x1 + bw, y1 + bh
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return False
    region = valid[y1:y2, x1:x2]
    return region.size > 0 and region.mean() >= 0.8


def paste(canvas, stamp, cx, cy):
    """Composite the masked vessel pixels of `stamp` onto `canvas` centered at (cx,cy). Returns pixel box."""
    patch, mask, bw, bh = stamp
    ph, pw = patch.shape[:2]
    x1, y1 = int(cx - pw / 2), int(cy - ph / 2)
    sub = canvas[y1:y1 + ph, x1:x1 + pw]
    sub[mask] = patch[mask]
    canvas[y1:y1 + ph, x1:x1 + pw] = sub
    return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def yolo_line(box, tile):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / tile, (y1 + y2) / 2 / tile
    w, h = (x2 - x1) / tile, (y2 - y1) / tile
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    out = args.exports / args.tag
    img_dir, lbl_dir = out / "images", out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    stamps, canvases, real_positives = [], [], []
    for img, lbl in read_pairs(args.exports, args.exclude, args.tag):
        arr = np.array(Image.open(img).convert("RGB"))
        boxes = load_boxes(lbl, args.tile)
        if boxes:
            real_positives.append((img, lbl, boxes))
            for b in boxes:
                s = make_stamp(arr, b)
                if s is not None:
                    stamps.append(s)
        else:
            canvases.append((img.name, arr))
    if not stamps:
        raise SystemExit("No boat stamps found (no positive chips outside the excluded scene).")
    if not canvases:
        raise SystemExit("No hard-negative canvases found.")
    print(f"stamps (boats): {len(stamps)}  |  canvases (empty chips): {len(canvases)}  |  "
          f"real positive chips: {len(real_positives)}  (excluded scene: '{args.exclude}')")

    n_synth = n_synth_boxes = 0
    for i in range(args.num):
        _, base = rng.choice(canvases)
        canvas = base.copy()
        valid = water_valid_map(canvas)
        k = rng.randint(args.min_boats, args.max_boats)
        placed = []
        for _ in range(k):
            stamp = rng.choice(stamps)
            _, _, bw, bh = stamp
            box = None
            for _try in range(40):
                cx = rng.randint(bw, args.tile - bw)
                cy = rng.randint(bh, args.tile - bh)
                if any(abs(cx - px) < max(bw, pbw) and abs(cy - py) < max(bh, pbh)
                       for (px, py, pbw, pbh) in placed):
                    continue
                if try_place(valid, cx, cy, stamp[0].shape[1], stamp[0].shape[0]):
                    box = paste(canvas, stamp, cx, cy)
                    placed.append((cx, cy, bw, bh))
                    break
            if box:
                n_synth_boxes += 1
        stem = f"aug-synth__{i:05d}"
        Image.fromarray(canvas, "RGB").save(img_dir / f"{stem}.png")
        lines = [yolo_line((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), args.tile)
                 for (cx, cy, bw, bh) in placed]
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_synth += 1

    n_over = 0
    for r in range(args.oversample_real):
        for (img, lbl, boxes) in real_positives:
            arr = np.array(Image.open(img).convert("RGB"))
            lines = [yolo_line(b, args.tile) for b in boxes]
            flip = rng.choice(["none", "h", "v"])
            if flip == "h":
                arr = arr[:, ::-1]
                lines = [_flip_line(l, "h") for l in lines]
            elif flip == "v":
                arr = arr[::-1, :]
                lines = [_flip_line(l, "v") for l in lines]
            stem = f"aug-over{r}__{img.stem}"
            Image.fromarray(np.ascontiguousarray(arr), "RGB").save(img_dir / f"{stem}.png")
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_over += 1

    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "synthetic_chips": n_synth, "synthetic_boxes": n_synth_boxes,
        "oversampled_real_chips": n_over, "stamps_used": len(stamps),
        "canvases_used": len(canvases), "excluded_scene": args.exclude, "seed": args.seed,
    }
    (out / "augment_log.json").write_text(json.dumps(summary, indent=2))
    print(f"WROTE -> {out}")
    print(f"  synthetic: {n_synth} chips / {n_synth_boxes} pasted boats")
    print(f"  oversampled real positives: {n_over} chips")
    print("Next: build_dataset.py (layers this tag) + re-apply scene-aware holdout, then train.py.")


def _flip_line(line, axis):
    p = line.split()
    cx, cy = float(p[1]), float(p[2])
    if axis == "h":
        cx = 1.0 - cx
    else:
        cy = 1.0 - cy
    return f"{p[0]} {cx:.6f} {cy:.6f} {p[3]} {p[4]}"


if __name__ == "__main__":
    main()
