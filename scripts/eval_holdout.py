"""Evaluate boat-detector weights on the scene-aware satellite holdout + FP probes.

Standard YOLO `val` reports IoU-based mAP, which is (a) dominated by the ~600 overhead base
photos and (b) unstable for 1-5 px vessels — a 1 px shift flips a true positive to a false
positive, so mAP50 on a ~26-box test set is mostly noise. This script instead reports
tiny-object-appropriate metrics on the held-out SATELLITE scene:

  * predictions matched to ground truth by CENTER-DISTANCE (not IoU>=0.5),
  * instance-level Precision / Recall at a fixed conf, and false-positives-per-chip,
  * recall STRATIFIED by ground-truth pixel size (1-2 / 3-5 / >5 px), and
  * a bootstrap confidence interval over the few positive chips.

It also keeps the old IoU-mAP line (secondary, for continuity) and a raw-detection-count
false-positive probe on mostly-empty scenes.

Usage:
  venv/Scripts/python.exe scripts/eval_holdout.py \
      --old data/training/weights/best.pt \
      --new data/training/runs/<run>/weights/best.pt

Both default to the promoted weights if omitted. FP-probe scenes are in-sample if that scene
is in TRAIN -> read those as directional. The holdout metrics are clean iff `--holdout` names
a scene that build_dataset's scene-aware reassignment keeps in TEST and out of TRAIN.
"""
import argparse
import random
import shutil
import statistics
import tempfile
from pathlib import Path

from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "data" / "training"
FULL_YML = TRAIN / "config.yml"
PROMOTED = TRAIN / "weights" / "best.pt"
TILE = 640  # export chip size; GT labels are normalized to this


def build_subset(dst, src_split, name_filter):
    """Copy AL chips (name contains `name_filter`) from images/<src_split> into a val-only set."""
    d = Path(dst)
    (d / "images" / "val").mkdir(parents=True, exist_ok=True)
    (d / "labels" / "val").mkdir(parents=True, exist_ok=True)
    imgs = [p for p in (TRAIN / "images" / src_split).glob("*__*.png") if name_filter in p.name]
    boxes = 0
    for img in imgs:
        shutil.copy2(img, d / "images" / "val" / img.name)
        lbl = TRAIN / "labels" / src_split / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, d / "labels" / "val" / lbl.name)
            boxes += len([l for l in lbl.read_text().splitlines() if l.strip()])
    (d / "data.yml").write_text(
        f"path: {d.resolve().as_posix()}\n"
        "train: images/val\nval: images/val\ntest: images/val\n\nnc: 1\nnames: ['boat']\n")
    return d / "data.yml", imgs, boxes


def gt_boxes(label_path):
    """-> list of (cx_px, cy_px, size_px) for one chip's YOLO label file."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = (float(v) for v in p[1:5])
        out.append((cx * TILE, cy * TILE, max(w, h) * TILE))
    return out


def pred_centers(model, img_path, conf, imgsz):
    """-> list of (cx_px, cy_px) predicted box centers for one chip.

    imgsz is the INFERENCE size (may exceed the chip's native TILE px — YOLO upscales, which
    helps 1-5 px vessels). Predictions come back in the chip's own pixel space regardless, so
    matching against GT (normalized to TILE) stays correct."""
    r = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, verbose=False)[0]
    xywh = r.boxes.xywh.cpu().numpy() if len(r.boxes) else []
    return [(float(b[0]), float(b[1])) for b in xywh]


def match(gts, preds, tol):
    """Greedy nearest-center matching. Returns (tp_gt_idx set, matched_pred count)."""
    pairs = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(preds):
            d = ((g[0] - p[0]) ** 2 + (g[1] - p[1]) ** 2) ** 0.5
            if d <= tol:
                pairs.append((d, gi, pi))
    pairs.sort()
    used_g, used_p = set(), set()
    for _, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
    return used_g, used_p


def instance_eval(model, imgs, conf, tol, imgsz):
    """Per-chip TP/FP/FN + size-stratified GT hit/miss. Returns a dict of aggregates + per-chip list."""
    bins = {"1-2px": [0, 0], "3-5px": [0, 0], ">5px": [0, 0]}  # [hit, total]

    def size_bin(s):
        return "1-2px" if s <= 2 else ("3-5px" if s <= 5 else ">5px")

    per_chip = []
    for img in imgs:
        gts = gt_boxes(label_for(img))
        preds = pred_centers(model, img, conf, imgsz)
        used_g, used_p = match(gts, preds, tol)
        tp, fp, fn = len(used_g), len(preds) - len(used_p), len(gts) - len(used_g)
        per_chip.append((tp, fp, fn))
        for gi, g in enumerate(gts):
            b = size_bin(g[2])
            bins[b][1] += 1
            if gi in used_g:
                bins[b][0] += 1
    TP = sum(c[0] for c in per_chip)
    FP = sum(c[1] for c in per_chip)
    FN = sum(c[2] for c in per_chip)
    P = TP / (TP + FP) if (TP + FP) else 0.0
    R = TP / (TP + FN) if (TP + FN) else 0.0
    return {"P": P, "R": R, "TP": TP, "FP": FP, "FN": FN,
            "fp_per_chip": FP / len(imgs) if imgs else 0.0,
            "bins": bins, "per_chip": per_chip, "n_chips": len(imgs)}


def bootstrap_recall(per_chip, n=1000, seed=42):
    """5th/95th percentile CI of aggregate recall, resampling chips with replacement."""
    rng = random.Random(seed)
    k = len(per_chip)
    if k == 0:
        return (0.0, 0.0)
    samples = []
    for _ in range(n):
        pick = [per_chip[rng.randrange(k)] for _ in range(k)]
        tp = sum(c[0] for c in pick)
        fn = sum(c[2] for c in pick)
        samples.append(tp / (tp + fn) if (tp + fn) else 0.0)
    samples.sort()
    return (samples[int(0.05 * n)], samples[int(0.95 * n)])


def label_for(img_path):
    """Map an image path to its YOLO label by swapping the '/images/' segment for '/labels/'.
    Works for the original split dirs (images/test -> labels/test) regardless of split name."""
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def iou_map(weights, data, imgsz, split="val"):
    r = YOLO(str(weights)).val(data=str(data), split=split, imgsz=imgsz,
                               verbose=False, plots=False, save_json=False)
    return r.box.map50, r.box.mr


def det_count(weights, img_dir, conf, imgsz):
    m = YOLO(str(weights))
    return sum(len(m.predict(source=str(p), imgsz=imgsz, conf=conf, verbose=False)[0].boxes)
               for p in sorted(Path(img_dir).glob("*.png")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", type=Path, default=PROMOTED)
    ap.add_argument("--new", type=Path, default=PROMOTED)
    ap.add_argument("--holdout", default="fallbacktest", help="Name filter for the held-out TEST scene.")
    ap.add_argument("--fp-scenes", nargs="*", default=["gbr-cairns-reefs", "testrun"],
                    help="TRAIN-split scene filters to count raw detections on (FP probe).")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--match-dist", type=float, default=8.0,
                    help="Center-distance px tolerance for a pred<->GT match (default 8 px = 80 m @10m GSD).")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--imgsz", type=int, default=TILE,
                    help="Default inference size for both models (default 640 = chip native). "
                         "Raise (e.g. 1024) to evaluate a model trained at that size, or to probe "
                         "test-time upscaling. Chips stay 640 px on disk; YOLO scales predictions back.")
    ap.add_argument("--old-imgsz", type=int, default=None, help="Override inference size for --old (default: --imgsz).")
    ap.add_argument("--new-imgsz", type=int, default=None, help="Override inference size for --new (default: --imgsz).")
    args = ap.parse_args()
    old_imgsz = args.old_imgsz if args.old_imgsz is not None else args.imgsz
    new_imgsz = args.new_imgsz if args.new_imgsz is not None else args.imgsz

    pairs = [(t, w, sz) for t, w, sz in
             [("OLD", args.old, old_imgsz), ("NEW", args.new, new_imgsz)] if w.exists()]
    if not pairs:
        raise SystemExit("no weights found to evaluate")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sat_yml, sat_imgs, sat_boxes = build_subset(tmp / "sat", "test", args.holdout)
        fp_dirs = {}
        for f in args.fp_scenes:
            _, imgs, _ = build_subset(tmp / f"fp_{f}", "train", f)
            fp_dirs[f] = (tmp / f"fp_{f}" / "images" / "val", len(imgs))
        print(f"satellite holdout '{args.holdout}': {len(sat_imgs)} chips, {sat_boxes} boxes"
              f"  (match tol {args.match_dist:.0f}px, conf {args.conf})")
        for f, (_, n) in fp_dirs.items():
            print(f"FP probe '{f}': {n} chips (in-sample if in TRAIN -> directional)")
        print(f"inference imgsz: OLD={old_imgsz}  NEW={new_imgsz}"
              + ("  (test-time upscaling in play)" if max(old_imgsz, new_imgsz) > TILE else ""))

        rows = []
        for tag, w, sz in pairs:
            model = YOLO(str(w))
            ie = instance_eval(model, sat_imgs, args.conf, args.match_dist, sz)
            ci = bootstrap_recall(ie["per_chip"], args.bootstrap)
            m50, _ = iou_map(w, sat_yml, sz)
            base50, baseR = iou_map(w, FULL_YML, sz, "test")
            fps = {f: det_count(w, d, args.conf, sz) for f, (d, _) in fp_dirs.items()}
            rows.append((tag, w, ie, ci, m50, base50, baseR, fps, sz))

        print("\n================ HOLDOUT — instance-level (center-distance matching) ================")
        print(f"{'weights':<7}{'P':>7}{'R':>7}{'TP':>5}{'FP':>5}{'FN':>5}{'FP/chip':>9}{'recall 90% CI':>18}")
        for tag, w, ie, ci, *_ in rows:
            print(f"{tag:<7}{ie['P']:>7.3f}{ie['R']:>7.3f}{ie['TP']:>5}{ie['FP']:>5}{ie['FN']:>5}"
                  f"{ie['fp_per_chip']:>9.2f}{f'[{ci[0]:.2f}, {ci[1]:.2f}]':>18}")

        print("\nrecall stratified by GT pixel size (hit/total):")
        print(f"{'weights':<7}{'1-2px':>12}{'3-5px':>12}{'>5px':>12}")
        for tag, w, ie, *_ in rows:
            b = ie["bins"]
            def cell(k):
                hit, tot = b[k]
                return f"{hit}/{tot}" + (f" ({hit/tot:.2f})" if tot else "")
            print(f"{tag:<7}{cell('1-2px'):>12}{cell('3-5px'):>12}{cell('>5px'):>12}")

        print("\nsecondary IoU-mAP (unstable at this scale — context only):")
        print(f"{'weights':<7}{'imgsz':>7}{'holdout mAP50':>14}{'base mAP50':>12}{'base R':>9}")
        for tag, w, ie, ci, m50, base50, baseR, fps, sz in rows:
            print(f"{tag:<7}{sz:>7}{m50:>14.3f}{base50:>12.3f}{baseR:>9.3f}")

        print(f"\nfalse-positive probe — raw detections @conf{args.conf} (lower is better):")
        for tag, w, ie, ci, m50, base50, baseR, fps, sz in rows:
            print(f"  {tag:<7} " + "   ".join(f"{f}: {c}" for f, c in fps.items()))
        for tag, w, *_ in rows:
            print(f"  {tag} = {w}")


if __name__ == "__main__":
    main()
