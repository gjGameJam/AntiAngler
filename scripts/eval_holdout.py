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

  # SAR: point --train-dir at the SAR tree; --old/--new then default to ITS promoted
  # checkpoint (best_sar.pt), never the optical one.
  venv/Scripts/python.exe scripts/eval_holdout.py \
      --train-dir data/training_sar --holdout s1deep-gibraltar --imgsz 1024

Both weights default to the promoted checkpoint registered for `--train-dir` (PROMOTED_BY_TREE).
FP-probe scenes are in-sample if that scene is in TRAIN -> read those as directional. The holdout
metrics are clean iff `--holdout` names a scene that build_dataset's scene-aware reassignment keeps
in TEST and out of TRAIN; a filter matching ZERO chips is a hard error, never a silent empty gate.

ultralytics/torch are imported LAZILY (inside the inference helpers) so this module stays importable
— and its matching/subset/CI logic unit-testable — in the offline suite, which pip-installs nothing.
"""
import argparse
import random
import shutil
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "data" / "training"       # the OPTICAL tree; --train-dir overrides it per run
FULL_YML = TRAIN / "config.yml"          # back-compat alias; main() derives this from --train-dir
PROMOTED = TRAIN / "weights" / "best.pt"  # back-compat alias
WEIGHTS = TRAIN / "weights"              # EVERY modality's promoted checkpoints live here
TILE = 640  # export chip size; GT labels are normalized to this

# Which promoted checkpoint gates which dataset tree. `--train-dir` selects the DATA, and both
# modalities' checkpoints sit in the same data/training/weights/ dir, so nothing on disk links a
# tree to its model. Without this table a bare SAR gate would fall back to the OPTICAL best.pt and
# print authoritative-looking garbage on SAR chips (wrong texture priors -> ~0 hits). Keys are
# repo-relative posix paths; add a row when a new modality tree appears.
PROMOTED_BY_TREE = {
    "data/training": "best.pt",          # optical (Sentinel-2)
    "data/training_sar": "best_sar.pt",  # SAR     (Sentinel-1)
}


def load_yolo():
    """Lazily resolve the ultralytics.YOLO class — heavy (pulls torch), and only inference needs it."""
    from ultralytics import YOLO  # noqa: PLC0415 - heavy (torch); keeps module import stdlib-only
    return YOLO


def promoted_weights(train_dir):
    """-> Path of the promoted checkpoint registered for `train_dir`, or None if unregistered."""
    try:
        key = Path(train_dir).resolve().relative_to(REPO).as_posix()
    except ValueError:  # a tree outside the repo — no convention to lean on
        return None
    name = PROMOTED_BY_TREE.get(key)
    return WEIGHTS / name if name else None


def available_weights():
    """-> a one-line inventory of data/training/weights/, for error messages."""
    found = sorted(p.name for p in WEIGHTS.glob("*.pt")) if WEIGHTS.is_dir() else []
    return f"Available in {WEIGHTS}: " + (", ".join(found) if found else "(none)")


def resolve_weights(explicit, train_dir, flag):
    """Explicit path (which MUST exist) or the tree's promoted checkpoint.

    A typo'd --new used to be dropped silently, so the run compared the old model against itself
    and looked like 'no regression'. Both failure modes now exit with guidance instead."""
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"{flag}: weights not found: {p}")
        return p
    w = promoted_weights(train_dir)
    if w is None:
        raise SystemExit(
            f"no promoted checkpoint is registered for --train-dir {train_dir}.\n"
            f"Pass {flag} explicitly, or add the tree to PROMOTED_BY_TREE in eval_holdout.py.\n"
            f"{available_weights()}")
    if not w.exists():
        raise SystemExit(
            f"{flag}: the promoted checkpoint for {train_dir} is missing: {w}\n{available_weights()}")
    return w


def scene_names(train_dir, split):
    """-> sorted distinct '<run>' prefixes of the AL chips in images/<split> (for error messages)."""
    d = Path(train_dir) / "images" / split
    return sorted({p.name.split("__")[0] for p in d.glob("*__*.png")}) if d.is_dir() else []


def check_holdout(train_dir, holdout, split="test"):
    """-> the sorted holdout chips, or SystemExit naming the scenes that ARE in the split.

    A filter matching nothing used to sail through and report P=R=0.000 over zero chips, which
    looks like a measured result. It is now a hard error, raised BEFORE any weight is loaded."""
    train_dir = Path(train_dir)
    hits = sorted(p for p in (train_dir / "images" / split).glob("*__*.png") if holdout in p.name)
    if not hits:
        avail = scene_names(train_dir, split)
        raise SystemExit(
            f"--holdout '{holdout}' matched no chips in {train_dir / 'images' / split}.\n"
            f"Scenes in that split: {', '.join(avail) if avail else '(none)'}")
    return hits


def build_subset(train_dir, dst, src_split, name_filter):
    """Copy AL chips (name contains `name_filter`) from <train_dir>/images/<src_split> into a val-only set."""
    train_dir = Path(train_dir)
    d = Path(dst)
    (d / "images" / "val").mkdir(parents=True, exist_ok=True)
    (d / "labels" / "val").mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in (train_dir / "images" / src_split).glob("*__*.png")
                  if name_filter in p.name)
    boxes = 0
    for img in imgs:
        shutil.copy2(img, d / "images" / "val" / img.name)
        lbl = train_dir / "labels" / src_split / f"{img.stem}.txt"
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
    r = load_yolo()(str(weights)).val(data=str(data), split=split, imgsz=imgsz,
                                      verbose=False, plots=False, save_json=False)
    return r.box.map50, r.box.mr


def det_count(weights, img_dir, conf, imgsz):
    m = load_yolo()(str(weights))
    return sum(len(m.predict(source=str(p), imgsz=imgsz, conf=conf, verbose=False)[0].boxes)
               for p in sorted(Path(img_dir).glob("*.png")))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", type=Path, default=TRAIN,
                    help="Dataset tree to evaluate against (default data/training = optical). "
                         "Pass data/training_sar for the SAR set; --old/--new then default to that "
                         "tree's promoted checkpoint.")
    ap.add_argument("--old", type=Path, default=None,
                    help="Baseline weights (default: --train-dir's promoted checkpoint).")
    ap.add_argument("--new", type=Path, default=None,
                    help="Candidate weights (default: --train-dir's promoted checkpoint).")
    ap.add_argument("--holdout", default="fallbacktest", help="Name filter for the held-out TEST scene.")
    ap.add_argument("--fp-scenes", nargs="*", default=["gbr-cairns-reefs", "testrun"],
                    help="TRAIN-split scene filters to count raw detections on (FP probe). "
                         "Defaults are OPTICAL scenes; pass SAR filters when --train-dir is the SAR tree.")
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
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train_dir = args.train_dir.resolve()
    if not (train_dir / "images").is_dir():
        raise SystemExit(f"--train-dir has no images/ dir: {train_dir}")
    check_holdout(train_dir, args.holdout)  # cheap dataset guard first — fail before loading a model
    old_imgsz = args.old_imgsz if args.old_imgsz is not None else args.imgsz
    new_imgsz = args.new_imgsz if args.new_imgsz is not None else args.imgsz
    old_w = resolve_weights(args.old, train_dir, "--old")
    new_w = resolve_weights(args.new, train_dir, "--new")

    pairs = [("OLD", old_w, old_imgsz), ("NEW", new_w, new_imgsz)]
    if old_w == new_w and old_imgsz == new_imgsz:
        pairs = [("MODEL", old_w, old_imgsz)]  # single-model gate — don't run identical inference twice
    full_yml = train_dir / "config.yml"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sat_yml, sat_imgs, sat_boxes = build_subset(train_dir, tmp / "sat", "test", args.holdout)
        fp_dirs = {}
        for f in args.fp_scenes:
            _, imgs, _ = build_subset(train_dir, tmp / f"fp_{f}", "train", f)
            if not imgs:
                print(f"warning: FP probe '{f}' matched no chips in {train_dir / 'images' / 'train'} "
                      f"— skipping (a zero-chip probe reports 0 detections, which reads as "
                      f"'no false positives'). Scenes there: "
                      f"{', '.join(scene_names(train_dir, 'train')) or '(none)'}")
                continue
            fp_dirs[f] = (tmp / f"fp_{f}" / "images" / "val", len(imgs))
        print(f"train dir: {train_dir}")
        print(f"satellite holdout '{args.holdout}': {len(sat_imgs)} chips, {sat_boxes} boxes"
              f"  (match tol {args.match_dist:.0f}px, conf {args.conf})")
        for f, (_, n) in fp_dirs.items():
            print(f"FP probe '{f}': {n} chips (in-sample if in TRAIN -> directional)")
        print(f"inference imgsz: " + "  ".join(f"{t}={sz}" for t, _, sz in pairs)
              + ("  (test-time upscaling in play)" if max(sz for _, _, sz in pairs) > TILE else ""))

        rows = []
        for tag, w, sz in pairs:
            model = load_yolo()(str(w))
            ie = instance_eval(model, sat_imgs, args.conf, args.match_dist, sz)
            ci = bootstrap_recall(ie["per_chip"], args.bootstrap)
            m50, _ = iou_map(w, sat_yml, sz)
            base50, baseR = iou_map(w, full_yml, sz, "test") if full_yml.exists() else (float("nan"),) * 2
            fps = {f: det_count(w, d, args.conf, sz) for f, (d, _) in fp_dirs.items()}
            rows.append((tag, w, ie, ci, m50, base50, baseR, fps, sz))

        print("\n================ HOLDOUT — instance-level (center-distance matching) ================")
        print(f"{'weights':<7}{'P':>7}{'R':>7}{'TP':>5}{'FP':>5}{'FN':>5}{'FP/chip':>9}{'recall 90% CI':>18}")
        for tag, w, ie, ci, *_ in rows:
            print(f"{tag:<7}{ie['P']:>7.3f}{ie['R']:>7.3f}{ie['TP']:>5}{ie['FP']:>5}{ie['FN']:>5}"
                  f"{ie['fp_per_chip']:>9.2f}{f'[{ci[0]:.2f}, {ci[1]:.2f}]':>18}")

        print("\nrecall stratified by GT pixel size (hit/total):")
        # >=14 wide: a full cell ('123/456 (0.48)') exactly fills 12 and collides with its neighbour.
        print(f"{'weights':<7}{'1-2px':>15}{'3-5px':>15}{'>5px':>15}")
        for tag, w, ie, *_ in rows:
            b = ie["bins"]
            def cell(k):
                hit, tot = b[k]
                return f"{hit}/{tot}" + (f" ({hit/tot:.2f})" if tot else "")
            print(f"{tag:<7}{cell('1-2px'):>15}{cell('3-5px'):>15}{cell('>5px'):>15}")

        print("\nsecondary IoU-mAP (unstable at this scale — context only):")
        print(f"  'base' = test split of {full_yml}"
              + ("" if full_yml.exists() else "  (MISSING -> nan)"))
        print(f"{'weights':<7}{'imgsz':>7}{'holdout mAP50':>14}{'base mAP50':>12}{'base R':>9}")
        for tag, w, ie, ci, m50, base50, baseR, fps, sz in rows:
            print(f"{tag:<7}{sz:>7}{m50:>14.3f}{base50:>12.3f}{baseR:>9.3f}")

        print(f"\nfalse-positive probe — raw detections @conf{args.conf} (lower is better):")
        if not fp_dirs:
            print("  (no FP probe scene matched this tree — pass --fp-scenes <filter> to enable)")
        for tag, w, ie, ci, m50, base50, baseR, fps, sz in rows:
            if fps:
                print(f"  {tag:<7} " + "   ".join(f"{f}: {c}" for f, c in fps.items()))
        for tag, w, *_ in rows:
            print(f"  {tag} = {w}")


if __name__ == "__main__":
    raise SystemExit(main())
