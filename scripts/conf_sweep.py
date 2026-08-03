"""Confidence-threshold sweep for a boat-detector checkpoint on the satellite holdout.

Picks the operating point. `eval_holdout.py` reports one fixed conf; this runs inference ONCE
at a low floor (so every box + its score is captured), then re-thresholds in Python across a
range of conf values — a cheap PR-style curve. Reuses eval_holdout's holdout/label/match logic
(including its `--train-dir` tree selection and promoted-checkpoint table).

Two views per conf:
  * HOLDOUT (busy scene): instance P / R / F1 / FP-per-chip via center-distance matching.
  * OPEN-WATER PROBE (near-empty False Bay 'testrun', in TRAIN -> directional): raw detections
    per chip = a proxy for the false-alarm rate on empty MPA water as conf drops.

Usage:
  venv/Scripts/python.exe scripts/conf_sweep.py \
      --weights data/training/runs/finetune-aug-1024/weights/best.pt --imgsz 1024

  # SAR: the tree selects both the chips AND the default checkpoint (best_sar.pt)
  venv/Scripts/python.exe scripts/conf_sweep.py \
      --train-dir data/training_sar --holdout s1deep-gibraltar --fp-scene s1deep-fujairah --imgsz 1024

ultralytics/torch are imported LAZILY (same reason as eval_holdout.py: the offline suite installs
nothing, so module import must stay stdlib-only).
"""
import argparse
from pathlib import Path

import eval_holdout as E  # same dir; only defines functions at import time


def scored_preds(model, img_path, imgsz, floor):
    """-> list of (cx_px, cy_px, score) for every box above `floor`."""
    r = model.predict(source=str(img_path), imgsz=imgsz, conf=floor, verbose=False)[0]
    if not len(r.boxes):
        return []
    xywh = r.boxes.xywh.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    return [(float(b[0]), float(b[1]), float(c)) for b, c in zip(xywh, conf)]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", type=Path, default=E.TRAIN,
                    help="Dataset tree to sweep against (default data/training = optical). "
                         "Pass data/training_sar for the SAR set.")
    ap.add_argument("--weights", type=Path, default=None,
                    help="Checkpoint to sweep (default: --train-dir's promoted checkpoint).")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--holdout", default="fallbacktest")
    ap.add_argument("--fp-scene", default="testrun",
                    help="Open-water probe scene filter (in TRAIN -> directional). The default is an "
                         "OPTICAL scene; pass a SAR filter when --train-dir is the SAR tree.")
    ap.add_argument("--match-dist", type=float, default=8.0)
    ap.add_argument("--floor", type=float, default=0.01, help="Inference conf floor (capture everything above this).")
    ap.add_argument("--confs", type=float, nargs="*",
                    default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train_dir = args.train_dir.resolve()
    if not (train_dir / "images").is_dir():
        raise SystemExit(f"--train-dir has no images/ dir: {train_dir}")
    hold_imgs = E.check_holdout(train_dir, args.holdout)  # cheap dataset guard before loading a model
    weights = E.resolve_weights(args.weights, train_dir, "--weights")

    fp_imgs = sorted(p for p in (train_dir / "images" / "train").glob("*__*.png") if args.fp_scene in p.name)
    if not fp_imgs:
        # A zero-chip probe would print 0.00 dets/chip, which reads as "no false alarms on open water".
        print(f"warning: open-water probe '{args.fp_scene}' matched no chips in "
              f"{train_dir / 'images' / 'train'} — that column will read 'n/a'. Scenes there: "
              f"{', '.join(E.scene_names(train_dir, 'train')) or '(none)'}")

    model = E.load_yolo()(str(weights))
    # Inference once per chip at the floor; cache scored predictions + GT.
    hold = [(E.gt_boxes(E.label_for(im)), scored_preds(model, im, args.imgsz, args.floor)) for im in hold_imgs]
    fp = [scored_preds(model, im, args.imgsz, args.floor) for im in fp_imgs]
    n_gt = sum(len(g) for g, _ in hold)

    print(f"train dir: {train_dir}")
    print(f"weights: {weights}")
    print(f"imgsz {args.imgsz}  holdout '{args.holdout}': {len(hold_imgs)} chips / {n_gt} boxes  "
          f"(match tol {args.match_dist:.0f}px)")
    print(f"open-water probe '{args.fp_scene}': {len(fp_imgs)} chips (in TRAIN -> directional)\n")
    print(f"{'conf':>5}{'P':>7}{'R':>7}{'F1':>7}{'TP':>5}{'FP':>5}{'FN':>5}"
          f"{'FP/chip':>9}{'openwtr dets/chip':>19}")

    for t in args.confs:
        TP = FP = FN = 0
        for gts, preds in hold:
            pc = [(p[0], p[1]) for p in preds if p[2] >= t]
            used_g, used_p = E.match(gts, pc, args.match_dist)
            TP += len(used_g)
            FP += len(pc) - len(used_p)
            FN += len(gts) - len(used_g)
        P = TP / (TP + FP) if (TP + FP) else 0.0
        R = TP / (TP + FN) if (TP + FN) else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) else 0.0
        fp_dets = sum(len([1 for p in preds if p[2] >= t]) for preds in fp)
        ow = f"{fp_dets / len(fp_imgs):.2f}" if fp_imgs else "n/a"
        print(f"{t:>5.2f}{P:>7.3f}{R:>7.3f}{F1:>7.3f}{TP:>5}{FP:>5}{FN:>5}{FP/len(hold_imgs):>9.2f}{ow:>19}")


if __name__ == "__main__":
    raise SystemExit(main())
