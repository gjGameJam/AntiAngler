import argparse
from pathlib import Path

from ultralytics import YOLO

# Train (or fine-tune) the boat detector, in-repo. Paths resolve off __file__, so run from anywhere.
#
# Two modes, one flag:
#   python scripts/train.py --weights best       # FINE-TUNE from the current best.pt (active learning)
#   python scripts/train.py --weights scratch    # train a NEW model from the yolov8n base
#   python scripts/train.py --weights path/to.pt # from an explicit checkpoint
#
# Base weights live in data/training/weights/ (best.pt = current model, yolov8n.pt = base). To avoid
# clobbering best.pt, every run writes to a NEW auto-incrementing dir under data/training/runs/;
# overwriting data/training/weights/best.pt requires an explicit --out. Fine-tuning defaults to a 10x
# lower lr0 (via a pinned SGD optimizer) so it adapts to the overhead chips without forgetting the
# original photos.
#
# Build the dataset first:  python scripts/build_dataset.py

REPO_ROOT = Path(__file__).resolve().parents[1]                 # C:/gtest/AntiAngler
TRAIN_DIR = REPO_ROOT / "data" / "training"
DATA_YML = TRAIN_DIR / "config.yml"
WEIGHTS_DIR = TRAIN_DIR / "weights"
BASELINE = WEIGHTS_DIR / "best.pt"
YOLOV8N = WEIGHTS_DIR / "yolov8n.pt"
RUNS_DIR = TRAIN_DIR / "runs"

WEIGHT_ALIASES = {
    "best": BASELINE, "baseline": BASELINE, "finetune": BASELINE,
    "scratch": YOLOV8N, "yolov8n": YOLOV8N, "new": YOLOV8N,
}


def resolve_weights(spec):
    if spec is None:
        return (BASELINE if BASELINE.exists() else YOLOV8N)
    return WEIGHT_ALIASES.get(spec.lower(), Path(spec))


def parse_args():
    p = argparse.ArgumentParser(description="Train or fine-tune the YOLOv8 boat detector (in-repo).")
    p.add_argument("--weights", default=None,
                   help="'best' (fine-tune from best.pt), 'scratch' (new model from yolov8n), "
                        "a checkpoint path, or a model .yaml (build a fresh arch, e.g. yolov8n-p2). "
                        "Default: best.pt if it exists, else yolov8n.")
    p.add_argument("--pretrained", default=None,
                   help="Warm-start a fresh arch (a .yaml --weights) by transferring matching layers from "
                        "this checkpoint (alias like 'yolov8n'/'best' or a path). Non-matching new layers "
                        "(e.g. a P2 head) stay randomly initialised. Recommended for a --weights *.yaml on a "
                        "small dataset; no-op-ish when --weights is already a .pt.")
    p.add_argument("--data", type=Path, default=DATA_YML, help="Dataset yaml (default data/training/config.yml).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None,
                   help="'cpu', or a CUDA index like '0'. Default: auto - CUDA 0 if available, else cpu "
                        "(the old always-cpu default silently ignored an idle GPU).")
    p.add_argument("--name", default="finetune",
                   help="Run dir under data/training/runs/. Auto-increments unless --force.")
    p.add_argument("--lr0", type=float, default=None,
                   help="Initial LR. Default: 0.001 when fine-tuning, ultralytics auto when from scratch.")
    p.add_argument("--optimizer", default="auto",
                   help="'auto' (ultralytics picks optimizer AND lr0), or a concrete one like 'SGD'/'AdamW'. "
                        "A custom --lr0 is only honoured with a concrete optimizer, so a fine-tune lr0 "
                        "auto-switches this to SGD.")
    p.add_argument("--freeze", type=int, default=None,
                   help="Freeze the first N layers (e.g. 10 freezes the backbone; good for tiny AL sets).")
    p.add_argument("--patience", type=int, default=20, help="Early-stop patience (0 disables).")
    # Small-object augmentation knobs. Default None = leave Ultralytics' default in place, so existing
    # flows are unchanged. For 1-5 px vessels: mosaic downscales already-tiny objects (lower it / close
    # it earlier), and copy_paste multiplies scarce positives. See docs/STATUS.md Phase 0.
    p.add_argument("--mosaic", type=float, default=None,
                   help="Mosaic aug probability (ultralytics default 1.0). Lower (~0.5) for tiny objects.")
    p.add_argument("--close-mosaic", type=int, default=None,
                   help="Disable mosaic for the last N epochs (ultralytics default 10). Higher closes it earlier.")
    p.add_argument("--copy-paste", type=float, default=None,
                   help="In-training copy-paste aug probability (ultralytics default 0.0).")
    p.add_argument("--degrees", type=float, default=None,
                   help="Random rotation degrees (ultralytics default 0.0); adds vessel-orientation variety.")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="Fraction of the train set to use (e.g. 0.05 for a quick smoke run).")
    p.add_argument("--force", action="store_true",
                   help="Allow writing into an existing run dir of --name.")
    p.add_argument("--resume", action="store_true", help="Resume an interrupted run of --name.")
    return p.parse_args()


def main():
    args = parse_args()
    weights = resolve_weights(args.weights)
    if not Path(weights).exists():
        raise SystemExit(f"Weights not found: {weights}")
    if not args.data.exists():
        raise SystemExit(f"Dataset yaml not found: {args.data}. Run scripts/build_dataset.py first.")

    pretrained = resolve_weights(args.pretrained) if args.pretrained is not None else None
    if pretrained is not None and not Path(pretrained).exists():
        raise SystemExit(f"Pretrained checkpoint not found: {pretrained}")

    is_finetune = Path(weights).resolve() == BASELINE.resolve()
    lr0 = args.lr0 if args.lr0 is not None else (0.001 if is_finetune else None)
    # ultralytics' optimizer='auto' determines lr0 itself and IGNORES any lr0 we pass. To make a
    # fine-tune's lower lr0 actually take effect, pin a concrete optimizer (SGD, YOLO's detection default).
    optimizer = args.optimizer
    if optimizer == "auto" and lr0 is not None:
        optimizer = "SGD"

    if args.device is None:
        import torch
        args.device = "0" if torch.cuda.is_available() else "cpu"

    print("TRAIN")
    print(f"  mode      : {'FINE-TUNE' if is_finetune else 'FROM-SCRATCH'}")
    print(f"  weights   : {weights}")
    if pretrained is not None:
        print(f"  pretrained: {pretrained} (transfer matching layers into fresh arch)")
    print(f"  data      : {args.data}")
    print(f"  epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} device={args.device} "
          f"fraction={args.fraction}")
    print(f"  optimizer={optimizer} lr0={lr0 if lr0 is not None else 'auto'} "
          f"freeze={args.freeze} patience={args.patience}")
    aug_over = {k: v for k, v in (("mosaic", args.mosaic), ("close_mosaic", args.close_mosaic),
                                  ("copy_paste", args.copy_paste), ("degrees", args.degrees)) if v is not None}
    if aug_over:
        print(f"  aug overrides: {aug_over}")
    print(f"  output    : data/training/runs/{args.name}"
          f"{' (force overwrite)' if args.force else ' (auto-increment)'}")
    if args.device == "cpu":
        print("  NOTE: training on CPU (no CUDA); expect this to be slow for full runs.")

    kwargs = dict(
        data=str(args.data), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        device=args.device, project=str(RUNS_DIR), name=args.name,
        exist_ok=args.force, seed=42, patience=args.patience, fraction=args.fraction,
        resume=args.resume, optimizer=optimizer,
    )
    if lr0 is not None:
        kwargs["lr0"] = lr0
    if args.freeze is not None:
        kwargs["freeze"] = args.freeze
    for name, val in (("mosaic", args.mosaic), ("close_mosaic", args.close_mosaic),
                      ("copy_paste", args.copy_paste), ("degrees", args.degrees)):
        if val is not None:
            kwargs[name] = val

    model = YOLO(str(weights))
    if pretrained is not None:
        # Transfer matching layers (backbone + shared head) from the checkpoint into the fresh arch;
        # non-matching new layers (e.g. the P2 head) keep their random init. Prints "Transferred X/Y".
        model.load(str(pretrained))
    model.train(**kwargs)

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"\nDONE. New weights -> {best}")
    print("Evaluate on the held-out test split:")
    print(f'  yolo detect val model="{best}" data="{args.data}" split=test')
    print("Promote to the current model (so --weights best uses it next time):")
    print(f'  copy "{best}" "{BASELINE}"')
    print("Run stage-2 detection with them:")
    print(f'  python scripts/detect_boats.py --run <run> --weights "{best}"')


if __name__ == "__main__":
    main()
