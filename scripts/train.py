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
                        "or a checkpoint path. Default: best.pt if it exists, else yolov8n.")
    p.add_argument("--data", type=Path, default=DATA_YML, help="Dataset yaml (default data/training/config.yml).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu", help="'cpu', or a CUDA index like '0' if a GPU is present.")
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

    is_finetune = Path(weights).resolve() == BASELINE.resolve()
    lr0 = args.lr0 if args.lr0 is not None else (0.001 if is_finetune else None)
    # ultralytics' optimizer='auto' determines lr0 itself and IGNORES any lr0 we pass. To make a
    # fine-tune's lower lr0 actually take effect, pin a concrete optimizer (SGD, YOLO's detection default).
    optimizer = args.optimizer
    if optimizer == "auto" and lr0 is not None:
        optimizer = "SGD"

    print("TRAIN")
    print(f"  mode      : {'FINE-TUNE' if is_finetune else 'FROM-SCRATCH'}")
    print(f"  weights   : {weights}")
    print(f"  data      : {args.data}")
    print(f"  epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} device={args.device} "
          f"fraction={args.fraction}")
    print(f"  optimizer={optimizer} lr0={lr0 if lr0 is not None else 'auto'} "
          f"freeze={args.freeze} patience={args.patience}")
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

    model = YOLO(str(weights))
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
