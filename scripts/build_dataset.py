import argparse
import json
import random
import shutil
from pathlib import Path

# Assemble the YOLO training dataset for the boat detector, in-repo.
#
# It LAYERS the human-reviewed active-learning chips (from export_labels.py, under
# data/processed/training_exports/<tag>/) on top of the base dataset that lives directly in
# data/training/images|labels/{train,val,test} (the ~621 originally-labelled boat photos,
# named `boat<N>`). Every active-learning chip is named `<run>__<chip>` (contains "__"), so the
# two are always distinguishable.
#
# Idempotent by construction: each run first PURGES every "__" file from the split dirs, then
# re-adds the exports with a fixed seed. Re-run it freely as you review more sat_fetch runs;
# the base `boat<N>` files are never touched.
#
#   python scripts/build_dataset.py                 # pull all export tags
#   python scripts/build_dataset.py --dry-run       # preview, write nothing
#   python scripts/build_dataset.py --exports D1 D2  # explicit export roots or tag dirs

REPO_ROOT = Path(__file__).resolve().parents[1]                 # C:/gtest/AntiAngler
DATA_DIR = REPO_ROOT / "data" / "training"                      # base split lives here
DEFAULT_EXPORTS = REPO_ROOT / "data" / "processed" / "training_exports"  # in-repo exports

SPLITS = ("train", "val", "test")
AL_MARKER = "__"  # active-learning files are "<run>__<chip>"; base originals are "boat<N>"


def parse_args():
    p = argparse.ArgumentParser(
        description="Layer active-learning exports onto the base boat dataset in data/training/.")
    p.add_argument("--exports", type=Path, nargs="+", default=[DEFAULT_EXPORTS],
                   help="Export ROOT(s) (containing <tag>/images,<tag>/labels) or a single tag dir. "
                        f"Default: {DEFAULT_EXPORTS}")
    p.add_argument("--data", type=Path, default=DATA_DIR,
                   help="Training dir holding images/{train,val,test} + labels/{...} + config.yml.")
    p.add_argument("--train", type=float, default=0.7, help="Train fraction of the AL chips.")
    p.add_argument("--val", type=float, default=0.2, help="Val fraction of the AL chips.")
    p.add_argument("--test", type=float, default=0.1, help="Test fraction (remainder).")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed for a reproducible AL split.")
    p.add_argument("--no-purge", action="store_true",
                   help="Do not remove existing '__' files first (default is to purge for idempotency).")
    p.add_argument("--dry-run", action="store_true", help="Report only; copy/delete nothing.")
    return p.parse_args()


def collect_tag_dirs(export_roots):
    """A path is a tag dir if it directly holds images/+labels/; otherwise treat it as a root and
    take each immediate subdir that does."""
    tag_dirs = []
    for root in export_roots:
        root = root.resolve()
        if not root.exists():
            print(f"  WARNING: export path does not exist: {root}")
            continue
        if (root / "images").is_dir() and (root / "labels").is_dir():
            tag_dirs.append(root)
        else:
            for d in sorted(root.iterdir()):
                if d.is_dir() and (d / "images").is_dir() and (d / "labels").is_dir():
                    tag_dirs.append(d)
    return tag_dirs


def collect_pairs(tag_dirs):
    """Return {stem: (image_path, label_path)} across all tags. Later tags win on stem collision
    (a chip re-exported after re-review supersedes the earlier copy)."""
    pairs = {}
    for td in tag_dirs:
        for img in sorted((td / "images").glob("*.png")):
            lbl = td / "labels" / f"{img.stem}.txt"
            if lbl.exists():  # a reviewed chip always has a label file (empty file = hard negative)
                pairs[img.stem] = (img, lbl)
            else:
                print(f"  WARNING: no label for {img.name} in {td.name}; skipping")
    return pairs


def purge_al_files(data_dir):
    removed = 0
    for split in SPLITS:
        for sub in ("images", "labels"):
            d = data_dir / sub / split
            if d.is_dir():
                for f in d.glob(f"*{AL_MARKER}*"):
                    f.unlink()
                    removed += 1
    return removed


def split_stems(stems, ratios, seed):
    rng = random.Random(seed)
    items = sorted(stems)          # deterministic base order, then a seeded shuffle
    rng.shuffle(items)
    n = len(items)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    return {"train": items[:n_tr], "val": items[n_tr:n_tr + n_va], "test": items[n_tr + n_va:]}


def count_base(data_dir):
    out = {}
    for split in SPLITS:
        d = data_dir / "images" / split
        out[split] = len([f for f in d.glob("*.png") if AL_MARKER not in f.name]) if d.is_dir() else 0
    return out


def write_config(data_dir):
    """Keep config.yml's `path` pinned to this machine's absolute training dir."""
    cfg = data_dir / "config.yml"
    nc, names = 1, "['boat']"
    if cfg.exists():
        for line in cfg.read_text().splitlines():
            s = line.strip()
            if s.startswith("nc:"):
                nc = s.split(":", 1)[1].split("#")[0].strip() or nc
            elif s.startswith("names:"):
                names = s.split(":", 1)[1].strip() or names
    cfg.write_text(
        "# config.yml for YOLOv8 boat-detector training (in-repo, AntiAngler).\n"
        "# `path` is rewritten to this machine's absolute training dir by scripts/build_dataset.py.\n"
        f"path: {data_dir.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {nc}\nnames: {names}\n")


def main():
    args = parse_args()
    data_dir = args.data.resolve()
    ratios = (args.train, args.val, args.test)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit(f"--train/--val/--test must sum to 1.0 (got {sum(ratios)}).")
    for split in SPLITS:
        (data_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (data_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("BUILD DATASET")
    print(f"  training dir : {data_dir}")
    tag_dirs = collect_tag_dirs(args.exports)
    print(f"  export tags ({len(tag_dirs)}): " + (", ".join(t.name for t in tag_dirs) or "(none)"))
    pairs = collect_pairs(tag_dirs)
    print(f"  active-learning chips found: {len(pairs)}")

    base = count_base(data_dir)
    print(f"  base kept (untouched): train={base['train']} val={base['val']} test={base['test']}")

    assignment = split_stems(list(pairs), ratios, args.seed)
    print("  active-learning split: " + " ".join(f"{s}={len(v)}" for s, v in assignment.items()))

    if args.dry_run:
        print("\nDRY RUN - nothing written. Re-run without --dry-run to apply.")
        return

    if not args.no_purge:
        removed = purge_al_files(data_dir)
        print(f"  purged {removed} existing '__' file(s) for a clean, reproducible rebuild")

    copied = 0
    for split, stems in assignment.items():
        for stem in stems:
            img, lbl = pairs[stem]
            shutil.copy2(img, data_dir / "images" / split / img.name)
            shutil.copy2(lbl, data_dir / "labels" / split / lbl.name)
            copied += 1

    write_config(data_dir)

    manifest = {
        "seed": args.seed,
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
        "export_tags": [t.name for t in tag_dirs],
        "active_learning_chips": len(pairs),
        "split_assignment": {s: sorted(v) for s, v in assignment.items()},
        "base_kept": base,
    }
    (data_dir / "active_learning_manifest.json").write_text(json.dumps(manifest, indent=2))

    tot = {s: base[s] + len(assignment[s]) for s in SPLITS}
    print(f"  copied {copied} AL image+label pair(s) into the split")
    print(f"\nDATASET READY  train={tot['train']} val={tot['val']} test={tot['test']}  "
          f"(base + active-learning)")
    print(f"  provenance -> {data_dir / 'active_learning_manifest.json'}")
    if any(len(assignment[s]) == 0 for s in ("val", "test")) and len(pairs) < 20:
        print("  NOTE: with so few AL chips some AL splits are empty; the base images still fill "
              "every split. Review more sat_fetch runs to grow overhead-domain val/test coverage.")
    print("\nNext: python scripts/train.py --weights best   (fine-tune)  |  --weights scratch  (new model)")


if __name__ == "__main__":
    main()
