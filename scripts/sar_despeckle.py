"""Speckle filtering for Sentinel-1 chips — the SAR precision filter (improvement plan S6 / ROADMAP W6).

SAR images carry multiplicative speckle that a detector can mistake for vessels. This applies a Lee
local-statistics filter to the 8-bit pseudo-RGB chips `providers/s1.py` writes, and is the tool the
W6 A/B measured.

THE DOMAIN POINT, because the obvious choice is the wrong one:
speckle is *multiplicative* in linear power, but `providers/s1.py` converts backscatter to dB and
percentile-stretches to 8-bit, and after that log transform speckle is approximately *additive* with
roughly constant variance. So the correct filter for these chips is Lee's ORIGINAL local-statistics
(additive) form:

    out = mean_local + k * (x - mean_local),   k = max(0, (var_local - var_noise) / var_local)

Applying the multiplicative variant to log-domain data would be mis-specified. `var_noise` is
estimated as the MEDIAN of local variances across the chip — homogeneous water dominates a maritime
scene, so the median local variance is a robust estimate of the speckle floor. That is self-calibrating,
which matters because s1.py stretches every scene differently (the same reason `sar_seed.py` uses an
adaptive `median + k*MAD` rather than a fixed threshold).

TWO MODES, and the second is the one that works:
  --mode lee   plain Lee. MEASURED TRAP (W6): best pixel contrast of anything tried (+41%) and the
               WORST detection — recall ceiling 0.696 -> 0.391, losing 7 of 16 findable vessels.
               Smoothing a compact target's peak destroys it even as peak-to-background improves.
  --mode peak  (default) Lee everywhere EXCEPT where x > mean + `--peak-sigma`*sd, where the original
               pixel is kept: smooth the background, never touch a compact target's peak. At matched
               recall this cut open-water false alarms 63% (797 -> 292 raw dets) and lifted the recall
               ceiling 0.696 -> 0.870 on the Gibraltar holdout.

NOT textbook Refined-Lee (Lee 1981's 8 edge-aligned sub-windows) — that preserves linear features like
coastlines; this targets compact bright objects. Worth trying the real thing if this is pursued.

⚠️ If a model is TRAINED on despeckled chips, inference MUST despeckle identically — train/inference
preprocessing has to match. Nothing in the pipeline does this automatically yet; see ROADMAP W6.

⚠️ Labels are geometric, so a filtered tree reuses the source tree's labels unchanged — the A/B
compares pixels only.

Usage:
    # filter a whole training tree (images filtered, labels + config copied)
    python scripts/sar_despeckle.py --src data/training_sar --dst data/training_sar_peak
    # filter one sat_fetch run's chips into a sibling dir
    python scripts/sar_despeckle.py --run data/raw/sentinel1/<run>/
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ("train", "val", "test")
CHIP_SUFFIXES = (".png", ".tif", ".tiff")


# --------------------------------------------------------------------------- pure: the filter

def box_mean(a, window):
    """Mean over a window*window neighbourhood (reflect-padded) via an integral image.

    Verified against a naive per-pixel reference for odd windows and non-square shapes; scipy is
    deliberately not a dependency of this repo."""
    r = window // 2
    p = np.pad(np.asarray(a, dtype=np.float64), r, mode="reflect")
    ii = np.zeros((p.shape[0] + 1, p.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = p.cumsum(0).cumsum(1)
    h, w = np.shape(a)
    total = (ii[window:window + h, window:window + w] - ii[0:h, window:window + w]
             - ii[window:window + h, 0:w] + ii[0:h, 0:w])
    return total / float(window * window)


def local_stats(chan, window):
    """-> (mean, variance) over the window, both same-shape as chan."""
    x = np.asarray(chan, dtype=np.float64)
    mean = box_mean(x, window)
    var = np.maximum(box_mean(x * x, window) - mean * mean, 0.0)
    return mean, var


def lee(chan, window=7):
    """Lee local-statistics filter for additive noise (the correct form for dB/8-bit SAR)."""
    x = np.asarray(chan, dtype=np.float64)
    mean, var = local_stats(x, window)
    var_noise = float(np.median(var))          # robust speckle floor: homogeneous water dominates
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(var > 0, (var - var_noise) / np.maximum(var, 1e-9), 0.0)
    return mean + np.clip(k, 0.0, 1.0) * (x - mean)


def lee_peak_preserving(chan, window=7, peak_sigma=2.0):
    """Lee, but a compact bright target's peak is left untouched — the variant that works (W6)."""
    x = np.asarray(chan, dtype=np.float64)
    filtered = lee(x, window)
    mean, var = local_stats(x, window)
    bright = x > (mean + peak_sigma * np.sqrt(var))
    return np.where(bright, x, filtered)


def despeckle_array(arr, window=7, mode="peak", peak_sigma=2.0):
    """Despeckle a HxW or HxWxC uint8 chip array -> uint8. Channels are filtered independently.

    Importable so an inference path can apply the SAME preprocessing the model was trained on."""
    a = np.asarray(arr)

    def one(c):
        if mode == "lee":
            return lee(c, window)
        return lee_peak_preserving(c, window, peak_sigma)

    out = one(a) if a.ndim == 2 else np.stack([one(a[:, :, c]) for c in range(a.shape[2])], axis=2)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- IO

def despeckle_image_file(src, dst, window, mode, peak_sigma):
    im = Image.open(src)
    out = despeckle_array(np.array(im), window, mode, peak_sigma)
    Image.fromarray(out, mode=im.mode).save(dst)


def despeckle_tree(src, dst, window, mode, peak_sigma):
    """Filter every image in a YOLO tree; labels + config are copied unchanged (labels are geometric)."""
    n = 0
    for split in SPLITS:
        src_images = src / "images" / split
        if not src_images.is_dir():
            continue
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in sorted(src_images.iterdir()):
            if img.suffix.lower() not in CHIP_SUFFIXES:
                continue
            despeckle_image_file(img, dst / "images" / split / img.name, window, mode, peak_sigma)
            lbl = src / "labels" / split / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, dst / "labels" / split / lbl.name)
            n += 1
        print(f"  {split}: {n} images so far")
    cfg = src / "config.yml"
    if cfg.exists():
        text = cfg.read_text(encoding="utf-8")
        text = text.replace(src.resolve().as_posix(), dst.resolve().as_posix())
        (dst / "config.yml").write_text(text, encoding="utf-8")
    return n


def despeckle_run(run, window, mode, peak_sigma, out_dir=None):
    """Filter a sat_fetch run's chips/ into a sibling dir, leaving the original run untouched."""
    chips = run / "chips"
    if not chips.is_dir():
        raise SystemExit(f"no chips/ dir in {run}")
    out = out_dir or (run / "chips_despeckled")
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for chip in sorted(chips.iterdir()):
        if chip.suffix.lower() in CHIP_SUFFIXES:
            despeckle_image_file(chip, out / chip.name, window, mode, peak_sigma)
            n += 1
    return n, out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Speckle-filter Sentinel-1 chips (Lee local statistics, log-domain). "
                    "Train and inference must use the SAME setting.")
    src = p.add_argument_group("input (provide one)")
    src.add_argument("--src", type=Path, help="Source YOLO tree (e.g. data/training_sar).")
    src.add_argument("--run", type=Path, help="A sat_fetch run dir; filters its chips/.")
    p.add_argument("--dst", type=Path, help="Destination tree for --src.")
    p.add_argument("--out", type=Path, help="Output dir for --run (default <run>/chips_despeckled).")
    p.add_argument("--mode", choices=("peak", "lee"), default="peak",
                   help="'peak' (default) keeps compact bright targets intact; 'lee' is plain Lee, a "
                        "MEASURED TRAP that costs ~45%% of the recall ceiling (ROADMAP W6).")
    p.add_argument("--window", type=int, default=7, help="Filter window (odd). Default 7.")
    p.add_argument("--peak-sigma", type=float, default=2.0,
                   help="A pixel above mean + this*sd is a target peak and is left unfiltered. Default 2.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if bool(args.src) == bool(args.run):
        raise SystemExit("provide exactly one of --src (a training tree) or --run (a sat_fetch run dir)")
    if args.window % 2 == 0 or args.window < 3:
        raise SystemExit(f"--window must be odd and >= 3 (got {args.window})")

    print("SAR DESPECKLE:")
    print(f"  mode={args.mode}  window={args.window}  peak_sigma={args.peak_sigma}")
    if args.mode == "lee":
        print("  WARNING: plain Lee measured WORSE than no filter on detection (ROADMAP W6) — "
              "it smooths compact targets away. 'peak' is the variant that works.")

    if args.src:
        if not args.dst:
            raise SystemExit("--src requires --dst")
        n = despeckle_tree(args.src.resolve(), args.dst.resolve(), args.window, args.mode,
                           args.peak_sigma)
        print(f"\nFiltered {n} images -> {args.dst}")
        print("Labels/config copied unchanged. Remember: a model trained here needs the SAME "
              "filter at inference.")
    else:
        n, out = despeckle_run(args.run.resolve(), args.window, args.mode, args.peak_sigma,
                               args.out.resolve() if args.out else None)
        print(f"\nFiltered {n} chips -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
