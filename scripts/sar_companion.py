"""Generate a co-registered OPTICAL companion image for every SAR chip in an s1.py run.

SAR speckle makes land/water ambiguous, so the review UI shows each SAR chip beside an optical image
of the exact same ground footprint — land is static, so the S1/S2 date mismatch is irrelevant for
telling land from water. For each SAR chip it reprojects the paired Sentinel-2 run's imagery into the
SAR chip's own grid (same CRS, transform, and H×W), so the two are pixel-aligned and the SAME
detections.json boxes overlay both. Writes companion/<chip>.png into the SAR run dir; review_server.py
auto-detects that dir and renders the side-by-side panel.
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Co-registered optical companions for a SAR review run.")
    p.add_argument("--sar-run", type=Path, required=True, help="The s1.py SAR run dir to annotate.")
    p.add_argument("--optical-run", type=Path, required=True,
                   help="A Sentinel-2 run over the same AOI (its imagery is the land reference).")
    return p.parse_args()


def main():
    args = parse_args()
    sar_run, opt_run = args.sar_run.resolve(), args.optical_run.resolve()
    opt_chips = sorted((opt_run / "chips").glob("*.tif"))
    if not opt_chips:
        raise RuntimeError(f"No optical chips in {opt_run}/chips")

    # one mosaic of the optical AOI (S2 UTM), reprojected per SAR chip below
    srcs = [rasterio.open(p) for p in opt_chips]
    try:
        mosaic, mosaic_tf = merge(srcs, indexes=[1, 2, 3])   # (3, H, W) uint8
        opt_crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()

    out_dir = sar_run / "companion"; out_dir.mkdir(exist_ok=True)
    sar_chips = sorted((sar_run / "chips").glob("*.tif"))
    n = 0
    for chip in sar_chips:
        with rasterio.open(chip) as s:
            dst_tf, dst_crs, H, W = s.transform, s.crs, s.height, s.width
        dst = np.zeros((3, H, W), np.uint8)
        for b in range(3):
            reproject(source=mosaic[b], destination=dst[b],
                      src_transform=mosaic_tf, src_crs=opt_crs,
                      dst_transform=dst_tf, dst_crs=dst_crs, resampling=Resampling.bilinear)
        Image.fromarray(np.transpose(dst, (1, 2, 0)), "RGB").save(out_dir / f"{chip.stem}.png")
        n += 1
    print(f"wrote {n} optical companions -> {out_dir}")


if __name__ == "__main__":
    main()
