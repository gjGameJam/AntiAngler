"""SAR vessel-candidate seeder for Sentinel-1 chips — the classical detector used to seed the human
review loop when no trained SAR model (best_sar.pt) exists yet. Writes detections.json in the exact
schema review_server.py reads, so the SAR review is identical to the optical one.

Method note: textbook cell-averaging CFAR assumes DARK ocean, but s1.py's dB-percentile stretch (tuned
for visual display) leaves water mid-bright (median ~136/255, p90 ~175) and saturates vessels at 255 —
so CFAR/top-hat thresholds ride above real ships or drown in the bright textured water (measured, not
guessed). What actually separates ships here is (a) brightness relative to the LOCAL water and (b) spatial
COHERENCE. So the seeder:
  1. dual-pol geometric mean sqrt(VV*VH) — a vessel is bright in BOTH pols; speckle is independent
     per-pol, so the product suppresses it,
  2. a light median despeckle,
  3. a PER-CHIP adaptive threshold median + k*(1.4826*MAD) of the chip's own water — robust (speckle-
     immune) and stretch-invariant, since s1.py normalizes each scene differently (a fixed 0-255
     threshold does NOT generalize: it floods bright-water scenes and misses dim-stretch ones), then
  4. connected-component blobs with an area window — the min-area kills isolated bright speckle
     (whitecaps, single-pixel spikes), the max-area/side kills extended land/coast.
Nodata (the SAR swath wedge) is masked and its boundary eroded so the bright data/nodata edge does not
generate false alarms. **Land (buildings/ports are very bright in SAR) is the dominant false alarm near
port anchorages — pass `--water-only` with a built water_mask.tif to drop it.** Leans toward recall —
the human review is the quality gate.

Input: an s1.py run dir (VV/VH/(VV/VH) pseudo-RGB GeoTIFF chips + manifest.json).
"""
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import cv2
from rasterio.warp import transform as warp_transform


def parse_args():
    p = argparse.ArgumentParser(description="SAR vessel-candidate seeder (saturation + dual-pol coherence).")
    p.add_argument("--run", type=Path, required=True, help="An s1.py SAR run dir (chips/ + manifest.json).")
    p.add_argument("--k", type=float, default=6.0, help="Per-chip adaptive threshold in robust sigmas: fire where "
                   "the despeckled dual-pol geo-mean exceeds median + k*(1.4826*MAD) of the chip's own water. "
                   "Stretch-invariant (s1.py normalizes per scene), so it self-adapts and stays quiet on empty "
                   "water. Lower = more candidates/recall; 5-7 typical.")
    p.add_argument("--min-area", type=int, default=8, help="Min blob area (px). Separates coherent vessels from "
                   "isolated bright speckle. ~8 keeps vessels >~30 m at S1 10 m.")
    p.add_argument("--max-area", type=int, default=1600, help="Max blob area (px). Rejects extended land/coast.")
    p.add_argument("--max-side", type=int, default=60, help="Max blob bbox side (px). Rejects long coastal strips.")
    p.add_argument("--kernel", type=int, default=13, help="Nodata-boundary erosion width (px) — drops the bright "
                   "data/nodata swath edge.")
    p.add_argument("--water-only", action="store_true", help="Also drop land candidates via the run's water_mask.tif.")
    p.add_argument("--min-depth", type=float, default=0.0, help="Deep-water gate (metres, positive): drop "
                   "candidates in water shallower than this via GEBCO bathymetry (e.g. 50 keeps water >=50 m "
                   "deep). Shallow water returns strong bathymetric SAR clutter that confuses detection. 0 = off.")
    return p.parse_args()


def detect_targets(vv, vh, a):
    """Return list of (x, y, w, h, cx, cy, peak) vessel candidates in one chip."""
    inten = np.sqrt(vv.astype(np.float32) * vh.astype(np.float32))       # dual-pol geo-mean (speckle down)
    valid = (vv > 0) & (vh > 0)
    desp = cv2.medianBlur(inten.astype(np.uint8), 3)                     # light despeckle
    water = desp[valid]
    if water.size < 100:                                                 # essentially all-nodata chip
        return []
    med = float(np.median(water))
    mad = float(np.median(np.abs(water - med))) * 1.4826 + 1e-6          # robust spread (speckle-immune)
    # per-chip, stretch-invariant; capped because vessels saturate ~255 in the dB stretch, so on
    # rough-sea / high-MAD scenes an uncapped median+k*MAD can exceed 255 and miss every ship.
    thresh = min(med + a.k * mad, 235.0)
    evalid = cv2.erode(valid.astype(np.uint8), np.ones((a.kernel, a.kernel), np.uint8))  # kill nodata edge
    mask = ((desp > thresh) & (evalid > 0)).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < a.min_area or area > a.max_area or max(w, h) > a.max_side:
            continue
        peak = float(desp[y:y + h, x:x + w].max())
        out.append((int(x), int(y), int(w), int(h), float(cents[i][0]), float(cents[i][1]), peak))
    return out


def main():
    args = parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())

    sampler = None
    if args.water_only:
        mask_path = run / "water_mask.tif"
        if mask_path.exists():
            import sys; sys.path.insert(0, str(Path(__file__).parent))
            from water_mask import WaterSampler
            sampler = WaterSampler(mask_path)
        else:
            print(f"  --water-only but no water_mask.tif in {run.name}; keeping all candidates")

    dets = []
    for chip in manifest["chips"]:
        with rasterio.open(run / chip["filename"]) as src:
            vv, vh = src.read(1), src.read(2)
            transform, crs = src.transform, src.crs
        for x, y, w, h, cx, cy, peak in detect_targets(vv, vh, args):
            ux, uy = transform * (cx, cy)
            lon, lat = warp_transform(crs, "EPSG:4326", [ux], [uy])
            if sampler is not None and not sampler.is_water(lon[0], lat[0]):
                continue
            dets.append({"chip": chip["filename"],
                         "bbox_pixel": [float(x), float(y), float(x + w), float(y + h)],
                         "confidence": round(peak / 255.0, 3),
                         "centroid_wgs84": [round(lon[0], 7), round(lat[0], 7)]})

    if args.min_depth > 0 and dets:
        import sys; sys.path.insert(0, str(Path(__file__).parent))
        from bathymetry import gebco_depths
        depths = gebco_depths([tuple(d["centroid_wgs84"]) for d in dets])
        kept = []
        for d, depth in zip(dets, depths):
            d["depth_m"] = depth
            if depth <= -args.min_depth:      # deeper than the gate (elevation more negative)
                kept.append(d)
        print(f"  deep-water gate (>={args.min_depth:.0f} m): {len(kept)}/{len(dets)} kept "
              f"(dropped {len(dets) - len(kept)} in shallow water)")
        dets = kept

    dets.sort(key=lambda d: d["confidence"], reverse=True)
    for i, d in enumerate(dets):
        d["detection_id"] = f"sar_{i:05d}"

    doc = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "run_dir": run.name, "model": "sar-saturation-dualpol",
           "params": {"k": args.k, "min_area": args.min_area, "max_area": args.max_area,
                      "max_side": args.max_side, "kernel": args.kernel},
           "count": len(dets), "detections": dets}
    tmp = run / "detections.json.tmp"
    tmp.write_text(json.dumps(doc, indent=2)); tmp.replace(run / "detections.json")
    n_chips = len({d["chip"] for d in dets})
    print(f"SAR seed: {len(dets)} candidates over {n_chips}/{len(manifest['chips'])} chips "
          f"-> {run / 'detections.json'}")


if __name__ == "__main__":
    main()
