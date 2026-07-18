"""Ingest the Gulf-of-Finland Sentinel-2 vessel dataset (Zenodo 15019034, CC-BY-4.0) into a YOLO
training export, reusing the live imagery pipeline so chips match deployment preprocessing exactly.

For each S2 product (one geopackage layer = one MGRS tile + date) it resolves the exact scene on
Planetary Computer, windowed-reads only the vessel-bearing 640 px tiles (B04/B03/B02, linear/3000
stretch = sat_fetch default), and rasterizes the geopackage boxes into YOLO labels. A tile is kept
only if it holds >=1 vessel with side >= --min-anchor-m (bias toward larger ships), and then EVERY
vessel in that tile is labeled (complete labels -> no undercount, no small-vessel forgetting).

Output: data/processed/training_exports/<tag>/{images,labels}/ + export_log.json, layered by
build_dataset.py exactly like the reviewed active-learning exports.
"""
import argparse, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
import requests
import geopandas as gpd
from pyogrio import list_layers
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from providers.pc import PlanetaryComputerS2Provider

CA = os.environ.get("REQUESTS_CA_BUNDLE") or True  # OpenSSL path works under Avast; GDAL /vsicurl does not

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
GPKG_DIR = REPO / "data/raw/external_ships/finland"
DEFAULT_OUT = REPO / "data/processed/training_exports/finland-s2"
TILE_PX = 640
STRETCH_MAX = 3000.0
BANDS = ["B04", "B03", "B02"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--min-anchor-m", type=float, default=60.0,
                   help="A 640px tile is kept only if it contains >=1 vessel with side >= this many "
                        "metres. Biases the set toward larger ships. All vessels in a kept tile are labeled.")
    p.add_argument("--min-valid", type=float, default=0.5,
                   help="Skip tiles whose non-nodata pixel fraction is below this (off-scene edges).")
    p.add_argument("--overwrite", action="store_true", help="Re-process products already on disk.")
    p.add_argument("--limit-products", type=int, default=None, help="Debug: only process first N products.")
    return p.parse_args()


def resolve_item(cat, tile, date):
    d = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    items = list(cat.search(collections=["sentinel-2-l2a"], datetime=f"{d}/{d}",
                            query={"s2:mgrs_tile": {"eq": tile}}).items())
    if not items:
        return None
    # multiple processing baselines of one acquisition -> newest processing (item.id timestamp) first
    items.sort(key=lambda it: it.id, reverse=True)
    return items[0]


def fetch_band(cat, item, tile, date, k):
    """Download one band via requests; on a 403 (SAS token expired on a long run) re-sign once."""
    for attempt in range(2):
        resp = requests.get(item.assets[k].href, verify=CA, timeout=300)
        if resp.status_code == 403 and attempt == 0:
            item = resolve_item(cat, tile, date)  # fresh SAS-signed href
            continue
        resp.raise_for_status()
        return resp.content
    raise RuntimeError(f"{tile} {date} band {k}: 403 after re-sign")


def main():
    args = parse_args()
    prov = PlanetaryComputerS2Provider()
    cat = prov.open_catalog(STAC, None)

    img_dir = args.out / "images"; lbl_dir = args.out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)

    products = []
    for gpkg in sorted(GPKG_DIR.glob("*.gpkg")):
        for lyr in [l[0] for l in list_layers(str(gpkg))]:
            products.append((gpkg, gpkg.stem, lyr))
    if args.limit_products:
        products = products[: args.limit_products]

    log = {"source": "Zenodo 15019034 (CC-BY-4.0)", "min_anchor_m": args.min_anchor_m,
           "tile_px": TILE_PX, "stretch": {"mode": "linear", "max": STRETCH_MAX}, "products": []}
    tot_chips = tot_boxes = 0
    size_hist = Counter()

    for gpkg, tile, date in products:
        # resume: skip products already exported (long runs outlive the SAS window; re-run to finish)
        done = sorted(img_dir.glob(f"finlandS2_{tile}{date}_*.png"))
        if done and not args.overwrite:
            nb = sum(sum(1 for ln in (lbl_dir / f"{p.stem}.txt").read_text().splitlines() if ln.strip())
                     for p in done)
            tot_chips += len(done); tot_boxes += nb
            log["products"].append({"tile": tile, "date": date, "item": "(resumed)",
                                    "chips": len(done), "boxes": nb, "gpkg_boxes": None})
            print(f"  {tile} {date}: resumed ({len(done)} chips on disk)")
            continue
        item = resolve_item(cat, tile, date)
        if item is None:
            print(f"  !! {tile} {date}: no PC item, skipped"); continue
        boxes = gpd.read_file(gpkg, layer=date)
        b = boxes.geometry.bounds.values  # minx,miny,maxx,maxy
        side_m = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])

        # Download each full band via requests (OpenSSL + CA bundle — the reliable path under a
        # TLS-intercepting AV; GDAL /vsicurl schannel reads are throttled to uselessness here) and
        # read it locally from memory, then tile in-memory. ~230 MB/band, 3 bands/product.
        full8 = None
        for bi, k in enumerate(BANDS):
            content = fetch_band(cat, item, tile, date, k)
            with rasterio.MemoryFile(content) as mf, mf.open() as src:
                if full8 is None:
                    T = src.transform; full8 = np.zeros((3, src.height, src.width), np.uint8)
                band = src.read(1)  # full uint16 DN
            full8[bi] = np.clip(band.astype(np.float32) / STRETCH_MAX * 255, 0, 255).astype(np.uint8)
            del band, content

        inv = ~T
        cx = (b[:, 0] + b[:, 2]) / 2; cy = (b[:, 1] + b[:, 3]) / 2
        cols, rows = inv * (cx, cy)
        ti = (cols // TILE_PX).astype(int); tj = (rows // TILE_PX).astype(int)

        # tiles containing >=1 anchor vessel (side >= min_anchor)
        by_tile = defaultdict(list)
        for k in range(len(boxes)):
            by_tile[(int(ti[k]), int(tj[k]))].append(k)
        keep_tiles = [t for t, idxs in by_tile.items()
                      if any(side_m[k] >= args.min_anchor_m for k in idxs)]

        p_chips = 0; p_boxes = 0
        for (bti, btj) in keep_tiles:
            c0, r0 = bti * TILE_PX, btj * TILE_PX
            sub = full8[:, r0:r0 + TILE_PX, c0:c0 + TILE_PX]
            rgb8 = np.zeros((3, TILE_PX, TILE_PX), np.uint8)  # pad edge tiles to 640
            rgb8[:, : sub.shape[1], : sub.shape[2]] = sub
            valid = np.any(rgb8 > 0, axis=0)
            if valid.mean() < args.min_valid:
                continue
            name = f"finlandS2_{tile}{date}_r{btj:02d}_c{bti:02d}"
            Image.fromarray(np.transpose(rgb8, (1, 2, 0)), "RGB").save(img_dir / f"{name}.png")

            lines = []
            for k in by_tile[(bti, btj)]:
                x0, y0, x1, y1 = b[k]
                pl, pt = inv * (x0, y1); pr, pbb = inv * (x1, y0)
                cxn = ((pl + pr) / 2 - c0) / TILE_PX; cyn = ((pt + pbb) / 2 - r0) / TILE_PX
                wn = (pr - pl) / TILE_PX; hn = (pbb - pt) / TILE_PX
                if not (0 <= cxn <= 1 and 0 <= cyn <= 1):
                    continue
                wn = min(max(wn, 1.0 / TILE_PX), 1.0); hn = min(max(hn, 1.0 / TILE_PX), 1.0)
                lines.append(f"0 {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f}")
                size_hist["large>=150" if side_m[k] >= 150 else
                          ("mid100-150" if side_m[k] >= 100 else
                           ("small60-100" if side_m[k] >= 60 else "tiny<60"))] += 1
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
            p_chips += 1; p_boxes += len(lines)

        tot_chips += p_chips; tot_boxes += p_boxes
        log["products"].append({"tile": tile, "date": date, "item": item.id,
                                "chips": p_chips, "boxes": p_boxes, "gpkg_boxes": len(boxes)})
        print(f"  {tile} {date}: {p_chips:>4} chips, {p_boxes:>5} labels  (of {len(boxes)} gpkg boxes)")

    log["totals"] = {"chips": tot_chips, "boxes": tot_boxes, "size_hist": dict(size_hist)}
    (args.out / "export_log.json").write_text(json.dumps(log, indent=1))
    print(f"\nDONE  {tot_chips} chips / {tot_boxes} labels -> {args.out}")
    print(f"label size mix: {dict(size_hist)}")


if __name__ == "__main__":
    main()
