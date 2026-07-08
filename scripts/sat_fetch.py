import argparse
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import planetary_computer
import rasterio
from affine import Affine
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from dotenv import load_dotenv

# The objective of this script:
# - Take a bounding box (explicit, or derived from an MPA in the WDPA GeoPackage)
# - Find a recent Sentinel-2 L2A scene that actually COVERS and is CLEAR over that AOI
#   (Planetary Computer STAC): candidates are probed with the SCL cloud mask over the AOI
#   itself, not just ranked by scene-wide eo:cloud_cover, which is misleading for small AOIs
# - Read true-color RGB (B04/B03/B02) clipped to the bbox, in the scene's native UTM
# - Tile it into georeferenced chips for a downstream ship-detection model
#
# --dry-run searches and probes candidates over the AOI and prints a ranked table without
# writing anything - use it to discover good --start/--end/--max-cloud for a new location.
#
# Known limitation: Sentinel-2 is 10 m GSD. That is marginal for small vessels
# (a 30 m boat is only ~3 px); fine for larger trawlers/cargo. SAR (Sentinel-1)
# would be better for small open-ocean targets — left as future work behind the
# same provider/sensor seam (open_catalog / search_items / read_aoi).
#
# Planetary Computer needs NO credentials. An optional PC_SDK_SUBSCRIPTION_KEY
# only raises rate limits.

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "sentinel2"
DEFAULT_GPKG = REPO_ROOT / "data" / "processed" / "wdpa_marine_ia_ib.gpkg"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")


def env_or(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def truthy(val):
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def parse_bbox_env(val):
    if val in (None, ""):
        return None
    parts = [p for p in re.split(r"[,\s]+", val.strip()) if p]
    if len(parts) != 4:
        raise RuntimeError(f"SAT_BBOX must have 4 numbers, got: {val!r}")
    return [float(p) for p in parts]


def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch a Sentinel-2 true-color scene for a bbox/MPA and tile it into georeferenced chips.")
    # --- Area of interest: explicit bbox OR an MPA lookup ---
    p.add_argument("--bbox", nargs=4, type=float,
                   metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
                   default=parse_bbox_env(env_or("SAT_BBOX", None)),
                   help="Explicit AOI in WGS84 lon/lat. Mutually exclusive with --wdpa-id/--wdpa-pid.")
    p.add_argument("--wdpa-id", type=float, default=(lambda v: float(v) if v else None)(env_or("SAT_WDPA_ID", None)),
                   help="Select an MPA by WDPAID from the GeoPackage.")
    p.add_argument("--wdpa-pid", default=env_or("SAT_WDPA_PID", None),
                   help="Select an MPA by WDPA_PID from the GeoPackage.")
    p.add_argument("--gpkg", type=Path, default=Path(env_or("SAT_GPKG", str(DEFAULT_GPKG))),
                   help="MPA GeoPackage to resolve --wdpa-id/--wdpa-pid against.")
    # --- Scene selection ---
    p.add_argument("--start", default=env_or("SAT_START", None),
                   help="YYYY-MM-DD. Defaults to 30 days before yesterday UTC.")
    p.add_argument("--end", default=env_or("SAT_END", None),
                   help="YYYY-MM-DD. Defaults to yesterday UTC.")
    p.add_argument("--max-cloud", type=float, default=float(env_or("SAT_MAX_CLOUD", 40)),
                   help="Scene-wide prefilter: only consider scenes with eo:cloud_cover below this percent.")
    p.add_argument("--max-search", type=int, default=int(env_or("SAT_MAX_SEARCH", 20)),
                   help="How many candidate scenes (least-cloudy first) to probe for AOI coverage/clarity.")
    p.add_argument("--min-coverage", type=float, default=float(env_or("SAT_MIN_COVERAGE", 0.80)),
                   help="Require a scene to cover at least this fraction of the AOI (0-1, SCL/nodata based).")
    p.add_argument("--max-aoi-cloud", type=float, default=float(env_or("SAT_MAX_AOI_CLOUD", 0.20)),
                   help="Require cloud/shadow OVER THE AOI (0-1, SCL based) at or below this. This is the real "
                        "clarity metric; scene-wide eo:cloud_cover can be misleading for a small AOI.")
    p.add_argument("--dry-run", action="store_true", default=truthy(env_or("SAT_DRY_RUN", "")),
                   help="Search + probe candidate scenes over the AOI, print a ranked table, and write nothing. "
                        "Use it to discover good --start/--end/--max-cloud for a location before fetching.")
    p.add_argument("--max-items", type=int, default=int(env_or("SAT_MAX_ITEMS", 5)),
                   help="How many probed candidate scenes to record in the manifest.")
    p.add_argument("--collection", default=env_or("SAT_COLLECTION", "sentinel-2-l2a"))
    p.add_argument("--stac-url", default=env_or("SAT_STAC_URL", STAC_URL))
    # --- Rendering ---
    p.add_argument("--bands", default=env_or("SAT_BANDS", "B04,B03,B02"),
                   help="Comma-separated RGB band order, or 'visual' for the pre-rendered TCI asset.")
    p.add_argument("--resolution", type=float, default=float(env_or("SAT_RESOLUTION", 10)),
                   help="Output GSD in metres (>=10; native S2 RGB is 10 m).")
    p.add_argument("--stretch", default=env_or("SAT_STRETCH", "linear"), choices=["linear", "percentile"],
                   help="Reflectance -> 8-bit conversion (ignored for --bands visual).")
    p.add_argument("--stretch-max", type=float, default=float(env_or("SAT_STRETCH_MAX", 3000)),
                   help="Linear-mode reflectance ceiling mapped to 255.")
    p.add_argument("--stretch-pct", default=env_or("SAT_STRETCH_PCT", "2,98"),
                   help="Percentile-mode low,high computed over the whole AOI.")
    # --- Tiling ---
    p.add_argument("--tile-size", type=int, default=int(env_or("SAT_TILE_SIZE", 640)))
    p.add_argument("--overlap", type=int, default=int(env_or("SAT_OVERLAP", 64)))
    p.add_argument("--min-valid-frac", type=float, default=float(env_or("SAT_MIN_VALID_FRAC", 0.05)),
                   help="Skip chips whose valid (non-nodata) fraction is below this.")
    p.add_argument("--max-bbox-deg", type=float, default=float(env_or("SAT_MAX_BBOX_DEG", 1.0)),
                   help="Reject an AOI whose lon or lat span exceeds this (guards against OOM).")
    # --- Output ---
    p.add_argument("--label", default=env_or("SAT_LABEL", None), help="Slug for the run directory name.")
    p.add_argument("--output-dir", type=Path, default=Path(env_or("SAT_OUTPUT_DIR", str(OUTPUT_DIR))))
    p.add_argument("--pc-subscription-key", default=env_or("PC_SDK_SUBSCRIPTION_KEY", None),
                   help="Optional Planetary Computer key for higher rate limits.")
    p.add_argument("--insecure-tls", action="store_true", default=truthy(env_or("SAT_INSECURE_TLS", "")),
                   help="Disable TLS verification for GDAL /vsicurl COG reads (GDAL_HTTP_UNSAFESSL=YES). "
                        "Needed only where a proxy/AV intercepts TLS and GDAL's schannel backend rejects the "
                        "re-signed chain. Off by default; the STAC/signing HTTPS still verifies via the "
                        "system/REQUESTS_CA_BUNDLE trust store.")
    return p.parse_args()


def slugify(text):
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug[:60] or "aoi"


def resolve_date_range(args):
    today = datetime.now(timezone.utc).date()
    end = args.end or (today - timedelta(days=1)).isoformat()
    start = args.start or (today - timedelta(days=31)).isoformat()
    return start, end


def bbox_from_wdpa(gpkg, wdpa_id, wdpa_pid):
    import geopandas as gpd  # heavy dep; only needed for the MPA-lookup path

    if not Path(gpkg).exists():
        raise RuntimeError(f"GeoPackage not found: {gpkg}")
    gdf = gpd.read_file(gpkg)
    if wdpa_id is not None:
        sel = gdf[gdf["WDPAID"] == float(wdpa_id)]
        key = f"WDPAID={wdpa_id}"
    else:
        sel = gdf[gdf["WDPA_PID"].astype(str) == str(wdpa_pid)]
        key = f"WDPA_PID={wdpa_pid}"
    if sel.empty:
        raise RuntimeError(f"No MPA matching {key} in {gpkg}")
    row = sel.iloc[0]
    minx, miny, maxx, maxy = (float(v) for v in sel.total_bounds)
    meta = {
        "WDPAID": float(row["WDPAID"]) if row.get("WDPAID") is not None else None,
        "WDPA_PID": str(row["WDPA_PID"]),
        "NAME": str(row["NAME"]),
    }
    return (minx, miny, maxx, maxy), meta["NAME"], meta


def resolve_bbox(args):
    has_bbox = args.bbox is not None
    has_wdpa = args.wdpa_id is not None or args.wdpa_pid is not None
    if has_bbox and has_wdpa:
        raise RuntimeError("Provide either --bbox or --wdpa-id/--wdpa-pid, not both.")
    if not has_bbox and not has_wdpa:
        raise RuntimeError("Provide an AOI: --bbox MINLON MINLAT MAXLON MAXLAT or --wdpa-id/--wdpa-pid.")
    if has_bbox:
        bbox = tuple(float(x) for x in args.bbox)
        return bbox, (args.label or "bbox"), None
    bbox, name, meta = bbox_from_wdpa(args.gpkg, args.wdpa_id, args.wdpa_pid)
    return bbox, (args.label or slugify(name)), meta


def validate_bbox(bbox, max_deg):
    minlon, minlat, maxlon, maxlat = bbox
    if not (-180 <= minlon < maxlon <= 180):
        raise RuntimeError(
            f"Bad longitudes {minlon}..{maxlon} (need -180 <= min < max <= 180). "
            "For an AOI that crosses the antimeridian (180 deg, e.g. Fiji/Kiribati/W Aleutians), "
            "fetch each side as its own run - the halves fall in different UTM zones anyway, so "
            "one scene cannot span them: e.g. --bbox 178 -18 180 -16 and --bbox -180 -18 -178 -16.")
    if not (-90 <= minlat < maxlat <= 90):
        raise RuntimeError(f"Bad latitudes {minlat}..{maxlat} (need -90 <= min < max <= 90).")
    w, h = maxlon - minlon, maxlat - minlat
    if w > max_deg or h > max_deg:
        raise RuntimeError(f"AOI span {w:.3f}x{h:.3f} deg exceeds --max-bbox-deg {max_deg}. "
                           "Shrink the AOI or raise the cap.")


def open_catalog(stac_url, sub_key):
    if sub_key:
        planetary_computer.set_subscription_key(sub_key)
    return Client.open(stac_url, modifier=planetary_computer.sign_inplace)


def search_candidates(catalog, collection, bbox, start, end, max_cloud, max_search):
    """Return up to max_search candidate scenes (least scene-cloud first) that pass the scene-wide
    cloud prefilter. AOI-specific coverage/clarity is assessed later per candidate."""
    dt = f"{start}/{end}"
    search = catalog.search(collections=[collection], bbox=list(bbox), datetime=dt,
                            query={"eo:cloud_cover": {"lt": max_cloud}})
    items = list(search.items())
    if not items:
        raise RuntimeError(f"No {collection} scenes for bbox={bbox} datetime={dt} scene-cloud<{max_cloud}%. "
                           "Widen --start/--end, raise --max-cloud, or check the AOI has Sentinel-2 "
                           "coverage (near-polar regions are sparse; verify lon/lat order).")
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100.0))
    return items[: max(1, max_search)]


def _gdal_env(insecure_tls):
    opts = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF")
    if insecure_tls:
        opts["GDAL_HTTP_UNSAFESSL"] = "YES"  # opt-in; TLS-intercepting proxy/AV only
    return opts


def _decimated_read(src, band_indexes, utm_bounds, resolution,
                    resampling=Resampling.bilinear, native_res=10.0):
    """Windowed read of one COG, clipped to utm_bounds, resampled toward `resolution` metres.
    native_res is the asset's own GSD (10 m for S2 RGB, 20 m for SCL) so decimation is correct."""
    win = from_bounds(*utm_bounds, transform=src.transform).round_offsets().round_lengths()
    scale = max(1.0, resolution / native_res)
    out_w = max(1, int(round(win.width / scale)))
    out_h = max(1, int(round(win.height / scale)))
    out_shape = (len(band_indexes), out_h, out_w)
    data = src.read(band_indexes, window=win, out_shape=out_shape,
                    boundless=True, fill_value=0, resampling=resampling)
    base_tf = src.window_transform(win)
    out_tf = base_tf * Affine.scale(win.width / out_w, win.height / out_h)
    return data, out_tf


# Sentinel-2 Scene Classification Layer (SCL) codes that hide the sea surface:
# 1 defective, 3 cloud shadow, 8/9 cloud (medium/high prob), 10 thin cirrus.
SCL_OBSCURED = (1, 3, 8, 9, 10)
SCL_NODATA = 0


def assess_aoi(item, bbox, insecure_tls, probe_res=60.0):
    """Probe how usable a scene is over THIS AOI (not the whole scene). Returns a record with
    aoi_coverage (fraction of AOI the scene actually contains) and aoi_cloud (fraction obscured by
    cloud/shadow, from the SCL mask via a cheap coarse read). aoi_cloud is None if there is no SCL."""
    rec = {"id": item.id, "datetime": item.properties.get("datetime"),
           "eo:cloud_cover": item.properties.get("eo:cloud_cover"),
           "aoi_coverage": 0.0, "aoi_cloud": None}
    with rasterio.Env(**_gdal_env(insecure_tls)):
        if "SCL" in item.assets:
            with rasterio.open(item.assets["SCL"].href) as src:
                ub = transform_bounds("EPSG:4326", src.crs, *bbox)
                data, _ = _decimated_read(src, [1], ub, probe_res,
                                          resampling=Resampling.nearest, native_res=20.0)
            scl = data[0]
            valid = scl != SCL_NODATA
            n_valid = int(valid.sum())
            rec["aoi_coverage"] = round(n_valid / scl.size, 4) if scl.size else 0.0
            if n_valid:
                obscured = np.isin(scl[valid], SCL_OBSCURED)
                rec["aoi_cloud"] = round(float(obscured.sum()) / n_valid, 4)
        else:
            # No SCL (e.g. --bands visual or another collection): coverage only, from band nodata.
            key = "visual" if "visual" in item.assets else next(iter(item.assets))
            with rasterio.open(item.assets[key].href) as src:
                ub = transform_bounds("EPSG:4326", src.crs, *bbox)
                data, _ = _decimated_read(src, [1], ub, probe_res, native_res=10.0)
            b = data[0]
            rec["aoi_coverage"] = round(float((b > 0).sum()) / b.size, 4) if b.size else 0.0
    return rec


def select_scene(candidates, bbox, insecure_tls, min_coverage, max_aoi_cloud):
    """Probe candidates in scene-cloud order; take the first that covers and is clear enough over the
    AOI. If none pass, return the best effort. Returns (item, chosen_rec, assessed_recs, fallback)."""
    assessed = []
    for it in candidates:
        rec = assess_aoi(it, bbox, insecure_tls)
        assessed.append((it, rec))
        ok_cov = rec["aoi_coverage"] >= min_coverage
        ok_cloud = rec["aoi_cloud"] is None or rec["aoi_cloud"] <= max_aoi_cloud
        cloud_str = "n/a" if rec["aoi_cloud"] is None else f"{rec['aoi_cloud']:.0%}"
        chosen_now = ok_cov and ok_cloud
        print(f"  probe {it.id[:44]:44s} cover={rec['aoi_coverage']:>4.0%} aoi_cloud={cloud_str:>4s}"
              + ("  <- selected" if chosen_now else ""))
        if chosen_now:
            return it, rec, [r for _, r in assessed], False

    def key(pair):
        r = pair[1]
        cl = 1.0 if r["aoi_cloud"] is None else r["aoi_cloud"]
        return (1 if r["aoi_coverage"] >= min_coverage else 0, -cl, r["aoi_coverage"])

    it, rec = max(assessed, key=key)
    return it, rec, [r for _, r in assessed], True


def read_aoi(item, bbox, bands, resolution, insecure_tls=False):
    """Return (stack[C,H,W], transform, crs, valid_mask) clipped to bbox in the scene's native UTM."""
    visual = bands.strip().lower() == "visual"
    band_keys = ["visual"] if visual else [b.strip() for b in bands.split(",") if b.strip()]

    stack = None
    transform = None
    crs = None
    with rasterio.Env(**_gdal_env(insecure_tls)):
        arrays = []
        for key in band_keys:
            if key not in item.assets:
                raise RuntimeError(f"Asset {key!r} not on item {item.id}; available: {sorted(item.assets)}")
            with rasterio.open(item.assets[key].href) as src:
                if crs is None:
                    crs = src.crs
                    utm_bounds = transform_bounds("EPSG:4326", crs, *bbox)
                idx = list(range(1, src.count + 1)) if visual else [1]
                data, tf = _decimated_read(src, idx, utm_bounds, resolution)
                if transform is None:
                    transform = tf
                arrays.append(data if visual else data[0])
        stack = arrays[0] if visual else np.stack(arrays, axis=0)

    valid = np.any(stack > 0, axis=0)
    return stack, transform, crs, valid


def scale_to_uint8(stack, valid, mode, stretch_max, stretch_pct):
    """Reflectance stack -> 8-bit RGB, stretched once over the whole AOI. uint8 input passes through."""
    if stack.dtype == np.uint8:
        return stack, {"mode": "passthrough"}

    out = np.zeros_like(stack, dtype=np.uint8)
    if mode == "linear":
        scaled = np.clip(stack.astype(np.float32) / float(stretch_max) * 255.0, 0, 255)
        out = scaled.astype(np.uint8)
        params = {"mode": "linear", "max": stretch_max}
    else:
        lo_p, hi_p = (float(x) for x in stretch_pct.split(","))
        los, his = [], []
        for b in range(stack.shape[0]):
            vals = stack[b][valid]
            lo = float(np.percentile(vals, lo_p)) if vals.size else 0.0
            hi = float(np.percentile(vals, hi_p)) if vals.size else 1.0
            hi = hi if hi > lo else lo + 1.0
            los.append(lo)
            his.append(hi)
            scaled = np.clip((stack[b].astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255)
            out[b] = scaled.astype(np.uint8)
        params = {"mode": "percentile", "pct": [lo_p, hi_p], "lo": los, "hi": his}

    out[:, ~valid] = 0  # keep nodata black
    return out, params


def tile_offsets(dim, tile, stride):
    if dim <= tile:
        return [0]
    offs = list(range(0, dim - tile + 1, stride))
    if offs[-1] != dim - tile:
        offs.append(dim - tile)  # shift the final tile inward for full coverage
    return offs


def iter_chips(stack_u8, valid, tile, overlap):
    c_count, H, W = stack_u8.shape
    stride = max(1, tile - overlap)
    for r in tile_offsets(H, tile, stride):
        for c in tile_offsets(W, tile, stride):
            h, w = min(tile, H - r), min(tile, W - c)
            chip = np.zeros((c_count, tile, tile), dtype=np.uint8)  # pad short edges with nodata=0
            sub = np.zeros((tile, tile), dtype=bool)
            chip[:, :h, :w] = stack_u8[:, r:r + h, c:c + w]
            sub[:h, :w] = valid[r:r + h, c:c + w]
            yield r, c, chip, sub


def chip_geo(transform, r, c, tile, crs):
    chip_tf = transform * Affine.translation(c, r)
    utm_bounds = array_bounds(tile, tile, chip_tf)  # (minx, miny, maxx, maxy)
    wgs84_bounds = transform_bounds(crs, "EPSG:4326", *utm_bounds)
    return chip_tf, utm_bounds, wgs84_bounds


def write_chip_geotiff(path, chip, crs, transform):
    count, height, width = chip.shape
    profile = dict(driver="GTiff", height=height, width=width, count=count, dtype="uint8",
                   crs=crs, transform=transform, nodata=0, compress="deflate",
                   tiled=True, blockxsize=256, blockysize=256)
    if count == 3:
        profile["photometric"] = "RGB"
    tmp = path.with_suffix(".tif.tmp")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(chip)
    tmp.replace(path)


def write_manifest(path, manifest):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(path)


args = parse_args()
bbox, label, wdpa_meta = resolve_bbox(args)
validate_bbox(bbox, args.max_bbox_deg)
start, end = resolve_date_range(args)

print("REQUEST:")
print(f"  aoi_bbox={bbox}")
print(f"  label={label}")
print(f"  date-range={start}/{end}")
print(f"  collection={args.collection}")
print(f"  select: scene-cloud<{args.max_cloud}% search={args.max_search} "
      f"min-cover={args.min_coverage:.0%} max-aoi-cloud={args.max_aoi_cloud:.0%}")
print(f"  bands={args.bands}  resolution={args.resolution}m  tile={args.tile_size}/{args.overlap}")

if args.insecure_tls:
    print("WARNING: --insecure-tls set - GDAL will NOT verify TLS certs on COG reads (GDAL_HTTP_UNSAFESSL=YES).")

catalog = open_catalog(args.stac_url, args.pc_subscription_key)
candidates = search_candidates(catalog, args.collection, bbox, start, end,
                               args.max_cloud, args.max_search)
print(f"\nFound {len(candidates)} candidate scene(s); probing AOI coverage/cloud...")

if args.dry_run:
    print("\nDRY RUN - ranked candidates over the AOI (nothing written):")
    print(f"  {'#':>2}  {'scene id':44s} {'date':10s} {'scene%':>6s} {'cover':>6s} {'aoicld':>7s}")
    recs = [assess_aoi(it, bbox, args.insecure_tls) for it in candidates]
    recs.sort(key=lambda r: (-r["aoi_coverage"], 1.0 if r["aoi_cloud"] is None else r["aoi_cloud"]))
    for i, r in enumerate(recs):
        sc = "n/a" if r["eo:cloud_cover"] is None else f"{r['eo:cloud_cover']:.0f}"
        ac = "n/a" if r["aoi_cloud"] is None else f"{r['aoi_cloud']:.0%}"
        print(f"  {i:>2}  {r['id'][:44]:44s} {(r['datetime'] or '')[:10]:10s} "
              f"{sc:>6s} {r['aoi_coverage']:>5.0%} {ac:>7s}")
    print("\nPick a window/threshold and re-run without --dry-run.")
    raise SystemExit(0)

item, chosen, assessed, fallback = select_scene(candidates, bbox, args.insecure_tls,
                                                args.min_coverage, args.max_aoi_cloud)
if chosen["aoi_coverage"] < 0.01:
    raise RuntimeError(f"No candidate scene overlaps the AOI (best coverage {chosen['aoi_coverage']:.0%}). "
                       "The bbox is likely outside Sentinel-2 coverage or the lon/lat order is swapped.")
if fallback:
    ac = "n/a" if chosen["aoi_cloud"] is None else f"{chosen['aoi_cloud']:.0%}"
    print(f"WARNING: no scene met --min-coverage {args.min_coverage:.0%} / --max-aoi-cloud "
          f"{args.max_aoi_cloud:.0%}; using best effort (cover={chosen['aoi_coverage']:.0%} "
          f"aoi_cloud={ac}). Widen --start/--end or relax the thresholds for a cleaner scene.")

print(f"\nSCENE: {item.id}")
print(f"  datetime={item.properties.get('datetime')}  scene_cloud={item.properties.get('eo:cloud_cover')}%"
      f"  aoi_cover={chosen['aoi_coverage']:.0%}"
      + ("" if chosen["aoi_cloud"] is None else f"  aoi_cloud={chosen['aoi_cloud']:.0%}"))

stack, transform, crs, valid = read_aoi(item, bbox, args.bands, args.resolution, args.insecure_tls)
print(f"  read {stack.shape} dtype={stack.dtype} crs={crs}  valid={valid.mean():.1%}")
if valid.sum() == 0:
    raise RuntimeError("AOI is entirely nodata for this scene (bbox may not intersect its footprint).")

stack_u8, stretch = scale_to_uint8(stack, valid, args.stretch, args.stretch_max, args.stretch_pct)

RUN_TS = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
GEN_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
run_dir = args.output_dir / f"{RUN_TS}_{label}"
tmp_dir = run_dir.with_suffix(run_dir.suffix + ".tmp") if run_dir.suffix else Path(str(run_dir) + ".tmp")
chips_dir = tmp_dir / "chips"
chips_dir.mkdir(parents=True, exist_ok=True)

chip_records = []
for r, c, chip, sub in iter_chips(stack_u8, valid, args.tile_size, args.overlap):
    if sub.mean() < args.min_valid_frac:
        continue
    chip_tf, utm_bounds, wgs84_bounds = chip_geo(transform, r, c, args.tile_size, crs)
    fname = f"chip_r{r:05d}_c{c:05d}.tif"
    write_chip_geotiff(chips_dir / fname, chip, crs, chip_tf)
    chip_records.append({
        "filename": f"chips/{fname}",
        "item_id": item.id,
        "pixel_window": {"row_off": r, "col_off": c, "width": args.tile_size, "height": args.tile_size},
        "utm_bounds": [round(v, 3) for v in utm_bounds],
        "wgs84_bounds": [round(v, 7) for v in wgs84_bounds],
        "valid_frac": round(float(sub.mean()), 4),
    })

if not chip_records:
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    raise RuntimeError("No chips passed --min-valid-frac; AOI may barely overlap the scene. "
                       "Lower --min-valid-frac or adjust the bbox.")

manifest = {
    "generated_utc": GEN_UTC,
    "provider": "planetary-computer",
    "collection": args.collection,
    "aoi_bbox_wgs84": [round(v, 7) for v in bbox],
    "wdpa": wdpa_meta,
    "scene_crs": str(crs),
    "tile_size": args.tile_size,
    "overlap": args.overlap,
    "resolution_m": args.resolution,
    "bands": args.bands,
    "stretch": stretch,
    "known_limitations": ["Sentinel-2 10 m GSD is marginal for vessels smaller than ~15 m."],
    "selection": {
        "max_cloud": args.max_cloud,
        "max_search": args.max_search,
        "min_coverage": args.min_coverage,
        "max_aoi_cloud": args.max_aoi_cloud,
        "fallback_used": fallback,
    },
    "chosen_item": {
        "id": item.id,
        "datetime": item.properties.get("datetime"),
        "eo:cloud_cover": item.properties.get("eo:cloud_cover"),
        "aoi_coverage": chosen["aoi_coverage"],
        "aoi_cloud": chosen["aoi_cloud"],
    },
    "candidates": assessed[: max(1, args.max_items)],
    "chips": chip_records,
}
write_manifest(tmp_dir / "manifest.json", manifest)

tmp_dir.replace(run_dir)
print(f"\nWrote {len(chip_records)} chips + manifest.json -> {run_dir}")
