import argparse
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from dotenv import load_dotenv

from providers import available_providers, get_provider
from providers.pc import PlanetaryComputerS2Provider
from providers._raster import decimated_read as _decimated_read, gdal_env as _gdal_env

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
# (a 30 m boat is only ~3 px); fine for larger trawlers/cargo. Sentinel-1 SAR
# (--provider s1) plugs in behind the provider seam for free all-weather / dark-vessel
# coverage; each source lives in scripts/providers/. See docs/ROADMAP.md.
#
# main() is a thin orchestrator: it resolves the AOI, asks the selected Provider to
# search/probe/read the imagery, then does the provider-agnostic tiling + chip/manifest
# IO. The four source-specific steps (open_catalog / search_candidates / assess_aoi /
# scale_to_uint8) + read_aoi live on the Provider (scripts/providers/).
#
# Planetary Computer needs NO credentials. An optional PC_SDK_SUBSCRIPTION_KEY
# only raises rate limits.

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_DIR = RAW_DIR / "sentinel2"   # optical default; kept as a module constant for importers
DEFAULT_GPKG = REPO_ROOT / "data" / "processed" / "wdpa_marine_ia_ib.gpkg"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")

# The Sentinel-2 read/select seam now lives on a Provider (providers/pc.py); main() routes
# through whichever one --provider selects. These bound-method aliases keep the historical
# module-level names importable (`from sat_fetch import open_catalog, ...`) for tests/docs.
_PC = PlanetaryComputerS2Provider()
open_catalog = _PC.open_catalog
search_candidates = _PC.search_candidates
assess_aoi = _PC.assess_aoi
read_aoi = _PC.read_aoi
scale_to_uint8 = _PC.scale_to_uint8


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


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fetch a Sentinel-2 true-color scene for a bbox/MPA and tile it into georeferenced chips.")
    # --- Imagery source ---
    p.add_argument("--provider", default=env_or("SAT_PROVIDER", "pc"), choices=available_providers(),
                   help="Imagery source (both free, no credentials): 'pc' = Planetary Computer Sentinel-2 "
                        "optical 10 m (default, live); 's1' = Planetary Computer Sentinel-1 SAR (free, "
                        "all-weather, dark-vessel). The provider sets the defaults for "
                        "--collection/--bands/--resolution/--stretch-max when unset.")
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
    p.add_argument("--collection", default=env_or("SAT_COLLECTION", None),
                   help="Source collection/item-type. Defaults per --provider (pc: sentinel-2-l2a, "
                        "s1: sentinel-1-grd).")
    p.add_argument("--stac-url", default=env_or("SAT_STAC_URL", STAC_URL))
    # --- Rendering ---
    p.add_argument("--bands", default=env_or("SAT_BANDS", None),
                   help="Comma-separated RGB band order, or 'visual' for the pre-rendered TCI asset. "
                        "For SAR (s1) these are polarizations. Defaults per --provider (pc: B04,B03,B02; "
                        "s1: vv,vh).")
    p.add_argument("--resolution", type=float,
                   default=(lambda v: float(v) if v not in (None, "") else None)(env_or("SAT_RESOLUTION", None)),
                   help="Output GSD in metres. Defaults per --provider (pc: 10 m native S2 RGB).")
    p.add_argument("--stretch", default=env_or("SAT_STRETCH", "linear"), choices=["linear", "percentile"],
                   help="Reflectance -> 8-bit conversion (ignored for --bands visual).")
    p.add_argument("--stretch-max", type=float,
                   default=(lambda v: float(v) if v not in (None, "") else None)(env_or("SAT_STRETCH_MAX", None)),
                   help="Linear-mode reflectance ceiling mapped to 255. Defaults per --provider (pc: 3000).")
    p.add_argument("--stretch-pct", default=env_or("SAT_STRETCH_PCT", "2,98"),
                   help="Percentile-mode low,high computed over the whole AOI.")
    # --- Tiling ---
    p.add_argument("--tile-size", type=int, default=int(env_or("SAT_TILE_SIZE", 640)))
    p.add_argument("--overlap", type=int, default=int(env_or("SAT_OVERLAP", 64)))
    p.add_argument("--min-valid-frac", type=float, default=float(env_or("SAT_MIN_VALID_FRAC", 0.05)),
                   help="Skip chips whose valid (non-nodata) fraction is below this.")
    p.add_argument("--max-bbox-deg", type=float, default=float(env_or("SAT_MAX_BBOX_DEG", 1.0)),
                   help="Reject an AOI whose lon or lat span exceeds this (guards against OOM).")
    # --- Per-chip postprocessing (applied AFTER tiling; provider decides what it means) ---
    p.add_argument("--despeckle", default=env_or("SAT_DESPECKLE", "off"), choices=["off", "peak", "lee"],
                   help="SAR only (--provider s1): despeckle each chip after tiling. 'peak' is the "
                        "peak-preserving Lee filter the despeckled SAR models were trained on; 'lee' is "
                        "plain Lee, a MEASURED TRAP (W6: -45%% recall ceiling). Default off, because the "
                        "promoted best_sar.pt and every SAR run on disk are RAW. ⚠️ Train and inference "
                        "must use the SAME setting or the model is silently out of domain.")
    p.add_argument("--despeckle-window", type=int, default=int(env_or("SAT_DESPECKLE_WINDOW", 7)),
                   help="Despeckle filter window (must match training; the SAR tree used 7).")
    p.add_argument("--despeckle-peak-sigma", type=float, default=float(env_or("SAT_DESPECKLE_PEAK_SIGMA", 2.0)),
                   help="peak mode: leave pixels brighter than mean + this*sd untouched (training used 2.0).")
    # --- Output ---
    p.add_argument("--label", default=env_or("SAT_LABEL", None), help="Slug for the run directory name.")
    p.add_argument("--output-dir", type=Path,
                   default=(lambda v: Path(v) if v else None)(env_or("SAT_OUTPUT_DIR", None)),
                   help="Run root. Default: data/raw/<provider subdir> — sentinel2/ for --provider pc, "
                        "sentinel1/ for --provider s1, so the optical and SAR streams never share a dir.")
    p.add_argument("--pc-subscription-key", default=env_or("PC_SDK_SUBSCRIPTION_KEY", None),
                   help="Optional Planetary Computer key for higher rate limits.")
    p.add_argument("--insecure-tls", action="store_true", default=truthy(env_or("SAT_INSECURE_TLS", "")),
                   help="Disable TLS verification for GDAL /vsicurl COG reads (GDAL_HTTP_UNSAFESSL=YES). "
                        "Needed only where a proxy/AV intercepts TLS and GDAL's schannel backend rejects the "
                        "re-signed chain. Off by default; the STAC/signing HTTPS still verifies via the "
                        "system/REQUESTS_CA_BUNDLE trust store.")
    return p.parse_args(argv)


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


def select_scene(provider, candidates, bbox, insecure_tls, min_coverage, max_aoi_cloud):
    """Probe candidates in scene-cloud order; take the first that covers and is clear enough over the
    AOI. If none pass, return the best effort. Returns (item, chosen_rec, assessed_recs, fallback)."""
    assessed = []
    for it in candidates:
        rec = provider.assess_aoi(it, bbox, insecure_tls)
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


def default_output_dir(provider):
    """Run root for this provider: ``data/raw/<provider.output_subdir>``.

    One root per MODALITY so SAR runs stop landing in the optical ``sentinel2/`` dir. That
    separation is the upstream half of the rule that keeps SAR chips out of the optical
    training set (``export_labels.py`` writes SAR to ``sar_training_exports/`` because
    ``build_dataset.py`` sweeps every tag in ``training_exports/`` into ``data/training/``).

    Purely additive: existing run dirs are untouched (every downstream stage takes an explicit
    ``--run <path>``), and ``--output-dir`` / ``SAT_OUTPUT_DIR`` still override."""
    return RAW_DIR / (provider.output_subdir or "sentinel2")


def main(argv=None):
    """CLI entry point: pick a Provider, resolve the AOI, select a scene, tile it into chips.

    Kept thin — the source-specific work (open_catalog / search_candidates / assess_aoi /
    read_aoi / scale_to_uint8) lives on the Provider that --provider selects (scripts/providers/),
    while this function owns the provider-agnostic tail (scene-selection loop, tiling, chip/manifest
    IO). A new imagery source is a new Provider (subclass + register), not a change here.
    See docs/ROADMAP.md.
    """
    args = parse_args(argv)
    provider = get_provider(args.provider)
    # Fill any knob the user (and env) left unset from the provider's defaults, so
    # --provider planet gets 3 m / BGRN, a SAR provider gets VV+VH, etc.
    if args.collection is None:
        args.collection = provider.default_collection
    if args.bands is None:
        args.bands = provider.default_bands
    if args.resolution is None:
        args.resolution = provider.default_resolution
    if args.stretch_max is None:
        args.stretch_max = provider.default_stretch_max
    if args.output_dir is None:
        args.output_dir = default_output_dir(provider)

    bbox, label, wdpa_meta = resolve_bbox(args)
    validate_bbox(bbox, args.max_bbox_deg)
    start, end = resolve_date_range(args)

    print("REQUEST:")
    print(f"  provider={provider.key} ({provider.display_name})")
    print(f"  aoi_bbox={bbox}")
    print(f"  label={label}")
    print(f"  date-range={start}/{end}")
    print(f"  collection={args.collection}")
    print(f"  select: scene-cloud<{args.max_cloud}% search={args.max_search} "
          f"min-cover={args.min_coverage:.0%} max-aoi-cloud={args.max_aoi_cloud:.0%}")
    print(f"  bands={args.bands}  resolution={args.resolution}m  tile={args.tile_size}/{args.overlap}")
    print(f"  output_root={args.output_dir}")

    if args.insecure_tls:
        print("WARNING: --insecure-tls set - GDAL will NOT verify TLS certs on COG reads (GDAL_HTTP_UNSAFESSL=YES).")

    catalog = provider.open_catalog(args.stac_url, args.pc_subscription_key)
    candidates = provider.search_candidates(catalog, args.collection, bbox, start, end,
                                            args.max_cloud, args.max_search)
    print(f"\nFound {len(candidates)} candidate scene(s); probing AOI coverage/cloud...")

    if args.dry_run:
        print("\nDRY RUN - ranked candidates over the AOI (nothing written):")
        print(f"  {'#':>2}  {'scene id':44s} {'date':10s} {'scene%':>6s} {'cover':>6s} {'aoicld':>7s}")
        recs = [provider.assess_aoi(it, bbox, args.insecure_tls) for it in candidates]
        recs.sort(key=lambda r: (-r["aoi_coverage"], 1.0 if r["aoi_cloud"] is None else r["aoi_cloud"]))
        for i, r in enumerate(recs):
            sc = "n/a" if r["eo:cloud_cover"] is None else f"{r['eo:cloud_cover']:.0f}"
            ac = "n/a" if r["aoi_cloud"] is None else f"{r['aoi_cloud']:.0%}"
            print(f"  {i:>2}  {r['id'][:44]:44s} {(r['datetime'] or '')[:10]:10s} "
                  f"{sc:>6s} {r['aoi_coverage']:>5.0%} {ac:>7s}")
        print("\nPick a window/threshold and re-run without --dry-run.")
        return 0

    item, chosen, assessed, fallback = select_scene(provider, candidates, bbox, args.insecure_tls,
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

    stack, transform, crs, valid = provider.read_aoi(item, bbox, args.bands, args.resolution, args.insecure_tls)
    print(f"  read {stack.shape} dtype={stack.dtype} crs={crs}  valid={valid.mean():.1%}")
    if valid.sum() == 0:
        raise RuntimeError("AOI is entirely nodata for this scene (bbox may not intersect its footprint).")

    stack_u8, stretch = provider.scale_to_uint8(stack, valid, args.stretch, args.stretch_max, args.stretch_pct)

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    gen_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_dir = args.output_dir / f"{run_ts}_{label}"
    tmp_dir = run_dir.with_suffix(run_dir.suffix + ".tmp") if run_dir.suffix else Path(str(run_dir) + ".tmp")
    chips_dir = tmp_dir / "chips"
    chips_dir.mkdir(parents=True, exist_ok=True)

    # Per-chip postprocessing runs AFTER tiling (see Provider.postprocess_chip for why that
    # placement is load-bearing, not incidental). Providers ignore keys they do not understand.
    chip_postprocess = {"despeckle": args.despeckle,
                        "despeckle_window": args.despeckle_window,
                        "despeckle_peak_sigma": args.despeckle_peak_sigma}
    if args.despeckle != "off":
        print(f"  chip postprocess: despeckle={args.despeckle} window={args.despeckle_window} "
              f"peak_sigma={args.despeckle_peak_sigma}")
        if provider.key != "s1":
            print(f"  WARNING: --despeckle is a SAR filter and provider '{provider.key}' ignores it.")

    chip_records = []
    for r, c, chip, sub in iter_chips(stack_u8, valid, args.tile_size, args.overlap):
        if sub.mean() < args.min_valid_frac:
            continue
        chip = provider.postprocess_chip(chip, chip_postprocess)
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
        "generated_utc": gen_utc,
        "provider": provider.manifest_tag,
        "collection": args.collection,
        "aoi_bbox_wgs84": [round(v, 7) for v in bbox],
        "wdpa": wdpa_meta,
        "scene_crs": str(crs),
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "resolution_m": args.resolution,
        "bands": args.bands,
        "stretch": stretch,
        # Records which preprocessing the chips on disk actually got, so a run is self-describing
        # about whether it matches a raw-trained or a despeckled-trained checkpoint. Runs written
        # before 2026-08-06 have no such key -> treat a missing key as {"despeckle": "off"}.
        "chip_postprocess": chip_postprocess,
        "known_limitations": list(provider.known_limitations),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
