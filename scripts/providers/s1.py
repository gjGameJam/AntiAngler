"""Planetary Computer Sentinel-1 GRD provider — free, all-weather SAR (Phase 2).

The dark-vessel workhorse: C-band SAR sees through cloud, day or night, and metallic
hulls are bright radar returns against near-black water — so it catches AIS-off vessels
optical misses (the actual product driver, see docs/ROADMAP.md Phase 2 + Phase 3 fusion).
Free and on the *same* Planetary Computer STAC as Sentinel-2, no credentials — so
``open_catalog`` is identical to pc.py; only the sensor physics differ.

What differs from optical (pc.py), all contained in this class:
  * No cloud concept. SAR carries no ``eo:cloud_cover``, so ``search_candidates`` drops the
    cloud query (that clause would return zero items on sentinel-1-grd) and ranks scenes
    newest-first instead; ``assess_aoi`` reports ``aoi_cloud=None`` and selection falls back
    to AOI coverage — which ``select_scene`` already handles.
  * Backscatter, not reflectance. ``scale_to_uint8`` converts to dB and percentile-stretches
    (a linear stretch on raw power is unusable — water and ships span orders of magnitude).
  * 1-2 polarizations -> pseudo-RGB. Dual-pol builds [VV, VH, VV/VH-ratio] so the same
    3-channel chip + detect_boats.py path work unchanged; single-pol falls back to grayscale.

Known limitation / next step: SAR needs its OWN detector (``best_sar.pt``) — the optical
best.pt encodes the wrong texture priors. Transfer from a public SAR ship set (xView3-SAR,
HRSID, SAR-Ship), not from the optical weights. This provider's job is producing correct,
self-describing chips; training the SAR model is downstream (docs/ROADMAP.md Phase 2 step 4).
"""
import numpy as np
import planetary_computer
import rasterio
from pystac_client import Client
from rasterio.warp import transform_bounds

from .base import Provider
from ._raster import decimated_read, gdal_env


def _to_db(x):
    """Linear backscatter -> decibels. Non-finite / non-positive pixels (nodata, dark-water
    zeros) are floored so log is finite; they're masked out of the stretch by ``valid`` anyway.
    Works whether the asset stores intensity or amplitude — the constant offset only shifts the
    window, which the percentile stretch re-centres."""
    x = np.where(np.isfinite(x) & (x > 0), x, 1e-5)
    return 10.0 * np.log10(x)


class Sentinel1GrdProvider(Provider):
    key = "s1"
    display_name = "Planetary Computer Sentinel-1 GRD (SAR)"
    manifest_tag = "planetary-computer-s1"
    # RTC (radiometrically terrain-corrected) is the default: its assets carry a real UTM CRS + 10 m
    # affine, so the S2 bbox-windowed-read/tile/geolocate path works unchanged. Raw GRD is in radar
    # ground-range geometry (crs=None, geolocation only via GCPs) and would need a warp/RPC reader —
    # pass --collection sentinel-1-grd only once that reader exists. RTC is also cleaner absolute σ⁰.
    default_collection = "sentinel-1-rtc"
    default_bands = "vv,vh"                 # polarizations, not RGB bands; IW coastal = vv,vh (EW open-ocean = hh,hv)
    default_resolution = 10.0
    default_stretch_max = 0.0              # unused: SAR uses a dB *percentile* stretch (see scale_to_uint8)
    known_limitations = (
        "Sentinel-1 SAR ~10-20 m: detects vessels as bright backscatter, so it needs a SAR-trained "
        "model (best_sar.pt), not the optical best.pt.",
        "Speckle and sea clutter cause false positives; bright land/coast returns are not vessels "
        "(mask to water for best results).",
    )

    def open_catalog(self, stac_url, sub_key):
        # Identical to pc.py: same PC STAC endpoint, same signing modifier (S1 blob assets need SAS tokens).
        if sub_key:
            planetary_computer.set_subscription_key(sub_key)
        return Client.open(stac_url, modifier=planetary_computer.sign_inplace)

    def search_candidates(self, catalog, collection, bbox, start, end, max_cloud, max_search):
        """Return up to max_search scenes covering bbox in the window, **newest-first**. max_cloud is
        ignored — SAR has no eo:cloud_cover, and querying it would return zero items."""
        dt = f"{start}/{end}"
        search = catalog.search(collections=[collection], bbox=list(bbox), datetime=dt)
        items = list(search.items())
        if not items:
            raise RuntimeError(f"No {collection} scenes for bbox={bbox} datetime={dt}. Widen --start/--end, "
                               "or try --collection sentinel-1-grd. Check the AOI has Sentinel-1 coverage "
                               "and that lon/lat order is correct.")
        # Newest-first: for patrol the most recent look at the AOI is the one we want. AOI coverage
        # (and, for optical, clarity) is assessed per-candidate later by select_scene.
        items.sort(key=lambda it: it.properties.get("datetime") or "", reverse=True)
        return items[: max(1, max_search)]

    def assess_aoi(self, item, bbox, insecure_tls, probe_res=60.0):
        """Coverage-only probe (SAR has no cloud mask). aoi_cloud stays None so select_scene ranks on
        coverage alone. Also records acquisition mode + polarizations for the manifest."""
        rec = {"id": item.id, "datetime": item.properties.get("datetime"),
               "eo:cloud_cover": None, "aoi_coverage": 0.0, "aoi_cloud": None,
               "sar:instrument_mode": item.properties.get("sar:instrument_mode"),
               "polarizations": item.properties.get("sar:polarizations")}
        pol = "vv" if "vv" in item.assets else ("hh" if "hh" in item.assets else None)
        if pol is None:
            return rec  # no co-pol amplitude asset to probe (unusual); leave coverage 0
        with rasterio.Env(**gdal_env(insecure_tls)):
            with rasterio.open(item.assets[pol].href) as src:
                sb = transform_bounds("EPSG:4326", src.crs, *bbox)
                data, _ = decimated_read(src, [1], sb, probe_res, native_res=10.0)
        b = data[0]
        valid = np.isfinite(b) & (b > 0)  # outside-swath is 0; calm water is low but > 0
        rec["aoi_coverage"] = round(float(valid.sum()) / b.size, 4) if b.size else 0.0
        return rec

    def read_aoi(self, item, bbox, bands, resolution, insecure_tls=False):
        """Return (stack[n_pol, H, W] float backscatter, transform, crs, valid_mask) clipped to bbox in
        the scene's native CRS. Requested polarizations the scene doesn't carry are skipped."""
        pols = [b.strip().lower() for b in bands.split(",") if b.strip()]

        stack = None
        transform = None
        crs = None
        with rasterio.Env(**gdal_env(insecure_tls)):
            arrays = []
            for pol in pols:
                if pol not in item.assets:
                    continue  # e.g. asked for vv,vh but this is an HH/HV (EW) scene
                with rasterio.open(item.assets[pol].href) as src:
                    if crs is None:
                        crs = src.crs
                        src_bounds = transform_bounds("EPSG:4326", crs, *bbox)
                    data, tf = decimated_read(src, [1], src_bounds, resolution, native_res=10.0)
                    if transform is None:
                        transform = tf
                    arrays.append(data[0].astype(np.float32))
            if not arrays:
                raise RuntimeError(f"None of the requested polarizations {pols} on item {item.id}; "
                                   f"available: {sorted(item.assets)}. Set --bands to match the scene's "
                                   "polarization (IW coastal = vv,vh; EW open-ocean = hh,hv).")
            stack = np.stack(arrays, axis=0)

        valid = np.any(np.isfinite(stack) & (stack > 0), axis=0)
        return stack, transform, crs, valid

    def scale_to_uint8(self, stack, valid, mode, stretch_max, stretch_pct):
        """Backscatter stack -> 8-bit **pseudo-RGB**, dB then percentile-stretched over the whole AOI.

        SAR spans orders of magnitude (dark water << bright ships), so the reflectance knobs
        (--stretch/--stretch-max) don't apply; only --stretch-pct is honored. Percentile-in-dB
        auto-adapts to the asset's units (amplitude vs intensity), so it can't silently produce a
        black chip the way a wrong fixed dB window would. Dual-pol -> [VV, VH, VV/VH]; single-pol ->
        grayscale. (A validated fixed dB window, e.g. [-25,0] for VV, can replace this once someone
        eyeballs real chips — docs/ROADMAP.md Phase 2 step 3.)"""
        lo_p, hi_p = (float(x) for x in stretch_pct.split(","))

        vv_db = _to_db(stack[0])
        if stack.shape[0] >= 2:
            vh_db = _to_db(stack[1])
            channels = [vv_db, vh_db, vv_db - vh_db]  # B = VV/VH cross-pol ratio in dB
            composite = "R=VV,G=VH,B=VV/VH (dB)"
        else:
            channels = [vv_db, vv_db, vv_db]           # single-pol -> grayscale
            composite = "R=G=B=VV (dB)"

        out = np.zeros((3, *vv_db.shape), dtype=np.uint8)
        los, his = [], []
        for i, ch in enumerate(channels):
            vals = ch[valid]
            lo = float(np.percentile(vals, lo_p)) if vals.size else 0.0
            hi = float(np.percentile(vals, hi_p)) if vals.size else 1.0
            hi = hi if hi > lo else lo + 1.0
            los.append(round(lo, 3))
            his.append(round(hi, 3))
            scaled = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255)
            out[i] = scaled.astype(np.uint8)

        out[:, ~valid] = 0  # keep nodata black
        params = {"mode": "sar-db-percentile", "pct": [lo_p, hi_p],
                  "db_lo": los, "db_hi": his, "composite": composite}
        return out, params
