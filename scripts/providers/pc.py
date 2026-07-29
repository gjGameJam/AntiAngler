"""Planetary Computer Sentinel-2 L2A provider — the default, live imagery source.

This is the reference implementation of the ``Provider`` interface: the exact Sentinel-2
code ``sat_fetch.py`` has always run, now behind the seam so the Sentinel-1 SAR connector
(``s1.py``) and others can be dropped in alongside it. Needs no credentials; an optional
``PC_SDK_SUBSCRIPTION_KEY`` only raises rate limits. See ``docs/ROADMAP.md``.
"""
import numpy as np
import planetary_computer
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

from .base import Provider
from ._raster import decimated_read, gdal_env

# Sentinel-2 Scene Classification Layer (SCL) codes that hide the sea surface:
# 1 defective, 3 cloud shadow, 8/9 cloud (medium/high prob), 10 thin cirrus.
SCL_OBSCURED = (1, 3, 8, 9, 10)
SCL_NODATA = 0


class PlanetaryComputerS2Provider(Provider):
    key = "pc"
    display_name = "Planetary Computer Sentinel-2 L2A"
    manifest_tag = "planetary-computer"
    default_collection = "sentinel-2-l2a"
    default_bands = "B04,B03,B02"
    default_resolution = 10.0
    default_stretch_max = 3000.0
    output_subdir = "sentinel2"
    known_limitations = ("Sentinel-2 10 m GSD is marginal for vessels smaller than ~15 m.",)

    def open_catalog(self, stac_url, sub_key):
        if sub_key:
            planetary_computer.set_subscription_key(sub_key)
        return Client.open(stac_url, modifier=planetary_computer.sign_inplace)

    def search_candidates(self, catalog, collection, bbox, start, end, max_cloud, max_search):
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

    def assess_aoi(self, item, bbox, insecure_tls, probe_res=60.0):
        """Probe how usable a scene is over THIS AOI (not the whole scene). Returns a record with
        aoi_coverage (fraction of AOI the scene actually contains) and aoi_cloud (fraction obscured by
        cloud/shadow, from the SCL mask via a cheap coarse read). aoi_cloud is None if there is no SCL."""
        rec = {"id": item.id, "datetime": item.properties.get("datetime"),
               "eo:cloud_cover": item.properties.get("eo:cloud_cover"),
               "aoi_coverage": 0.0, "aoi_cloud": None}
        with rasterio.Env(**gdal_env(insecure_tls)):
            if "SCL" in item.assets:
                with rasterio.open(item.assets["SCL"].href) as src:
                    ub = transform_bounds("EPSG:4326", src.crs, *bbox)
                    data, _ = decimated_read(src, [1], ub, probe_res,
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
                    data, _ = decimated_read(src, [1], ub, probe_res, native_res=10.0)
                b = data[0]
                rec["aoi_coverage"] = round(float((b > 0).sum()) / b.size, 4) if b.size else 0.0
        return rec

    def read_aoi(self, item, bbox, bands, resolution, insecure_tls=False):
        """Return (stack[C,H,W], transform, crs, valid_mask) clipped to bbox in the scene's native UTM."""
        visual = bands.strip().lower() == "visual"
        band_keys = ["visual"] if visual else [b.strip() for b in bands.split(",") if b.strip()]

        stack = None
        transform = None
        crs = None
        with rasterio.Env(**gdal_env(insecure_tls)):
            arrays = []
            for key in band_keys:
                if key not in item.assets:
                    raise RuntimeError(f"Asset {key!r} not on item {item.id}; available: {sorted(item.assets)}")
                with rasterio.open(item.assets[key].href) as src:
                    if crs is None:
                        crs = src.crs
                        utm_bounds = transform_bounds("EPSG:4326", crs, *bbox)
                    idx = list(range(1, src.count + 1)) if visual else [1]
                    data, tf = decimated_read(src, idx, utm_bounds, resolution)
                    if transform is None:
                        transform = tf
                    arrays.append(data if visual else data[0])
            stack = arrays[0] if visual else np.stack(arrays, axis=0)

        valid = np.any(stack > 0, axis=0)
        return stack, transform, crs, valid

    def scale_to_uint8(self, stack, valid, mode, stretch_max, stretch_pct):
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
