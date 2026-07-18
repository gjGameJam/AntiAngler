"""Provider-agnostic rasterio helpers shared by every imagery provider.

These do the low-level windowed COG read that the Planetary Computer Sentinel-2 and
Sentinel-1 connectors all reuse unchanged. Kept in their own module (no provider or
sat_fetch imports) so a new provider can use them without depending on another
provider's code. ``sat_fetch.py`` re-exports them as ``_decimated_read`` / ``_gdal_env``
for back-compat.
"""
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import from_bounds


def gdal_env(insecure_tls):
    """GDAL config for /vsicurl COG reads. ``insecure_tls`` disables cert verification —
    opt-in, only for a TLS-intercepting proxy/AV whose re-signed chain GDAL's schannel
    backend rejects (the STAC/signing HTTPS still verifies via the normal trust store)."""
    # Sentinel-2 COGs are .tif; Sentinel-1 (GRD/RTC) measurement assets are .tiff. Both must be
    # allowed or /vsicurl short-circuits and reports the asset as non-existent before any fetch.
    opts = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff,.TIFF")
    if insecure_tls:
        opts["GDAL_HTTP_UNSAFESSL"] = "YES"  # opt-in; TLS-intercepting proxy/AV only
    return opts


def decimated_read(src, band_indexes, utm_bounds, resolution,
                   resampling=Resampling.bilinear, native_res=10.0):
    """Windowed read of one COG, clipped to ``utm_bounds``, resampled toward ``resolution``
    metres. ``native_res`` is the asset's own GSD (10 m for S2 RGB / S1 GRD, 20 m for SCL,
    ...) so the decimation factor is correct. Returns ``(data, transform)``."""
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
