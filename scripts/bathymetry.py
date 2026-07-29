"""GEBCO bathymetry via the free opentopodata public API (no key, no download, no auth).

Depth is returned in metres, negative = below sea level. Used to gate SAR vessel detection to DEEP
water: shallow water (roughly > -20..-50 m) returns strong bathymetric/wave SAR clutter that lights up
the pseudo-RGB and confuses vessel detection, so we keep only detections/chips over deep water. GEBCO
is ~450 m resolution — coarse but fine for a deep/shallow decision that varies over km scales.
"""
import os
import time

import _http  # stdlib-only at import time; lazy-imports requests inside build_session

API = "https://api.opentopodata.org/v1/gebco2020"


def gebco_depths(lonlats, ca=None, pause=1.1):
    """Batch GEBCO depth (m, negative = below sea level) for [(lon, lat), ...].
    <=100 locations/call; the public API is rate-limited to ~1 req/s.

    A whole SAR chip grid is many chunked calls, so one throttled 429 partway through used to
    lose the run (AUDIT #4). The retry policy covers 429 and honours the server's Retry-After
    header, which is exactly the signal this API sends; `pause` still paces the happy path."""
    if ca is None:
        ca = os.environ.get("REQUESTS_CA_BUNDLE") or True
    out = []
    session = _http.build_session(_http.retry_policy(), verify=ca)
    try:
        for i in range(0, len(lonlats), 100):
            chunk = lonlats[i:i + 100]
            locs = "|".join(f"{lat},{lon}" for lon, lat in chunk)
            r = session.get(API, params={"locations": locs}, timeout=60)
            r.raise_for_status()
            out += [res["elevation"] for res in r.json()["results"]]
            if i + 100 < len(lonlats):
                time.sleep(pause)
    finally:
        session.close()
    return out
