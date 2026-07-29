"""Shared HTTP retry/backoff for the network-facing scripts (docs/AUDIT.md #4).

Every outbound call in this repo used to be single-shot, so one transient 5xx/timeout
failed the daily GFW cron, an SCL probe mid-fetch, or a long MarineCadastre download.
This centralises the policy so there is ONE place to tune it.

Design notes:

* ``import _http`` is **stdlib-only** — ``requests``/``urllib3`` are lazy-imported inside
  ``build_session``. The offline suite runs with no ``pip install`` (see
  ``.github/workflows/tests.yml``), so ``retry_policy`` is a pure dict-builder that CI can
  import and assert on directly.
* Underscore-prefixed *utility*, following ``providers/_raster.py``. The house rule that
  ingesters never import each other is about **data-stream** coupling; a retry policy is
  infrastructure, not a stream. AUDIT #7 defers DRYing ``env_or`` "until the helper needs
  to change in more than one place at once" — a backoff policy is exactly that helper, so
  it is shared rather than copy-pasted into five scripts.
* ``raise_on_status=False`` is deliberate: when retries are exhausted the *response* comes
  back, so callers keep their existing ``print(status)`` + ``raise_for_status()`` flow and
  the familiar ``HTTPError`` message instead of a bare urllib3 ``RetryError``.
"""

# Retried statuses: 429 (rate limit — opentopodata throttles to ~1 req/s and GFW rate-limits
# too) plus the transient 5xx family. Any other 4xx is a caller error and is never retried.
DEFAULT_STATUS_FORCELIST = (429, 500, 502, 503, 504)

# POST is included on purpose. urllib3's default allow-list is idempotent methods only, but
# every POST in this repo is query-shaped (GFW /v3/4wings/report and /v3/insights/vessels
# return a report; they create nothing), so replaying one is safe.
DEFAULT_ALLOWED_METHODS = ("HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST")

DEFAULT_TOTAL = 3
DEFAULT_BACKOFF_FACTOR = 1.0   # -> sleeps of 0s, 2s, 4s (see backoff_schedule)
BACKOFF_MAX = 120.0            # urllib3's own ceiling, mirrored here for backoff_schedule


def retry_policy(total=DEFAULT_TOTAL, backoff_factor=DEFAULT_BACKOFF_FACTOR,
                 status_forcelist=DEFAULT_STATUS_FORCELIST,
                 allowed_methods=DEFAULT_ALLOWED_METHODS):
    """Return the ``urllib3.util.Retry`` kwargs as a plain dict.

    Pure + stdlib on purpose: this is the piece the offline suite asserts on, so the policy
    cannot drift silently even though CI can't construct a real ``Retry``."""
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    if backoff_factor < 0:
        raise ValueError(f"backoff_factor must be >= 0, got {backoff_factor}")
    return {
        "total": total,
        "connect": total,
        "read": total,
        "status": total,
        "backoff_factor": backoff_factor,
        "status_forcelist": tuple(status_forcelist),
        "allowed_methods": frozenset(allowed_methods),
        "raise_on_status": False,
        "respect_retry_after_header": True,
    }


def backoff_schedule(policy):
    """The sleeps (seconds) urllib3 takes before each retry, for logging and for the offline
    test to pin the wall-clock cost. Mirrors ``Retry.get_backoff_time()``: the first retry is
    immediate, then it doubles. Descriptive only — urllib3 does the real sleeping."""
    bf = policy["backoff_factor"]
    return [0.0 if n <= 1 else min(BACKOFF_MAX, bf * (2 ** (n - 1)))
            for n in range(1, policy["total"] + 1)]


def describe(policy):
    """One-line summary for a script's request log, e.g.
    ``retry: up to 3x on 429/500/502/503/504 (waits 0s, 2s, 4s)``."""
    codes = "/".join(str(c) for c in policy["status_forcelist"])
    waits = ", ".join(f"{s:g}s" for s in backoff_schedule(policy))
    return f"retry: up to {policy['total']}x on {codes} (waits {waits})"


def build_session(policy=None, verify=None, headers=None):
    """A ``requests.Session`` with the retry policy mounted on http(s).

    Lazy-imports requests/urllib3 so this module stays stdlib-only at import time."""
    try:
        import requests
        from requests.adapters import HTTPAdapter
    except ImportError as exc:
        raise RuntimeError(
            "requests is not installed (pip install -r requirements.txt)") from exc
    from urllib3.util import Retry

    kwargs = dict(policy or retry_policy())
    try:
        retry = Retry(**kwargs)
    except TypeError:
        # urllib3 < 1.26 spells it method_whitelist. (Pinned envs run 1.26+, which accepts
        # allowed_methods; this keeps an older transitive resolve from breaking the cron.)
        kwargs["method_whitelist"] = kwargs.pop("allowed_methods")
        retry = Retry(**kwargs)

    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    if verify is not None:
        session.verify = verify
    if headers:
        session.headers.update(headers)
    return session
