"""Offline tests for scripts/_http.py (the shared retry/backoff policy, docs/AUDIT.md #4).

Pure stdlib, no network. `_http` lazy-imports requests/urllib3 inside build_session, so the
policy itself (a plain dict-builder) is assertable in CI, which installs nothing. The one test
that constructs a real urllib3 Retry is skipped when requests is absent — it runs locally and
guards the kwarg names (`allowed_methods` vs the pre-1.26 `method_whitelist`).
Run either of:
    python scripts/tests/test_http.py
    python -m unittest scripts.tests.test_http
"""
import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import _http  # noqa: E402

try:
    import requests  # noqa: F401
    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False


class TestRetryPolicy(unittest.TestCase):
    def test_defaults(self):
        p = _http.retry_policy()
        self.assertEqual(p["total"], 3)
        self.assertEqual(p["backoff_factor"], 1.0)
        # connect/read/status must all be set, not just total: urllib3 counts them separately,
        # and a connect error is the exact failure mode the daily cron hits.
        for key in ("connect", "read", "status"):
            self.assertEqual(p[key], 3, f"{key} should track total")

    def test_retries_transient_statuses_only(self):
        forced = _http.retry_policy()["status_forcelist"]
        for code in (429, 500, 502, 503, 504):
            self.assertIn(code, forced)
        # Caller errors must NOT be retried. 422 especially: GFW's /v3/events 422s on a malformed
        # request (missing `offset`), and replaying that would just burn the backoff budget.
        for code in (400, 401, 403, 404, 422):
            self.assertNotIn(code, forced)

    def test_post_is_retryable(self):
        """Deliberate deviation from urllib3's idempotent-only default: the GFW POSTs
        (/v3/4wings/report, /v3/insights/vessels) are query-shaped and create nothing."""
        methods = _http.retry_policy()["allowed_methods"]
        self.assertIn("POST", methods)
        self.assertIn("GET", methods)

    def test_exhausted_retries_return_the_response(self):
        """raise_on_status=False keeps callers' `print(status)` + raise_for_status() flow
        working instead of surfacing a bare urllib3 RetryError."""
        self.assertFalse(_http.retry_policy()["raise_on_status"])

    def test_respects_retry_after(self):
        """opentopodata (bathymetry.py) throttles with 429 + Retry-After."""
        self.assertTrue(_http.retry_policy()["respect_retry_after_header"])

    def test_overrides(self):
        p = _http.retry_policy(total=5, backoff_factor=0.5, status_forcelist=(503,),
                               allowed_methods=("GET",))
        self.assertEqual(p["total"], 5)
        self.assertEqual(p["read"], 5)
        self.assertEqual(p["backoff_factor"], 0.5)
        self.assertEqual(p["status_forcelist"], (503,))
        self.assertEqual(p["allowed_methods"], frozenset({"GET"}))

    def test_zero_disables_retries(self):
        p = _http.retry_policy(total=0)
        self.assertEqual(p["total"], 0)
        self.assertEqual(_http.backoff_schedule(p), [])

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            _http.retry_policy(total=-1)
        with self.assertRaises(ValueError):
            _http.retry_policy(backoff_factor=-0.1)


class TestBackoffSchedule(unittest.TestCase):
    def test_default_schedule(self):
        """First retry is immediate, then urllib3 doubles: 0s, 2s, 4s = 6s worst case."""
        self.assertEqual(_http.backoff_schedule(_http.retry_policy()), [0.0, 2.0, 4.0])

    def test_scales_with_backoff_factor(self):
        self.assertEqual(_http.backoff_schedule(_http.retry_policy(backoff_factor=0.5)),
                         [0.0, 1.0, 2.0])

    def test_capped(self):
        sched = _http.backoff_schedule(_http.retry_policy(total=12, backoff_factor=10.0))
        self.assertTrue(all(s <= _http.BACKOFF_MAX for s in sched))
        self.assertEqual(sched[-1], _http.BACKOFF_MAX)


class TestDescribe(unittest.TestCase):
    def test_mentions_count_codes_and_waits(self):
        line = _http.describe(_http.retry_policy())
        self.assertIn("up to 3x", line)
        self.assertIn("429", line)
        self.assertIn("0s, 2s, 4s", line)


@unittest.skipUnless(HAVE_REQUESTS, "requests/urllib3 not installed (CI runs stdlib-only)")
class TestBuildSession(unittest.TestCase):
    def test_policy_is_accepted_by_urllib3(self):
        """Guards the kwarg names against a urllib3 version bump (allowed_methods was
        method_whitelist before 1.26)."""
        session = _http.build_session()
        try:
            adapter = session.get_adapter("https://example.invalid")
            retry = adapter.max_retries
            self.assertEqual(retry.total, 3)
            self.assertIn(429, retry.status_forcelist)
        finally:
            session.close()

    def test_mounts_both_schemes(self):
        session = _http.build_session()
        try:
            for url in ("https://example.invalid", "http://example.invalid"):
                self.assertEqual(session.get_adapter(url).max_retries.total, 3)
        finally:
            session.close()

    def test_verify_and_headers_applied(self):
        session = _http.build_session(verify="/tmp/ca.pem", headers={"X-Test": "1"})
        try:
            self.assertEqual(session.verify, "/tmp/ca.pem")
            self.assertEqual(session.headers["X-Test"], "1")
        finally:
            session.close()


class TestImportStaysStdlibOnly(unittest.TestCase):
    """CI runs the suite with NO pip install, so importing these must not pull requests.

    _http added an import line to several scripts; this is the property that would break
    silently if someone moved `import requests` back to a module top."""

    STDLIB_ONLY = ("_http", "gfw_fetch", "gfw_vessels", "marinecadastre_fetch", "bathymetry")

    def test_modules_import_without_requests(self):
        import builtins
        import importlib

        blocked = ("requests", "urllib3", "dotenv")
        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in blocked:
                raise ImportError(f"blocked for this test: {name}")
            return real_import(name, *args, **kwargs)

        saved = {m: sys.modules.pop(m, None) for m in self.STDLIB_ONLY}
        builtins.__import__ = guard
        try:
            for mod in self.STDLIB_ONLY:
                with self.subTest(module=mod):
                    importlib.import_module(mod)
        finally:
            builtins.__import__ = real_import
            for mod, prev in saved.items():
                if prev is not None:
                    sys.modules[mod] = prev
                else:
                    sys.modules.pop(mod, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
