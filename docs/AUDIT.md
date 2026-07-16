# Codebase Audit — antipatterns & inefficiencies (priority order)

Scope: `scripts/{prep_polygons,view_polygons,gfw_fetch,sat_fetch}.py`, `.github/workflows/gfw.yml`,
`requirements.txt`. Reviewed 2026-07-07. Each finding has a file:line anchor and a one-line fix.
Secrets hygiene is clean (tokens read from env, stripped, never logged; `--insecure-tls` is opt-in
and documented; no hardcoded credentials).

## P1 — High (blocks planned stage 2–3 work, or risks the downstream data)

1. **No `main()` / `if __name__ == "__main__"` guard — orchestration runs at module import.**
   ~~All four scripts execute their orchestration at top level.~~ **`sat_fetch.py` FIXED (2026-07-15):**
   its orchestration (was `:409-529`) is wrapped in `def main(argv=None):` under
   `if __name__ == "__main__": raise SystemExit(main())`, and `parse_args(argv=None)` takes an explicit
   argv. Importing `sat_fetch` no longer runs `parse_args()` or a fetch; the seam helpers
   (`open_catalog`/`search_candidates`/`assess_aoi`/`select_scene`/`read_aoi`/`scale_to_uint8` + the
   tiling/IO helpers) are now importable and unit-testable. See `docs/ROADMAP.md` "Cross-cutting
   prerequisite" for the reusable-seam table.
   **Still to do (same fix, other scripts):** `gfw_fetch.py:57-111`, `prep_polygons.py:22-58`,
   `view_polygons.py:22-43` still run at top level — apply the same `main()` guard when each is next
   touched (lower priority; none are imported by another module today).

2. **`prep_polygons.py` is fragile and off-house-style, and it writes the file everything else
   depends on.** `prep_polygons.py:16-58`:
   - Writes the GeoPackage **non-atomically** (`:57` `mpas.to_file(out_file)`) — a crash mid-write
     corrupts `wdpa_marine_ia_ib.gpkg`, which `sat_fetch.py` and `view_polygons.py` both read. Every
     other script uses `.tmp` + `Path.replace()`; this one doesn't.
   - No existence check on the 3 input shapefiles (`:16-27`) → cryptic error if WDPA isn't
     downloaded. No argparse/`env_or` knobs (every other script has them). Dead commented code
     (`:32-34`).
   **Fix:** guard inputs with a fail-fast `RuntimeError`; write atomically; adopt the `env_or`/argparse
   idiom for at least the IUCN/MARINE filter and output path.

## P2 — Medium (reliability & reproducibility)

3. **`requirements.txt` doesn't cover steps 1–2.** `pandas`, `geodatasets`, `matplotlib` (imported by
   `prep_polygons.py` / `view_polygons.py`) are unpinned — a fresh clone can't run the first two
   pipeline steps from `pip install -r requirements.txt` alone (CLAUDE.md admits "install manually").
   **Fix:** pin the three at `==` after verifying in the venv, per the repo's pin-once-verified rule.

4. **No network retry/backoff anywhere.** `gfw_fetch.py:96` is a single `requests.post`; `sat_fetch.py`
   does STAC search + N candidate SCL probes + COG reads with no retry. One transient 5xx/timeout fails
   the daily cron or a probe mid-run.
   **Fix:** `requests.Session` + `urllib3.util.Retry` for GFW; add `GDAL_HTTP_MAX_RETRY=3` /
   `GDAL_HTTP_RETRY_DELAY=2` to `sat_fetch._gdal_env` for the `/vsicurl` reads.

5. **CI runs no lint/compile/test and never exercises `sat_fetch.py`.** `gfw.yml` only runs
   `gfw_fetch.py`; a syntax error in any other script ships undetected.
   **Fix:** add a cheap job — `python -m py_compile scripts/*.py` (optionally `ruff check`). `sat_fetch`
   needs no credentials, so even a `--help` smoke test is free coverage.

6. **No `.env.example`.** The env surface (`GFW_API_TOKEN`, `PC_SDK_SUBSCRIPTION_KEY`, all `GFW_*` and
   `SAT_*` fallbacks) is only described in prose. **Fix:** add a committed `.env.example` listing every
   var with a comment and a safe placeholder.

## P3 — Low (polish; several are deliberate house-style trade-offs)

7. **Duplicated `env_or`** (`gfw_fetch.py:18`, `sat_fetch.py:43`). A `scripts/_common.py` would DRY it,
   but the repo deliberately favors standalone one-script-per-stream files. Leave until a third copy
   appears, then extract.

8. **Chosen scene is probed (SCL) then re-opened for the RGB read** in `sat_fetch` — different assets,
   so not truly redundant, but it re-signs and recomputes `transform_bounds`. Negligible; not worth
   caching.

9. **Magic numbers in `sat_fetch`:** SCL `probe_res=60` (`assess_aoi`) and the `0.01` hard coverage
   floor are inline constants, not CLI knobs. Fine internally; promote to flags only if users need to
   tune probe accuracy.

10. **`gfw_fetch.py` string-indexed params** (`datasets[0]`, `filters[0]` at `:68,74`) — brittle if
    extended to multiple datasets/filters, fine for the single-dataset call today.

11. **`print()`-only logging** across all scripts — no levels/timestamps. Intentional house style; a
    shared `logging` config is optional and would help CI debugging.

---

### Suggested order of attack
P1.1 (main guard) → P1.2 (prep_polygons hardening) unblock and de-risk everything downstream; do them
alongside the stage-2 build. P2.3–P2.6 are quick reproducibility wins (one PR). P3 is opportunistic.
