# Codebase Audit — antipatterns & inefficiencies (priority order)

Scope: the four **early** scripts `scripts/{prep_polygons,view_polygons,gfw_fetch,sat_fetch}.py`,
`.github/workflows/gfw.yml`, `requirements.txt`. Reviewed 2026-07-07; **status refreshed 2026-07-23**
(findings updated to current state; anchors are function/symbol names — line numbers rot). The later
stage-2/3 and P3 scripts adopt the house style these findings established (`main()`-guard, atomic writes,
`env_or`/argparse, `--dry-run`). Secrets hygiene is clean (tokens read from env, stripped, never logged;
`--insecure-tls` is opt-in and documented; no hardcoded credentials).

## P1 — High (blocks planned stage 2–3 work, or risks the downstream data)

1. **No `main()` / `if __name__ == "__main__"` guard — orchestration runs at module import.**
   ~~All four scripts execute their orchestration at top level.~~ **`sat_fetch.py` FIXED (2026-07-15):**
   its orchestration (was `:409-529`) is wrapped in `def main(argv=None):` under
   `if __name__ == "__main__": raise SystemExit(main())`, and `parse_args(argv=None)` takes an explicit
   argv. Importing `sat_fetch` no longer runs `parse_args()` or a fetch; the seam helpers
   (`open_catalog`/`search_candidates`/`assess_aoi`/`select_scene`/`read_aoi`/`scale_to_uint8` + the
   tiling/IO helpers) are now importable and unit-testable. See `docs/ROADMAP.md` "Cross-cutting
   prerequisite" for the reusable-seam table.
   **Still to do (same fix, other scripts):** `gfw_fetch.py`, `prep_polygons.py`, and `view_polygons.py`
   still run their orchestration at module top level — apply the same `main()` guard when each is next
   touched (lower priority; none are imported by another module today).

2. **`prep_polygons.py` is fragile and off-house-style, and it writes the file everything else
   depends on** (STILL OPEN):
   - Writes the GeoPackage **non-atomically** (`mpas.to_file(out_file)`) — a crash mid-write
     corrupts `wdpa_marine_ia_ib.gpkg`, which `sat_fetch.py` and `view_polygons.py` both read. Every
     other script uses `.tmp` + `Path.replace()`; this one doesn't.
   - No existence check on the 3 input shapefiles → cryptic error if WDPA isn't
     downloaded. No argparse/`env_or` knobs (every other script has them). Dead commented code.
   **Fix:** guard inputs with a fail-fast `RuntimeError`; write atomically; adopt the `env_or`/argparse
   idiom for at least the IUCN/MARINE filter and output path.

## P2 — Medium (reliability & reproducibility)

3. **`requirements.txt` doesn't cover steps 1–2.** `pandas`, `geodatasets`, `matplotlib` (imported by
   `prep_polygons.py` / `view_polygons.py`) are unpinned — a fresh clone can't run the first two
   pipeline steps from `pip install -r requirements.txt` alone (CLAUDE.md admits "install manually").
   **Fix:** pin the three at `==` after verifying in the venv, per the repo's pin-once-verified rule.

4. **No network retry/backoff anywhere** (STILL OPEN). `gfw_fetch.py` is a single `requests.post`;
   `sat_fetch.py` does STAC search + N candidate SCL probes + COG reads with no retry; the P3 ingesters
   (`gfw_vessels.py`, `aisstream_fetch.py`, `marinecadastre_fetch.py --date`) are single-shot too. One
   transient 5xx/timeout fails the daily cron or a probe mid-run.
   **Fix:** `requests.Session` + `urllib3.util.Retry` for GFW; add `GDAL_HTTP_MAX_RETRY=3` /
   `GDAL_HTTP_RETRY_DELAY=2` to `sat_fetch._gdal_env` for the `/vsicurl` reads.

5. **CI runs no lint/compile/test** (PARTLY ADDRESSED). An **offline test suite now exists**
   (`scripts/tests/`, 81 tests covering the P3 stages — pure stdlib, no network/key/ML deps), but
   **CI still doesn't run it**: `gfw.yml` only runs `gfw_fetch.py`, so a syntax error in any other
   script — or a P3 regression — ships undetected.
   **Fix:** add a cheap job — `python -m py_compile scripts/*.py` + `python -m unittest discover -s
   scripts/tests -p 'test_*.py'` (optionally `ruff check`). No credentials needed, so it's free coverage.

6. **No `.env.example`.** The env surface (`GFW_API_TOKEN`, `PC_SDK_SUBSCRIPTION_KEY`, all `GFW_*` and
   `SAT_*` fallbacks) is only described in prose. **Fix:** add a committed `.env.example` listing every
   var with a comment and a safe placeholder.

## P3 — Low (polish; several are deliberate house-style trade-offs)

7. **Duplicated `env_or`** — the `env_or` helper is now copy-pasted across **~10 scripts** (`gfw_fetch`,
   `sat_fetch`, `detect_boats`, and every P3 ingester). A `scripts/_common.py` would DRY it, but the repo
   deliberately favors standalone one-script-per-stream files (no shared-import coupling). The "leave
   until a third copy" trigger has long passed; extraction is a **conscious deferral**, not an oversight
   — revisit only if the helper needs to change in more than one place at once.

8. **Chosen scene is probed (SCL) then re-opened for the RGB read** in `sat_fetch` — different assets,
   so not truly redundant, but it re-signs and recomputes `transform_bounds`. Negligible; not worth
   caching.

9. **Magic numbers in `sat_fetch`:** SCL `probe_res=60` (`assess_aoi`) and the `0.01` hard coverage
   floor are inline constants, not CLI knobs. Fine internally; promote to flags only if users need to
   tune probe accuracy.

10. **`gfw_fetch.py` string-indexed params** (`datasets[0]`, `filters[0]` in the request `params`) —
    brittle if extended to multiple datasets/filters, fine for the single-dataset call today.

11. **`print()`-only logging** across all scripts — no levels/timestamps. Intentional house style; a
    shared `logging` config is optional and would help CI debugging.

---

### Suggested order of attack
P1.1 (main guard) is done for `sat_fetch`; **the highest-value open items now are #5 (wire the existing
81-test suite into CI) and #6 (`.env.example`)** — quick reproducibility wins, one PR. Then #2
(`prep_polygons` hardening) and #4 (network retry). #7–#11 are conscious house-style trade-offs; leave.
These items are also indexed under `docs/ROADMAP.md` → "Remaining TODO items" (🧹 hygiene).
