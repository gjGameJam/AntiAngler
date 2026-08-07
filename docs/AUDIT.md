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
   ~~**Still to do (same fix, other scripts):** `gfw_fetch.py`, `prep_polygons.py`, and
   `view_polygons.py`.~~ **NOW DONE (2026-07-25):** all three wrap their orchestration in
   `def main(argv=None):` under `if __name__ == "__main__": raise SystemExit(main())` with
   `parse_args(argv=None)`. AST-verified: importing any of them runs no work (only imports/consts/defs
   at module level). `gfw_fetch.py` additionally soft-imports `dotenv` and lazy-imports `requests` in
   `main()`, so `import gfw_fetch` is stdlib-only and its pure helpers are now offline-tested
   (`test_gfw_fetch.py`).
   **Extended to the eval scripts (2026-08-03, ROADMAP W5):** `eval_holdout.py` and `conf_sweep.py` now
   use `parse_args(argv=None)` / `main(argv=None)` under the same `raise SystemExit(main())` guard, and
   `ultralytics` moved out of module scope into `load_yolo()`. Both are therefore importable
   stdlib-only, and the promotion-gate logic (center-distance matching, size strata, bootstrap CI,
   subset building, tree→checkpoint resolution) is covered by 50 offline tests in
   `test_eval_holdout.py` — including an `__import__`-blocking guard test, mirroring `test_http.py`'s,
   that fails if `ultralytics`/`torch` returns to a module top. **The house rule is now uniform: no
   script does work or imports a heavy dep at import time.**

2. **`prep_polygons.py` is fragile and off-house-style, and it writes the file everything else
   depends on** — ✅ **DONE (2026-07-25):**
   - Now writes the GeoPackage **atomically** (`write_gpkg_atomic`: sibling `.tmp.gpkg` → `Path.replace()`,
     tmp cleaned up on any failure) — a crash mid-write no longer corrupts `wdpa_marine_ia_ib.gpkg`,
     which `sat_fetch.py` and `view_polygons.py` both read.
   - `check_inputs()` fail-fasts with a clear `FileNotFoundError` (+ download pointer) if any of the 3
     shapefile parts is missing; `filter_marine_ia_ib()` validates the `MARINE`/`IUCN_CAT` columns
     exist and `main()` refuses to write an empty result. Added an argparse `--out` knob; dead
     commented code removed. (`view_polygons.py` similarly fail-fasts if the input GeoPackage is
     missing and writes `--save` PNGs atomically.)

## P2 — Medium (reliability & reproducibility)

2b. **`build_dataset.py` purged layered files by FILENAME PATTERN, not provenance — train/val leak.**
   ✅ **FIXED (2026-08-06).** `purge_al_files()` deleted only files whose name contained the `__`
   marker from `<run>__<chip>`. That marker is a *convention of `export_labels.py`*, not a guarantee:
   `ingest_s2ships_finland.py` writes `finlandS2_<tile><date>_r00_c02`, with no `__`. Those files
   therefore survived every purge, while `split_stems()` (a seeded shuffle over the current stem set)
   kept reassigning them as the set grew — so a chip assigned to `train` in one build and `val` in the
   next existed in **both**. Measured at discovery: **510 of 1034 Finland chips duplicated across
   train/val**, and `count_base()` (same pattern rule) miscounted them as untouched base photos, so
   "base kept" reported 1452/650 instead of the true 434/124/63 = 621.
   **Blast radius:** the holdout **TEST** split was never affected — it is populated by scene-name
   filters over `__`-named chips — so every `eval_holdout.py`/`conf_sweep.py` number and every
   promotion decision remains valid. What was contaminated is the **val** split YOLO uses for
   best-epoch selection, from `finetune-bigship` (2026-07-26) through `finetune-adriatic`.
   **Fix:** purge and count by **provenance** — `f.stem in al_stems`, where `al_stems` is every stem
   collected from the export tags — with the `__` glob retained so stale files from a deleted tag are
   still removed. **Lesson:** when a directory is rebuilt idempotently, the delete rule must be derived
   from the same source of truth as the write rule; deriving it from a *naming convention* silently
   breaks the moment a sibling ingester picks different names.
   **Regression test:** `scripts/tests/test_build_dataset.py` (6 tests, pure stdlib) — the key case was
   verified to FAIL against the pre-fix implementation before being committed to the suite.

3. **`requirements.txt` doesn't cover steps 1–2** — ✅ **DONE (2026-07-23).** `pandas==3.0.5`,
   `geodatasets==2026.5.1`, `matplotlib==3.11.1` (imported by `prep_polygons.py` / `view_polygons.py`)
   are now pinned in `requirements.txt`; the pip resolver confirmed they co-resolve with the existing
   `numpy==2.4.6` / `geopandas==1.1.4` pins (dry-run). A fresh clone can now run steps 1–2 from
   `pip install -r requirements.txt` alone.

4. **No network retry/backoff anywhere** — ✅ **DONE (2026-07-29).** The policy now lives in one place,
   `scripts/_http.py`: `retry_policy()` (a pure dict-builder) + `build_session()` (a `requests.Session`
   with `urllib3.util.Retry` mounted on http/https). Defaults: **3 attempts**, backoff factor 1.0
   (waits 0s/2s/4s), retrying **429 + 500/502/503/504** and connect/read errors only.
   - **Deliberate choices**, each locked by a test in `scripts/tests/test_http.py` (16):
     **POST is retryable** (urllib3's default allow-list is idempotent-only, but every POST here —
     `/v3/4wings/report`, `/v3/insights/vessels` — is query-shaped and creates nothing);
     **4xx other than 429 are never retried** (notably 422, which GFW returns for a malformed
     request — replaying it would just burn the backoff budget); and **`raise_on_status=False`**, so an
     exhausted retry still returns the response and callers keep their existing `print(status)` +
     `raise_for_status()` flow instead of surfacing a bare `RetryError`.
   - **Wired into:** `gfw_fetch.py` (the daily cron POST, + `--retries`/`--retry-backoff` knobs with
     `GFW_RETRIES`/`GFW_RETRY_BACKOFF` env fallbacks), `gfw_vessels.py` (all three v3 endpoints share one
     session per batch, same knobs under `GFWV_*`), `marinecadastre_fetch.py` (the large `--date` zip;
     note there is no range-resume — a retry restarts it), `bathymetry.py` (opentopodata throttles with
     429 + `Retry-After`, which the policy honours), and `ingest_s2ships_finland.py` (~1000 chips × 3
     bands; its application-level 403 SAS re-sign stays as the outer loop).
   - **`/vsicurl` COG reads:** `providers/_raster.py` `gdal_env()` now sets `GDAL_HTTP_MAX_RETRY=3` /
     `GDAL_HTTP_RETRY_DELAY=2` (both overridable per call), so a `sat_fetch` run — one STAC search + N
     SCL probes + a band read — survives a transient failure.
   - **The websocket case — ✅ CLOSED 2026-08-03 (ROADMAP W8).** `aisstream_fetch.py` reconnects with
     capped backoff (`reconnect_delay()`, 0/2/4/8s… capped at 30s) until a **wall-clock** deadline
     fixed at entry — so an outage shortens the data collected rather than extending the capture,
     which is what keeps a capture pairable with a known `scene.datetime`. The semantics question that
     had deferred it was answered by *recording* what happened (`elapsed_s`, `connected_s`,
     `reconnects`, `gaps`, `truncated`, `duplicates_dropped`) instead of silently claiming the full
     window. `--no-reconnect` restores the old behaviour; an AISStream `error` frame still fails fast.
   - **Does NOT help** the Avast schannel partial-chain failure — that is a cert-store problem; use the
     CA-bundle + `--insecure-tls` recipe in `docs/STATUS.md`.

5. **CI runs no lint/compile/test** — ✅ **DONE (2026-07-23).** `.github/workflows/tests.yml` runs on
   every push + PR: `python -m compileall -q scripts` (syntax gate for the whole repo, including the
   heavy ML/geo scripts — compileall parses, doesn't import) + `python -m unittest discover -s
   scripts/tests` (the offline suite, 280 tests as of 2026-08-06). Pure stdlib, no credentials/deps — free coverage. A
   syntax error in any script or a P3 regression now fails CI instead of shipping. (`ruff check` could
   be added later as an optional lint step.)

6. **No `.env.example`** — ✅ **DONE (2026-07-23).** A committed `.env.example` leads with the three
   credentials (`GFW_API_TOKEN`, `AISSTREAM_API_KEY`, optional `PC_SDK_SUBSCRIPTION_KEY`) and a grouped
   legend of the ~100 per-script env-var prefixes (`GFW_*`/`SAT_*`/`DET_*`/`FUSE_*`/`AIS_*`/`GFWV_*`/
   `MC_*`/`SCAN_*`/`REPORT_*`/…) with a `--help` pointer for the full list — an explicit per-flag dump
   would be a 100-line wall of noise. `.env` stays gitignored; `.env.example` is committed.

## P3 — Low (polish; several are deliberate house-style trade-offs)

7. **Duplicated `env_or`** — the `env_or` helper is now copy-pasted across **~10 scripts** (`gfw_fetch`,
   `sat_fetch`, `detect_boats`, and every P3 ingester). A `scripts/_common.py` would DRY it, but the repo
   deliberately favors standalone one-script-per-stream files (no shared-import coupling). The "leave
   until a third copy" trigger has long passed; extraction is a **conscious deferral**, not an oversight
   — revisit only if the helper needs to change in more than one place at once.
   **Precedent set 2026-07-29 (#4):** the retry policy *did* meet that trigger — one backoff policy has to
   change everywhere at once — so it was extracted to `scripts/_http.py` rather than copy-pasted into six
   scripts. That is the second underscore-prefixed shared **utility** (after `providers/_raster.py`), and
   it does not weaken the house rule, which is about **data-stream** coupling: an ingester still never
   imports another stream or the matcher. `env_or` stays duplicated — it is 3 lines and stable, so it has
   not met the trigger.

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
**All P1/P2 findings are now closed.** Reproducibility wins #3 (`requirements.txt`), #5 (CI test job) and
#6 (`.env.example`) closed 2026-07-23; #1 (`main()` guards on all four scripts) and #2 (`prep_polygons`
hardening) closed 2026-07-25; **#4 (network retry/backoff) closed 2026-07-29** via `scripts/_http.py` +
the GDAL `/vsicurl` retry config, with the websocket (`aisstream_fetch.py`) reconnect case explicitly
left open as a semantics decision — and then **closed 2026-08-03** (ROADMAP W8) by recording what the
capture actually did rather than guessing. #7–#11 remain conscious house-style trade-offs; leave.
