# ROADMAP — improving the detector & building the MPA-violation product

_Snapshot: 2026-07-14 (O5/O6 eval refresh 2026-07-25). Durable handoff for Phases 0(remainder)–3.
Read `docs/STATUS.md` first (live model, per-run eval numbers, open problems), then this. Original
research/rationale: `~/.claude/plans/what-will-actually-improve-swirling-truffle.md` (user machine, not
in repo)._

## Where we are

The detector (`data/training/weights/best.pt`, YOLOv8n-**P2** on Sentinel-2 10 m RGB) is the
**`finetune-smallvessel`** model, deployed at **imgsz 1024, conf 0.15**. The active-learning data loop
has now closed **both** generalization gates: cross-*region* (the Singapore-only P2 had ~0 recall on
fresh open-ocean → real open-ocean data, Fujairah+Gibraltar, lifted Port Said held-out recall
0.00→0.71) and cross-*type* (the openocean model recalled only 0.31 on small fishing vessels → real
small-vessel Med data, Cyclades 219 + Egadi 223, lifted Alonnisos held-out recall **0.31→0.67 @ conf
0.15**, precision held, no forgetting). The dominant early lever was **inference resolution** (1–5 px →
2–10 px), then the **P2 stride-4 head stacked**, then **real, diverse training data** proved the
highest-leverage remaining move — twice. **All of Phase 0 is done, and the loop is the proven engine.**
Next is scaling data diversity further (smaller artisanal boats, more regions, a 2nd small-vessel
holdout to confirm the 0.67 magnitude) plus the imagery/AIS phases below.

**Eval refresh (2026-07-25):** the two sandbox-runnable optical levers are now spent — `eval_holdout.py`
+ `conf_sweep.py` on the live `best.pt` confirmed the deploy point (imgsz 1024 / conf 0.15; Alonnisos
P0.40/R0.70, Port Said P0.97/R0.69) and showed **plain imgsz 1280 is a non-win** (Port Said flat,
Alonnisos recall 0.70→0.49 — the big-ship lift needs `--augment` TTA, not raw imgsz). conf 0.15 stays the
F1-peak operating point. So the remaining optical accuracy work is the **data-driven O3/O4 levers**
(same-region hard negatives + more diverse positives + a 2nd holdout), all 🏠 do-from-home
(`docs/IMPROVEMENT_PLAN.md` O5/O6).

**The remaining ceiling is physical:** at 10 m GSD a 20 m boat ≈ 2 px, near the optical information
floor. Free model tricks are largely spent, and there is **no free global higher-resolution optical**
source to raise it. So the next real gains come from **free SAR imagery**
(Sentinel-1, Phase 2 — all-weather, finds dark/AIS-off vessels), **more/diverse training data**, and
**reframing the goal as AIS-mismatch detection** (Phase 3), which is the actual product.

### Target architecture — how the phases combine
```
DETECT                       CORRELATE                  CONFIRM
S2 optical  (P0, live)     ┐
Sentinel-1 SAR (P2, built) ├─► vs AIS/GFW (P3) ─► dark? ─► S1×S2 cross-check + human review (free)
                           ┘         │
                                     └─► violation_events (schema in CLAUDE.md)
```
Each phase ships value alone. **Recommended order: P2 (Sentinel-1 SAR — provider built, train the SAR
detector next) and P3 (AIS fusion).** P3 is the highest strategic value and works even with a
modest-recall detector; P2 adds free all-weather/night/dark coverage. Both free imagery providers
(`pc`, `s1`) are built behind the `sat_fetch.py` seam (see the prerequisite section).

---

## Cross-cutting prerequisite (done before the imagery phases): AUDIT #1 — the `main()`-guard refactor ✅ DONE (2026-07-15)

**Done.** The former module-top-level orchestration is now wrapped in
`def main(argv=None):` guarded by `if __name__ == "__main__": raise SystemExit(main())`, and
`parse_args(argv=None)` takes an optional argv (so importers/tests can supply args explicitly instead
of hijacking `sys.argv`). Importing `sat_fetch` no longer runs `parse_args()` or a fetch.

**What is now importable and reusable** (compose these for a new provider — none of them touch argv or
do any I/O beyond their one job):

| Seam function | Signature | Role in the fetch |
|---|---|---|
| `open_catalog(stac_url, sub_key)` | → STAC `Client` | Open + auth the catalog (PC signing modifier) |
| `search_candidates(catalog, collection, bbox, start, end, max_cloud, max_search)` | → `[Item]` | Scene-wide prefilter, least-cloud-first |
| `assess_aoi(item, bbox, insecure_tls, probe_res=60.0)` | → `{aoi_coverage, aoi_cloud, …}` | AOI-specific coverage/clarity probe (SCL) |
| `select_scene(candidates, bbox, insecure_tls, min_coverage, max_aoi_cloud)` | → `(item, rec, assessed, fallback)` | Pick first scene that passes, else best-effort |
| `read_aoi(item, bbox, bands, resolution, insecure_tls)` | → `(stack[C,H,W], transform, crs, valid)` | Windowed COG read clipped to bbox in native UTM |
| `scale_to_uint8(stack, valid, mode, stretch_max, stretch_pct)` | → `(uint8_stack, params)` | Reflectance → 8-bit RGB stretch |

Plus the provider-agnostic tiling/geo/IO helpers that a new provider reuses **unchanged**:
`_decimated_read`, `_gdal_env`, `tile_offsets`, `iter_chips`, `chip_geo`, `write_chip_geotiff`,
`write_manifest`, `validate_bbox`, `resolve_date_range`.

**How a new provider composes them:** the S2-specific pieces are exactly `open_catalog` (auth),
`search_candidates` (the `eo:cloud_cover` query), `assess_aoi` (the SCL mask), and `scale_to_uint8`
(the reflectance stretch). A Sentinel-1 (or other) connector supplies its own version of *those four*
plus a `read_aoi`-shaped reader, and feeds the same `iter_chips`/`chip_geo`/`write_*` tail — so the
run-dir/manifest data contract downstream stages read is identical. See the per-phase "generalize the
S2 leaks" notes below for the exact fields to swap.

**Update (2026-07-15): the provider registry is built, and both free providers are live.** The six seam
functions moved onto a `Provider` object (`scripts/providers/base.py` interface, `providers/pc.py` =
Sentinel-2 optical, `providers/s1.py` = Sentinel-1 SAR), `--provider {pc,s1}` selects one, and `main()`
routes through it while staying the orchestrator. The seam names above are still importable from
`sat_fetch` (back-compat bound-method aliases). A new source is now a new `Provider` subclass + one
`register()` line. **`providers/s1.py` (Sentinel-1 GRD SAR) is fully implemented** (Phase 2 provider,
below).

**Verified:** `import sat_fetch` runs no fetch; the six seams + `main` are callable; `--help` exits 0;
the no-AOI path still raises + exits 1. This was AUDIT.md item #1; it unblocks both imagery phases and
unit testing (helpers can now be called on synthetic STAC items without a network fetch).

---

## Phase 0 remainder — P2 detection head ✅ DONE (promoted 2026-07-12, `finetune-p2-1024`)

**Result: it stacked.** The P2 stride-4 head trained fresh at 1024 (warm-started backbone via
`train.py --pretrained best`) beats the prior plain-n model on the holdout at conf 0.20 (recall
0.73→0.77, precision 0.76→0.87, FP/chip 1.5→0.75; F1-peak conf 0.15 gives 0.85/0.85/1.0), with no
base-domain regression. Promoted at conf 0.20. **Memory gotcha:** batch 16 @1024 with the P2 head
exhausts memory → silent torch abort mid-epoch; train at **`--batch 4`** (see STATUS ³). Caveat: thin
single-location holdout — needs a fresh cross-location scene to confirm magnitude. **Steps below kept
for the record / reproduction.**

The one small-object *architecture* lever not yet pulled. A **P2 head adds a stride-4 detection layer**
(160×160 grid at imgsz 640 / 256×256 at 1024) so tiny objects stop being sub-cell — the standard SOD
fix, and it may **stack** with the imgsz-1024 gain.

**Steps:**
1. Create a custom model yaml (e.g. `data/training/yolov8n-p2.yaml`) — copy Ultralytics' `yolov8-p2.yaml`
   head onto the `n` scale (adds the P2/stride-4 branch to the Detect layer).
2. Train fresh from that arch on the current dataset (P2 needs a from-scratch-ish head; can't load the
   plain-`n` head weights): `train.py --weights data/training/yolov8n-p2.yaml --imgsz 1024 --mosaic 0.5
   --close-mosaic 15 --epochs 30` (add `train.py` support for a `.yaml` weights spec if `resolve_weights`
   rejects it — it currently treats non-alias strings as a `Path`, which should already work).
3. Evaluate: `eval_holdout.py --old data/training/weights/best.pt --old-imgsz 1024 --new <run>/best.pt
   --new-imgsz 1024`, then `conf_sweep.py` to pick the operating point. Promote only if holdout recall
   improves without FP/chip blowing out (STATUS.md recipe).

**Lower-priority alternative:** base `n→s`. More capacity, but overfitting-prone on ~11 real positive
chips and it doesn't address the sub-cell problem — try only if P2 plateaus.
**Verification / effort:** same eval harness; **medium** (custom yaml + a fresh train).

---

## Phase 2 — Sentinel-1 SAR (free, complementary modality) — provider ✅ + real-chip validation ✅ (2026-07-15)

**Goal:** all-weather, day/night, **dark metallic-hull** detection — the wide-area workhorse GFW/Skylight
use (ESA/Paolo 2024: ~75% of industrial fishing vessels untracked). SAR floor ~15–20 m, so it
*complements* optical (catches larger/metallic vessels in cloud/night), not replaces it.

**Status:** the imagery provider is built, wired end to end, AND now **validated on real chips**
(2026-07-15, Fujairah anchorage) — `scripts/providers/s1.py`, registered as `--provider s1`, default
`--collection sentinel-1-rtc`, `--bands vv,vh`. It drops the cloud query (ranks scenes newest-first),
reports `aoi_cloud=None`, and converts backscatter → 8-bit **pseudo-RGB** `[VV, VH, VV/VH]` via a **dB
percentile** stretch. A real S1 fetch over the *same AOI* as the existing Fujairah S2 scene (dates ~1
week apart) confirms the pipeline end to end: **bright backscatter blobs coincide with the
optically-detected vessels**, and SAR even shows vessels the optical detector missed (side-by-side in the
SAR run dir's `comparison_overview.png` / `comparison_detail.png`).

**Two fixes were required to get a real read to succeed (both done):**
1. **`.tiff` extension allowlist** — `_raster.py`'s `/vsicurl` `CPL_VSIL_CURL_ALLOWED_EXTENSIONS` was
   `.tif,.TIF` only, but S1 measurement assets are `.tiff`, so GDAL short-circuited and reported the
   asset "does not exist" before any fetch. Now `.tif,.TIF,.tiff,.TIFF` (additive; S2 unaffected).
2. **Default collection GRD → RTC** — raw `sentinel-1-grd` is in radar ground-range geometry
   (`crs=None`, geolocation only via 189 GCPs, identity transform) and is **incompatible** with the S2
   bbox→UTM windowed-read/tile path (`transform_bounds(..., crs=None)` crashes). `sentinel-1-rtc`
   (radiometrically terrain-corrected) is on a **real UTM grid** (EPSG:326xx, clean 10 m affine, no
   GCPs) — drop-in with the existing machinery and radiometrically cleaner σ⁰. This corrects the earlier
   "GRD is adequate" call in step 2 below. Pass `--collection sentinel-1-grd` only once a GCP/warp reader
   exists.

TLS: the Avast MITM is handled by the certifi+Avast CA bundle + `--insecure-tls` recipe, same as S2.
**Caveat:** the validated RTC scene covered only ~82% of the AOI (a no-data swath wedge on the east
edge — detections there have no SAR data); for full coverage pick an orbit/date that fully covers the AOI
or mosaic adjacent scenes. **Deliberate deviation from step 3:** kept a *percentile*-in-dB stretch, not a
fixed `[-25,0] dB` window — it auto-adapts and the real chips look correct, so keep it unless a fixed
window proves better on more scenes. **Remaining for Phase 2 = train a separate SAR detector (step 4).**

**Prerequisites:** the `main()`-guard refactor ✅ (AUDIT #1). **Sentinel-1 GRD IS on Planetary Computer** (`sentinel-1-grd`), so the
STAC/catalog side is close to a `--collection` change — but the sensor differs fundamentally.

**Steps (1-3 ✅ done in `providers/s1.py`; 4 remaining):**
1. **Drop the cloud logic (it silently breaks search otherwise):** SAR items carry **no** `eo:cloud_cover`
   property, so the existing `search_candidates` query `{"eo:cloud_cover": {"lt": max_cloud}}`
   returns **zero** items on `sentinel-1-grd` — you must drop that clause for the SAR
   provider (rank by acquisition date / relative-orbit instead). `assess_aoi` already degrades gracefully
   with no SCL (coverage-only from band nodata), so `aoi_cloud` just stays `None` and
   selection falls back to coverage — that part needs no change.
2. **Choose GRD vs RTC — RESOLVED: use RTC (now the default).** PC has both `sentinel-1-grd`
   (ground-range detected) and `sentinel-1-rtc` (radiometrically terrain-corrected, analysis-ready σ⁰).
   The original note here favored GRD ("water is flat"), but that missed that **PC's GRD assets carry no
   CRS** — they're in radar geometry with GCPs (`crs=None`, identity transform), so the S2-style
   `transform_bounds`+`from_bounds` windowed read breaks. **RTC is on a real UTM grid (EPSG:326xx, 10 m
   affine)** and is drop-in with the existing reader, so the provider defaults to it. Use GRD only after
   adding a GCP/warp-based reader.
3. **Backscatter, not reflectance:** replace `scale_to_uint8`'s reflectance stretch with **dB
   normalization** — `10*log10(x)` on the linear-power DN, then clip to roughly **[-25, 0] dB** for VV
   (ships are bright returns near 0 dB, calm water very dark ≲ -20 dB) and map to 0–255. Sweep the clip
   window on real chips. Band model is **VV+VH (2-band)**, not RGB, so:
   - `read_aoi`: request assets `["vv","vh"]`; they're separate single-band COGs like S2's bands, so the
     existing per-band loop works.
   - **Chip writer:** `write_chip_geotiff` sets `photometric="RGB"` only when `count==3` — a 2-band chip
     writes fine as a plain 2-band GeoTIFF, **but the YOLO detector wants 3 channels**. Standard trick:
     build a **pseudo-RGB [VV, VH, VV/VH-ratio]** (or VV, VH, VV−VH) 3-band chip so the same
     `detect_boats.py` inference path and the 3-channel model input work unchanged.
4. 🏠 **DO FROM HOME — Train a separate SAR detector.** Needs local GPU/compute with the machine kept
   awake (a real 30-epoch run; the remote sandbox can't train — see `docs/STATUS.md` on the smoke-test
   sleep-inflation). Different modality → its own `best_sar.pt`; **transfer from public SAR ship
   datasets** — **xView3-SAR** (arXiv 2206.00897, the closest analog: Sentinel-1 dual-pol, maritime,
   dark-vessel labels), HRSID, or SAR-Ship — rather than the optical base weights, which encode the wrong
   texture priors. Optionally speckle-filter (Lee/Refined-Lee) before tiling; test whether it helps recall
   on 1–5 px targets or just erases them. The reviewed SAR training set already exists
   (`data/processed/sar_training_exports/deep/`, 206 boxes / 112 hard-neg) + the build/train flow is
   smoke-validated — this step is the full training run + promote.

**Data contract:** same run-dir shape; chips carry CRS+affine so geolocation is unchanged. Detections
feed the same Phase 3 fusion. Manifest already records `provider`/`collection`/`bands`, so a SAR run is
self-describing.
**Verification:** `--dry-run` `sentinel-1-grd` over an AOI → fetch a VV/VH scene → eyeball the dB-stretched
pseudo-RGB → run the SAR model; confirm it flags vessels optical missed under cloud/night (fetch S1 and S2
over the same AOI within a day and compare).
**Effort: medium-hard** (new normalization + 2→3 band packing + a separate model + its own labels).

---

## Phase 3 — AIS dark-vessel reframe (the operational payoff) — all stages built ✅, live validation remaining (see `docs/PHASE3_PLAN.md`)

**Goal / the product:** **dark-vessel = detection − AIS**. An in-MPA detection with no matching AIS
track = a candidate violation. This converts "impossible perfect recall" into achievable "mismatch
precision" — and it *filters the detector's false positives* (the open-water FP problem from conf-0.20),
so it makes the current model good enough. This is the project's stated purpose.

**Status: the whole sandbox-buildable pipeline is BUILT & offline-tested (2026-07-23).** Every stage that
can be written and validated without live credentials is done; the only remaining work is 🏠 do-from-home
(live validation + real runs). The actionable checklist + definition of done is in **`docs/PHASE3_PLAN.md`**;
per-script contracts + first-run setup are in `CLAUDE.md`.

Built — six scripts, each its own module (ingestion never imports the matcher — house rule):
- **`fuse_violations.py`** (pure stdlib) — the matcher. Per in-MPA detection, dark-vs-matched via a
  **velocity-aware feasible-movement envelope** (`reach = sog·Δt + slack`, haversine ≤ reach, max-speed
  fallback when sog is missing/zero — never a fixed radius, which over-flags fast vessels and under-flags
  during AIS gaps). Emits `violation_events.json` (+ dark `.geojson`). Optional `--vessels` GFW enrichment
  lifts a matched score when GFW shows the identified vessel fishing / IUU-listed.
- **`aisstream_fetch.py`** — live AISStream websocket adapter → normalized AIS contract (`data/raw/ais/`).
  Live-only (near-real-time workflow).
- **`marinecadastre_fetch.py`** — historical AIS: normalizes a NOAA MarineCadastre archive CSV into the
  same AIS contract, AOI/time-filtered (for fusing *past* scenes over US waters).
- **`gfw_vessels.py`** — GFW per-vessel identity + fishing hours (+ optional Insights: IUU / AIS-off /
  no-take-MPA fishing; `--region-bbox`/`--wdpa-id` makes fishing_probability MPA-specific) → vessel-records
  (`data/raw/gfw_vessels/`), read by `fuse_violations.py --vessels`.
- **`report_violations.py`** — read-only operator output: ranks `violation_events.json` into a CSV + a
  self-contained HTML report by `violation_score`.
- **`scan_violations.py`** — thin batch driver sequencing detect → fuse → (GFW-enrich) → report via each
  stage's `main(argv)`.

The `violation_events.json` schema is pinned in `docs/PIPELINE.md`. **104 offline tests** (synthetic
fixtures + a committed end-to-end contract guard; no network / key / ML deps). 🏠 **DO FROM HOME — not yet
run against the live AISStream / GFW services** (the remote sandbox blocks those hosts and has no
key/token). Live validation of the calls + a first real violation from real imagery is the crux of
finishing the phase — `docs/PHASE3_PLAN.md` Workstream 1.

**Why it makes the modest detector good enough:** dark-vessel = detection − AIS *inverts the metric* — a
low-conf false positive sitting on a matching AIS track is filtered out, a real dark vessel with no AIS
survives. The product cares about *mismatch precision* (which fusion improves), not the raw *recall* the
10 m GSD caps. This is the payoff that justifies Phases 0–2.

---

## Remaining TODO items (consolidated, 2026-07-23)

The technical pipeline (Phases 0/2/3) is built and offline-tested (104 tests). What's left is grouped by
what it needs. This consolidates the gaps flagged in `CLAUDE.md`'s capability→code status table, the
phase sections above, `docs/PHASE3_PLAN.md`, `docs/IMPROVEMENT_PLAN.md`, and `docs/AUDIT.md` — read
those for detail; this is the single index.

### 🏠 Do-from-home (need credentials / GPU — cannot run in the remote sandbox)
- **Live-validate the AISStream + GFW calls + produce a first real violation.** Top priority — it
  unblocks trust in the whole chain. `docs/PHASE3_PLAN.md` Workstream 1 (endpoint-verification checklist).
- **Train the SAR detector `best_sar.pt`** (Phase 2 step 4). The reviewed SAR set
  (`data/processed/sar_training_exports/deep/`) + the build/train flow exist; needs a GPU run + promote.
- **Live runs** of `scan_violations.py` (batch driver) and the `marinecadastre_fetch.py --date` archive
  download (the `--csv <local file>` path already runs offline).

### 🟢 Feasible offline (buildable in the sandbox; not yet built)
- **Alerting interface** — the design proposal's final promised output. Today `report_violations.py`
  emits a ranked CSV/HTML; an alerts layer would filter high-`violation_score` / dark / IUU events into a
  digest (JSON/markdown/console, or a watch mode). Pure stdlib over `violation_events.json`.
- **AIS-track MPA geofencing** — flag *cooperative* vessels transmitting from inside a no-take IUCN Ia/Ib
  MPA (AIS position ∈ MPA), independent of a satellite detection — the matched/identified counterpart to
  the dark signal. bbox test is stdlib; precise point-in-polygon needs shapely.
- **Local speed-based fishing heuristic** — the summary's "infer fishing via GFW *or local ML*"; only the
  GFW path exists today. Lower priority (GFW covers it); a speed/loitering heuristic removes the GFW
  dependency for the fishing signal.
- **Cross-modality S1×S2 validation** — `docs/IMPROVEMENT_PLAN.md` S7: confirm SAR flags dark/large
  vessels the optical detector missed under cloud/night.
- **Skylight dark-call cross-check** — `docs/PHASE3_PLAN.md` 3B (free ground-truth for our dark calls;
  gated on Skylight API access).
- **Detector-accuracy levers** — `docs/IMPROVEMENT_PLAN.md`: **O3** same-region hard negatives (top
  precision lever) and **O4** more diverse real positives + a 2nd small-vessel holdout (top recall lever)
  are the remaining open levers — both need 🏠 fetch + GPU + human review. (**O5** inference/tiling tuning
  and **O6** eval-harness hardening were **measured/exercised 2026-07-25**: imgsz 1024 / conf 0.15 is the
  confirmed operating point, imgsz 1280 a non-win — see the "Where we are" eval-refresh note.)

### 🧹 Repo hygiene / reproducibility (`docs/AUDIT.md`, still open)
- ✅ **DONE (2026-07-23)** — a cheap **CI job** (`.github/workflows/tests.yml`): `compileall` + the
  104-test offline suite on every push/PR. (`gfw.yml` still only runs `gfw_fetch.py` on schedule.)
- ✅ **DONE (2026-07-23)** — committed **`.env.example`** (credentials + grouped per-script env-var legend)
  and **pinned `pandas`/`geodatasets`/`matplotlib`** in `requirements.txt` (resolver-verified).
- ✅ **DONE (2026-07-25)** — `main()`-guard + input/atomic-write hardening for `gfw_fetch.py` /
  `prep_polygons.py` / `view_polygons.py` (all wrap orchestration in `main(argv=None)`; fail-fast input
  validation; atomic `.tmp`→`replace` writes with cleanup; `gfw_fetch` now stdlib-importable + offline-tested).
- Still open: network retry/backoff on the GFW + `/vsicurl` calls (AUDIT #4).
- Cosmetic: SAR runs write under `data/raw/sentinel2/` — add a `data/raw/sentinel1/` output dir so the
  modalities don't mix (`docs/IMPROVEMENT_PLAN.md` cross-cutting).

### ⛔ Deliberately out of scope ($0-budget, local)
The cloud-native stack in the design proposal (Kubernetes, Kafka /
Kinesis, PostgreSQL+PostGIS, Airflow, Prometheus/Grafana) and paid data (VMS, Spire satellite AIS,
commercial VHR/SAR per-alert tasking). Recorded as the north-star vision, not a build target.

---

## Quick reference — commands & files

| Task | Command / file |
|---|---|
| Evaluate candidate @1024 | `eval_holdout.py --old .../best.pt --old-imgsz 1024 --new <run>/best.pt --new-imgsz 1024` |
| Pick operating point | `conf_sweep.py --weights <run>/best.pt --imgsz 1024` |
| Re-apply scene holdout after `build_dataset.py` | snippet in `docs/STATUS.md` (Operational recipes) |
| Promote | `cp <run>/weights/best.pt data/training/weights/best.pt` (back up first) |
| Detector deploy defaults | `detect_boats.py`: imgsz 1024, conf 0.15, weights → in-repo `best.pt` (confirmed 2026-07-25; imgsz 1280 is a non-win — use `--augment` for the opt-in big-ship profile) |
| Add an imagery source | subclass `Provider` (`scripts/providers/base.py`), `register()` it (`providers/__init__.py`); copy `providers/pc.py` (optical) or `providers/s1.py` (SAR) |
| Select a source | `sat_fetch.py --provider {pc,s1}` (default `pc`; `SAT_PROVIDER` env). `pc`=S2 optical, `s1`=S1 SAR — both free, no credentials |
| Fetch a SAR scene | `sat_fetch.py --provider s1 --bbox … --dry-run` then without `--dry-run` (free, all-weather; needs its own `best_sar.pt`) |
| Refactor prerequisite ✅ | `main()`-guard + provider dispatch both done (AUDIT.md #1) |
| Ingest live AIS (P3) ✅ | `aisstream_fetch.py --bbox … --duration 120` (or `--wdpa-id`) → `data/raw/ais/ais_positions_*.json`; needs `AISSTREAM_API_KEY`, `--dry-run` to test wiring |
| Fuse AIS → dark vessels (P3) ✅ | `fuse_violations.py --run <run> --ais data/raw/ais/<file>.json` → `violation_events.json` (+ dark `.geojson`) |
| GFW identity/fishing (P3) ✅ | `gfw_vessels.py --from-events <violation_events.json> --start … --end …` → `data/raw/gfw_vessels/*.json`; needs `GFW_API_TOKEN`, `--dry-run` to test wiring |
| Enrich matched events (P3) ✅ | re-run `fuse_violations.py … --vessels data/raw/gfw_vessels/<file>.json` (fills identity + fishing, lifts score for confirmed-fishing / IUU) |
| Historical AIS (P3) ✅ | `marinecadastre_fetch.py --csv AIS_YYYY_MM_DD.csv --wdpa-id <id> --start … --end …` → `data/raw/ais/…` (fuse past US-waters scenes) |
| Rank / report (P3) ✅ | `report_violations.py --run <run>` → ranked `violations_report_*.csv` + self-contained `.html` |
| One-shot driver (P3) ✅ | `scan_violations.py --run <run> --ais <file>.json [--gfw --wdpa-id <id>]` (fuse → enrich → report) |
| Test the P3 stages offline | `python -m unittest discover -s scripts/tests -p 'test_*.py'` (104 tests; synthetic, no network / key / ML deps) |
| Violation schema | `docs/PIPELINE.md` → `violation_events.json` (pinned) |

## Key sources
Small-object: SAHI 2202.06934, copy-paste 1902.07296, NWD 2110.13389, P2/SOD-YOLOv8 2408.04786.
Imagery: Kanjir survey (PMC5877374), S2-YOLO RSE 2025. Dark vessels: ESA/GFW (75% untracked),
Skylight (skylight.global/platform), xView3-SAR 2206.00897.
