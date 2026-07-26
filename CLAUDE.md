# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AntiAngler is a geospatial data processing and MPA violation detection project. It fuses AIS tracking, Global Fishing Watch (GFW) fishing-effort data, and satellite imagery (Sentinel-1 SAR and Sentinel-2 optical) to identify vessels entering Marine Protected Areas (MPAs) and assess their behavior.

The initial pipeline filters the World Database of Protected Areas (WDPA) for MPAs with IUCN category Ia (Strict Nature Reserve) or Ib (Wilderness Area). The full system then ingests AIS/VMS feeds and GFW APIs, preprocesses tracks, intersects vessel positions with MPA geofences, infers fishing activity, cross-references vessel identity with registries, and scores each violation event.

The end-to-end satellite ship-detection loop — collect imagery over MPAs (`sat_fetch.py`) → scan chips for boats (`detect_boats.py`) → report detections + human-in-the-loop review (`review_server.py`) → export corrected labels to retrain the model (`export_labels.py`) — is built and specified in `docs/PIPELINE.md` (the build spec with all inter-stage data contracts). `docs/AUDIT.md` tracks known antipatterns/inefficiencies in priority order. `docs/STATUS.md` is the live status/handoff — which model is promoted, per-run eval numbers, open problems, and next steps (read it first when resuming detector/training work). `docs/ROADMAP.md` is the forward plan — the phased path beyond the current model (P2 head, Sentinel-1 SAR, and the AIS dark-vessel fusion that is the actual product), with file-level pointers and the `main()`-guard refactor those phases depend on. `docs/PHASE3_PLAN.md` is the actionable "what's left to finish Phase 3" checklist (live validation, batch driver, ranked output, optional Skylight/historical-AIS) with a definition of done — read it when resuming the AIS-fusion product work.

## Running Scripts

All scripts resolve paths relative to `Path(__file__)`, so they can be run from any working directory:

```bash
# Step 1: Filter raw WDPA shapefiles → processed GeoPackage
python scripts/prep_polygons.py

# Step 2: Visualize filtered MPAs on a world map (interactive by default; pass --save PATH.png for headless)
python scripts/view_polygons.py

# Step 3: Fetch GFW fishing-effort data and persist to data/raw/gfw/
python scripts/gfw_fetch.py

# Step 4: Fetch a Sentinel-2 scene for an MPA (or explicit bbox) and tile it into
# georeferenced RGB chips under data/raw/sentinel2/ (Planetary Computer — no credentials)
python scripts/sat_fetch.py --wdpa-id <WDPAID>
python scripts/sat_fetch.py --bbox <minLon> <minLat> <maxLon> <maxLat>
# Preview what imagery is available over an AOI (ranked by cloud OVER the AOI) without fetching:
python scripts/sat_fetch.py --bbox <minLon> <minLat> <maxLon> <maxLat> --dry-run

# Step 5: Run the YOLOv8 boat detector over a run dir; write detections.json (+ .geojson)
python scripts/detect_boats.py --run data/raw/sentinel2/<run>/

# Step 6: Review detections in a local browser UI (prints a tokenized 127.0.0.1 URL to open)
python scripts/review_server.py --run data/raw/sentinel2/<run>/

# Step 7: Export reviewed chips as a YOLO training set for fine-tuning the detector
python scripts/export_labels.py --run data/raw/sentinel2/<run>/

# Step 8: Assemble the training dataset — layer the reviewed exports onto the base
# boat dataset in data/training/ (idempotent; --dry-run to preview)
python scripts/build_dataset.py

# Step 9: Train the detector in-repo. Fine-tune from the current best.pt, or --weights scratch
# for a new model from yolov8n. Writes to data/training/runs/<name>/weights/best.pt
python scripts/train.py --weights best      # fine-tune (active learning)
python scripts/train.py --weights scratch   # train a fresh model

# Step 10a (P3 ingest): stream live AIS from AISStream.io over a bbox/MPA and normalize it into
# the AIS contract file the matcher reads (needs AISSTREAM_API_KEY; --dry-run needs neither key nor net)
python scripts/aisstream_fetch.py --bbox <minLon> <minLat> <maxLon> <maxLat> --duration 120
python scripts/aisstream_fetch.py --wdpa-id <WDPAID> --duration 120

# Step 10b (P3 fusion): match a run's detections against normalized AIS positions with a
# velocity-aware gate; write violation_events.json (+ .geojson of dark points). Pure stdlib.
python scripts/fuse_violations.py --run data/raw/sentinel2/<run>/ --ais data/raw/ais/<ais_file>.json

# Step 10c (P3 identity/fishing): resolve the matched vessels' MMSIs to GFW identity + fishing
# activity, then re-fuse with --vessels to fill vessel_name/flag/gear/fishing_* and lift the score
# for confirmed-fishing vessels (needs GFW_API_TOKEN; --dry-run needs neither token nor net)
python scripts/gfw_vessels.py --from-events data/raw/sentinel2/<run>/violation_events.json --start <YYYY-MM-DD> --end <YYYY-MM-DD>
python scripts/fuse_violations.py --run data/raw/sentinel2/<run>/ --ais data/raw/ais/<ais_file>.json --vessels data/raw/gfw_vessels/<file>.json

# Step 10d (P3 output): rank the violation_events into a CSV + self-contained HTML report
python scripts/report_violations.py --run data/raw/sentinel2/<run>/

# Batch driver (P3): sequence fuse → (optional --gfw enrich) → report for a run in one command
python scripts/scan_violations.py --run data/raw/sentinel2/<run>/ --ais data/raw/ais/<ais_file>.json [--gfw --wdpa-id <WDPAID>]

# Historical AIS (P3): normalize a NOAA MarineCadastre archive CSV into the AIS contract, AOI-filtered
python scripts/marinecadastre_fetch.py --csv AIS_2024_01_01.csv --wdpa-id <WDPAID> --start <ISO> --end <ISO>

# Run the offline P3 tests (synthetic fixtures; no network / no key / no ML deps).
# Or all at once: python -m unittest discover -s scripts/tests -p 'test_*.py'
python scripts/tests/test_fuse_violations.py
python scripts/tests/test_aisstream_fetch.py
python scripts/tests/test_gfw_vessels.py
python scripts/tests/test_report_violations.py
python scripts/tests/test_pipeline_e2e.py
```

No build system or test suite yet. Runtime dependencies are pinned in `requirements.txt` at the repo root — install with `pip install -r requirements.txt`. The pinned set covers `gfw_fetch.py` (`requests`, `python-dotenv`) and `sat_fetch.py` (`numpy`, `rasterio`, `geopandas`, `pystac-client`, `planetary-computer` — a heavier stack; `rasterio`/`geopandas` bundle GDAL/GEOS/PROJ, installed from PyPI wheels on Windows). The stage 2–3 loop (`detect_boats.py`, `review_server.py`, `export_labels.py`) needs the heavier ML/web stack pinned separately in `requirements-ml.txt` (`ultralytics`→torch, `fastapi`, `uvicorn`, `pillow`) — install it *in addition* (`pip install -r requirements-ml.txt`); it is kept out of `requirements.txt` so the daily gfw cron does not pull torch. The AIS ingester (`aisstream_fetch.py`) needs only `websocket-client`, pinned separately in `requirements-ais.txt` (`pip install -r requirements-ais.txt`) and lazy-imported, so neither the cron nor the offline P3 tests require it. `prep_polygons.py` and `view_polygons.py` additionally need `pandas`, `geodatasets`, and `matplotlib`, now **pinned in `requirements.txt`** (resolver-verified against the existing pins), so a fresh clone can run steps 1–2 from `pip install -r requirements.txt` alone. A `venv/` is present locally but uncommitted.

GFW API requires a `GFW_API_TOKEN` in a `.env` file at the project root or inside `scripts/` (`gfw_fetch.py` loads both locations). `sat_fetch.py` needs no credentials (Planetary Computer is open); an optional `PC_SDK_SUBSCRIPTION_KEY` only raises rate limits. Copy the committed **`.env.example`** to `.env` for the full credential/knob template (`GFW_API_TOKEN`, `AISSTREAM_API_KEY`, and the per-script `*_` env fallbacks); `.env` is gitignored.

**Phase 3 (AIS fusion) first-run setup — one reference.** The fusion/analysis stages are pure stdlib and need **no credentials or extra deps**: `fuse_violations.py`, `report_violations.py`, `marinecadastre_fetch.py` (local-CSV path), `scan_violations.py` (driver), and all P3 tests run as-is. Only the **live ingest** stages need setup, and those are do-from-home (see `docs/PHASE3_PLAN.md` Workstream 1):

| Stage | Needs | Install / key |
|---|---|---|
| `aisstream_fetch.py` (live AIS) | `AISSTREAM_API_KEY` in `.env` + `websocket-client` | `pip install -r requirements-ais.txt`; free key at aisstream.io |
| `gfw_vessels.py` (identity/fishing) | `GFW_API_TOKEN` in `.env` + `requests` | already in `requirements.txt` (same token as `gfw_fetch.py`) |
| `marinecadastre_fetch.py --date` (download) | `requests` | `requirements.txt`; the `--csv <local file>` path needs nothing |

Every P3 ingester has a `--dry-run` (or a local-file path) that works with neither a key nor the network. `.env` at the repo root or `scripts/` is loaded by all of them (dotenv is soft-imported, so its absence is not fatal).

**TLS note (corporate/AV proxy environments):** if an antivirus or proxy intercepts TLS (e.g. Avast), `sat_fetch.py`'s Python HTTPS calls need the interceptor's CA — set `REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE` to a bundle containing it (certifi + the AV root). GDAL's `/vsicurl` COG reads use the Windows schannel backend, which may still fail with a partial-chain error under interception; that is an environment/cert-store matter, not a code issue.

## CI / Automation

`.github/workflows/gfw.yml` runs `scripts/gfw_fetch.py` on:
- `workflow_dispatch` (manual trigger)
- `schedule: cron '0 0 * * *'` (daily at 00:00 UTC)

The workflow uses Python 3.11 with `actions/setup-python@v5` pip caching (keyed on `requirements.txt`), installs deps via `pip install -r requirements.txt`, reads `GFW_API_TOKEN` from GitHub secrets, and uploads `data/raw/gfw/` as the `gfw-fishing-effort-report` artifact with `if-no-files-found: error` (CI fails loudly if no file was written).

`.github/workflows/tests.yml` runs the **offline test suite** on every `push` + `pull_request` (and `workflow_dispatch`): `python -m compileall -q scripts` (a syntax gate over *every* script, including the heavy ML/geo ones — `compileall` parses without importing) then `python -m unittest discover -s scripts/tests` (the 104 offline tests). It needs **no credentials and no `pip install`** — the suite is pure stdlib (the P3 ingesters soft-import `python-dotenv` and lazy-import `requests`/`websocket-client`), so keep new tests dependency-free. This catches syntax errors in any script and P3 regressions that `gfw.yml` (which only runs `gfw_fetch.py`) would miss.

## Data Layout

- `data/raw/WDPA/` — Raw WDPA September 2025 shapefiles, split into 3 parts (`WDPA_Sep2025_Public_shp_0`, `_1`, `_2`) due to size. These are gitignored.
- `data/raw/gfw/` — Per-run GFW 4WINGS API responses written by `gfw_fetch.py` as `gfw_report_<UTC-timestamp>.json`. Gitignored; surfaced via the CI artifact.
- `data/raw/ais/` — Normalized AIS position snapshots as `ais_positions_<UTC-timestamp>_<label>.json`: `{generated_utc, source, bbox_wgs84, ..., positions:[{mmsi, lon, lat, timestamp, sog, cog, name?}]}`. The `positions` list is the AIS input contract `fuse_violations.py --ais` reads. Written by `aisstream_fetch.py` (`source: "aisstream"`, live) and `marinecadastre_fetch.py` (`source: "marinecadastre"`, historical US-waters archive) — same dir + contract. Gitignored.
- `data/raw/gfw_vessels/` — Normalized GFW per-vessel records written by `gfw_vessels.py` as `gfw_vessels_<UTC-timestamp>.json`: `{generated_utc, source, window, insights, vessels:[{mmsi, vessel_id, shipname, flag, geartype, imo, callsign, fishing_hours, fishing_probability, iuu_listed?, ais_off_events?, fishing_in_no_take_mpa?, source}]}`. The `vessels` list is the identity/fishing contract `fuse_violations.py --vessels` reads to enrich matched events. Gitignored. Distinct from `data/raw/gfw/` (the coarse gridded 4WINGS effort density from `gfw_fetch.py` — a different GFW stream).
- `data/raw/sentinel2/<UTC-timestamp>_<label>/` — Per-run Sentinel-2 imagery from `sat_fetch.py`: a `chips/` dir of georeferenced RGB GeoTIFF tiles plus a `manifest.json` (chosen scene id/datetime/cloud + its AOI coverage & AOI cloud, the `selection` block with the thresholds used and whether a best-effort `fallback_used` scene was accepted, probed `candidates`, AOI bbox, scene CRS, stretch params, and per-chip pixel window + WGS84/UTM bounds). Gitignored. The `manifest.json` is the seam `detect_boats.py` reads (schema in `docs/PIPELINE.md`). Stage 2–3 then write more files *into the same run dir*: `detections.json` + `detections.geojson` (stage 2 output — geolocated boxes, the QGIS-loadable GeoJSON is points, each detection carries `inside_mpa` + `on_water`), `reviews.json` (stage 3 human verdicts, keyed by chip), `water_mask.tif` (optional ESA WorldCover 10 m land/water cache from `water_mask.py`, consumed by `detect_boats.py --water-only`), and `violation_events.json` + `violation_events.geojson` (stage 4 `fuse_violations.py` output — per in-MPA detection, dark-vs-AIS-matched with a velocity-aware gate; the GeoJSON is the dark points).
- `data/processed/training_exports/<tag>/` — YOLO training sets from `export_labels.py`: `images/` (chip PNGs) + `labels/` (`cls cx cy w h`, empty file = hard negative) + `export_log.json` provenance. `<tag>` defaults to the UTC date and accumulates runs. Gitignored. Consumed by `build_dataset.py`. External labeled datasets land here too as their own tag: `scripts/ingest_s2ships_finland.py` writes `finland-s2/` (the Zenodo 15019034 Gulf-of-Finland S2 vessel set, resolved on Planetary Computer + tiled to match `sat_fetch` preprocessing; source geopackages in `data/raw/external_ships/finland/`). **This dir is OPTICAL-only** — `build_dataset.py` layers *every* tag here onto the optical `data/training/`, so never drop SAR exports in it.
- `data/processed/sar_training_exports/<tag>/` — **SAR** training sets (same `export_labels.py` format: chip PNGs from the S1 VV/VH/(VV/VH) pseudo-RGB + YOLO labels). Deliberately a SEPARATE dir from `training_exports/` so `build_dataset.py` (optical) never sweeps SAR chips into `data/training/`. First set: `deep/` = the reviewed deep-water `*_s1deep-*` runs (Fujairah/Gibraltar/Hormuz), 149 chips / 206 vessel boxes / 112 empty hard-negatives. Feeds the future `best_sar.pt` (which still needs its own SAR base dir + build/train flow — not yet built). Gitignored.
- `data/training/` — the in-repo YOLO training set the detector is trained on. `images/{train,val,test}` + `labels/{train,val,test}` hold the base ~621 labelled boat photos (named `boat<N>`, copied from the BoatDetection dataset) with the reviewed active-learning chips (named `<run>__<chip>`) layered on by `build_dataset.py`. Also holds `config.yml` (the YOLO data yaml, its `path` auto-pinned), `weights/` (`yolov8n.pt` base + `best.pt` current model), `runs/<name>/` (training outputs), and `active_learning_manifest.json` (split provenance). NOT gitignored by default (it is the training base); commit or ignore per preference.
- `data/processed/wdpa_marine_ia_ib.gpkg` — Output: filtered marine MPAs in WGS84 (EPSG:4326).
- `data/raw/WDPA/Resources_in_English/` — Official WDPA attribute/metadata PDFs for reference.

## Script Architecture (Compartmentalized by Data Stream)

Each data stream has its own script. Do not mix data-stream concerns across scripts.

| Script | Data Stream | Description |
|--------|-------------|-------------|
| `prep_polygons.py` | WDPA / MPA boundaries | Filters raw WDPA shapefiles → GeoPackage |
| `view_polygons.py` | Visualization | Overlays processed MPAs on a world map |
| `gfw_fetch.py` | Global Fishing Watch API | Queries GFW 4WINGS API for fishing effort by vessel |
| `sat_fetch.py` | Sentinel-2 optical imagery | Fetches an AOI-clear recent S2 L2A true-color scene from Planetary Computer STAC for a bbox/MPA and tiles it into georeferenced RGB chips |
| `detect_boats.py` | Ship detection (stage 2) | Runs the YOLOv8 boat detector over a run dir's chips, geolocates + dedups boxes, writes `detections.json`/`.geojson` |
| `sar_seed.py` | SAR candidate seeding (stage 2, S1) | Classical Sentinel-1 vessel detector (dual-pol saturation + coherent-blob) that seeds `detections.json` for the SAR review loop when no `best_sar.pt` exists yet |
| `sar_companion.py` | SAR review land reference | Reprojects a paired S2 scene into each SAR chip's grid → `companion/<chip>.png`; `review_server.py` shows it beside the SAR chip so land is unambiguous under speckle |
| `bathymetry.py` | GEBCO depth (deep-water gate) | Free no-auth GEBCO depth via the opentopodata API; `sar_seed.py --min-depth` drops shallow-water candidates (shallow water = strong bathymetric SAR clutter) |
| `ingest_s2ships_finland.py` | External S2 dataset ingest | Pulls the Zenodo Gulf-of-Finland S2 vessel set through the pipeline into a preprocessing-matched YOLO export (`training_exports/finland-s2/`) |
| `water_mask.py` | Land/water mask (precision) | Fetches ESA WorldCover 10 m over a run's AOI → cached `water_mask.tif`; `detect_boats.py --water-only` then drops land (coast/rock/islet) detections. Shared by the optical + SAR detectors. |
| `ndwi_mask.py` | Open-water mask (precision) | Builds a per-run NDWI raster (`ndwi.tif`) from the scene's B03/B08; `detect_boats.py --open-water` drops detections on pure water. Improvement plan O2 — targets ON-water false alarms (distinct from `water_mask.py`, which drops land). A calibrated non-win on small-vessel scenes (see `docs/IMPROVEMENT_PLAN.md`); off by default. |
| `review_server.py` | Human review (stage 3) | Local hardened FastAPI UI to mark detections correct/incorrect and draw missed boats → `reviews.json` |
| `export_labels.py` | Active learning (stage 3) | Turns reviewed chips into a YOLO training set to fine-tune the detector |
| `build_dataset.py` | Model training (stage 4) | Layers reviewed exports onto the base dataset in `data/training/` (idempotent) |
| `train.py` | Model training (stage 4) | Trains/fine-tunes the YOLOv8 boat detector in-repo; writes `data/training/runs/<name>/` |
| `eval_holdout.py` | Model evaluation (stage 4) | Evaluates weights on the scene-aware satellite holdout with tiny-object metrics (center-distance P/R, FP/chip, recall stratified by px size, bootstrap CI) — not IoU-mAP |
| `conf_sweep.py` | Model evaluation (stage 4) | Sweeps the confidence threshold on the holdout to pick the operating point (PR-style curve); reuses `eval_holdout`'s matching logic |
| `augment_positives.py` | Training data augmentation | Water-gated copy-paste synthesis of positive chips → a new training-export tag, to counter the negative-heavy set. **Superseded** as the live recall lever by real diverse data (`docs/STATUS.md`); kept for the record |
| `aisstream_fetch.py` | AIS positions (ingest, P3) | Streams live AIS from AISStream.io over a bbox/MPA and normalizes it into the AIS contract file `fuse_violations.py` reads |
| `marinecadastre_fetch.py` | Historical AIS (ingest, P3) | Normalizes a NOAA MarineCadastre archive AIS CSV into the same AIS contract, AOI/time-filtered (for fusing *past* scenes over US waters) |
| `gfw_vessels.py` | GFW per-vessel identity/activity (ingest, P3) | Resolves MMSIs to GFW identity (name/flag/gear/imo) + fishing hours (+ optional IUU/AIS-off insights); writes the vessel-records file `fuse_violations.py --vessels` enriches with |
| `fuse_violations.py` | AIS × detection fusion (stage 4, P3) | Matches a run's detections against normalized AIS positions with a velocity-aware gate; emits dark-vessel `violation_events.json` (optionally GFW-enriched via `--vessels`) |
| `report_violations.py` | Operator output (stage 4, P3) | Aggregates `violation_events.json` file(s) into a ranked CSV + self-contained HTML report by `violation_score` (read-only, pure stdlib) |
| `scan_violations.py` | Batch driver (stage 4, P3) | Thin orchestrator sequencing detect → fuse → (GFW-enrich) → report for a run/MPA via each stage's `main(argv)` |

`prep_polygons.py` reads each of the 3 split shapefiles, concatenates them into a single GeoDataFrame, filters rows where `MARINE` is `1` or `2` and `IUCN_CAT` is `Ia` or `Ib`, reprojects to EPSG:4326, and writes to a GeoPackage.

`view_polygons.py` loads the processed GeoPackage and overlays it on a Natural Earth world background via `geodatasets`. Defaults to interactive `plt.show()`; pass `--save PATH.png` to switch matplotlib to the `Agg` backend and write a PNG instead (headless-safe for CI).

`gfw_fetch.py` posts to the GFW `/v3/4wings/report` endpoint, requesting fishing effort grouped by vessel ID within a region. All eight query knobs (`--region-id`, `--region-dataset`, `--start`, `--end`, `--gear`, `--group-by`, `--spatial-resolution`, `--temporal-resolution`, `--format`, `--dataset`) are exposed as CLI flags with matching `GFW_*` env-var fallbacks — CLI overrides env. Defaults: EEZ `8466` / `public-eez-areas`, gear `trawlers`, `VESSEL_ID` / `LOW` / `ENTIRE` / `JSON`, dataset `public-global-fishing-effort:latest`. If `--start`/`--end` are omitted, the window is yesterday UTC (single day), so the daily cron pulls a rolling window. On 2xx, the JSON response is written atomically to `data/raw/gfw/gfw_report_<UTC-timestamp>.json` (write to `.tmp`, then `Path.replace()`); non-2xx responses raise before any file is written, so the raw dir stays clean.

`sat_fetch.py` takes an AOI as either an explicit `--bbox minLon minLat maxLon maxLat` or an MPA lookup (`--wdpa-id` / `--wdpa-pid`, resolved against `wdpa_marine_ia_ib.gpkg`). It searches Planetary Computer's `sentinel-2-l2a` STAC in the window (default: ~30 days ending yesterday, scene-wide `eo:cloud_cover < --max-cloud`), then **selects AOI-aware**: it probes up to `--max-search` candidates (least scene-cloud first) with the SCL cloud mask read *over the AOI itself*, and picks the first that meets `--min-coverage` (AOI overlap) and `--max-aoi-cloud` (cloud/shadow over the AOI) — because scene-wide cloud % is misleading for a small AOI. If none pass, it uses the best effort and warns; `--dry-run` prints the ranked candidate table and writes nothing (use it to find a good `--start`/`--end`/`--max-cloud` for a new location). The chosen scene's assets are signed and B04/B03/B02 windowed-read in the scene's **native UTM** via `rasterio`; surface reflectance is stretched to 8-bit RGB once over the whole AOI (`--stretch linear|percentile`; match this to the detection model's training preprocessing), then tiled into `--tile-size` (default 640) chips with `--overlap` (default 64), skipping blank/nodata chips. Every knob has a `SAT_*` env-var fallback (CLI > env > default). Output is written atomically (temp run dir → `Path.replace()`). Each chip GeoTIFF is self-describing (carries its own CRS + affine), so downstream code needs only `rasterio.open(chip)` to map pixels → lat/lon. The imagery source is a swappable **Provider** (`--provider {pc,s1}`, default `pc`; `SAT_PROVIDER` env): `main()` is a thin orchestrator that routes the source-specific steps (`open_catalog`/`search_candidates`/`assess_aoi`/`read_aoi`/`scale_to_uint8`) through the selected provider under `scripts/providers/` (`pc.py` = the live Sentinel-2 optical impl; `s1.py` = Sentinel-1 GRD SAR, also free on Planetary Computer — both free, no credentials), then does the provider-agnostic tiling + chip/manifest IO. The provider fills any unset `--collection/--bands/--resolution/--stretch-max` from its own defaults. The **`s1` SAR provider** (`--provider s1`, default `--collection sentinel-1-grd`, `--bands vv,vh`) drops the cloud query (no `eo:cloud_cover` on SAR; ranks scenes newest-first, `aoi_cloud=None`) and converts backscatter → 8-bit pseudo-RGB `[VV, VH, VV/VH]` via a dB percentile stretch — it needs its **own** SAR-trained detector (`best_sar.pt`), not the optical `best.pt`. Known limitation: Sentinel-2's 10 m GSD is marginal for vessels smaller than ~15 m; SAR (`s1`) sees through cloud and finds dark/AIS-off vessels but is also ~10-20 m. There is no free global higher-resolution optical source, so 10 m is the optical GSD floor here (see `docs/ROADMAP.md`). Downstream, the model-runner `detect_boats.py` (stage 2, in-repo — originally the sibling `C:\gtest\BoatDetection`; full spec in `docs/PIPELINE.md`) consumes `manifest.json`, runs inference per chip, and maps detections back to lat/lon via each chip's transform.

`detect_boats.py` (stage 2) reads a `sat_fetch.py` run dir and runs the YOLOv8 boat detector (`--weights`, default the **in-repo promoted model** `data/training/weights/best.pt` — originally the sibling `C:\gtest\BoatDetection` `best.pt`, now canonical in-repo) on each chip. Each chip GeoTIFF is self-describing, so a box's pixels map chip→UTM (chip affine)→WGS84 directly. Detections are pooled in the scene's UTM and **deduplicated by merging centroids within `--merge-dist` metres** (IoU is unstable for 1–5 px vessels, so dedup is metric, not pixel/degree). For WDPA runs it tags each detection `inside_mpa` (shapely point-in-polygon; `--inside-mpa-only` drops the rest). Writes `detections.json` (+ point `detections.geojson`) atomically. Keep `--conf` low (the deploy default is **0.20** for the `finetune-bigship` model promoted 2026-07-26 — the recall-leaning point that Pareto-dominates the old 0.15 deploy on every holdout metric; F1 peak 0.25) for recall; the model was trained on high-resolution overhead boat photos (large, sharp, mostly one vessel per frame), whereas Sentinel-2's 10 m GSD renders vessels as 1–5 px specks in cluttered scenes, so treat first-pass hits as *candidates to review*, not answers. `--water-only` tags each detection `on_water` from the run's cached `water_mask.tif` (build it first with `water_mask.py`) and drops land detections — the confident rock/islet false-positive class (improvement plan O1). Land is only ~16% of false positives, though; the dominant on-water false alarms on small-vessel scenes need same-region hard negatives, not this filter (see `docs/IMPROVEMENT_PLAN.md` Step 0). `--augment` enables test-time augmentation (multi-scale + flips) — a **big-ship-AOI-only** recall profile: with `--imgsz 1280` it lifts large-ship recall (Port Said 0.69→0.81) at unchanged precision, but it amplifies water-texture false alarms on small-vessel/cluttered scenes, so it is opt-in and never the default (~4× slower; see `docs/STATUS.md`).

`sar_seed.py` (stage 2 for the SAR stream) is the classical Sentinel-1 vessel detector used to seed the review loop before a trained `best_sar.pt` exists — the SAR analog of `detect_boats.py`. It writes the same `detections.json` schema, so `review_server.py` reviews SAR chips identically. It does **not** use textbook CFAR: `s1.py`'s dB-percentile stretch (tuned for display) leaves water mid-bright (median ~136/255) and saturates vessels at 255, so CFAR/top-hat (which assume dark water) fail. Instead it thresholds the despeckled **dual-pol geometric mean** `sqrt(VV·VH)` (a vessel is bright in both pols, speckle is not) at a **per-chip adaptive `median + k·MAD`** level — robust and stretch-invariant, since s1.py normalizes each scene differently (a fixed 0–255 threshold floods some scenes and misses others), and it stays quiet on empty water. It then keeps coherent blobs by an area window (min-area separates ships from bright speckle; max-area/side + a nodata-boundary erosion drop land and the swath edge). Two spatial gates: **`--water-only`** (WorldCover `water_mask.tif` — buildings/ports are very bright in SAR) and **`--min-depth <m>`** (GEBCO via `bathymetry.py` — shallow water returns strong bathymetric clutter, so SAR targets DEEP water: deep anchorages like Fujairah + lane chokepoints like Gibraltar/Hormuz, not shallow coastal anchorages). Tuned on Fujairah; leans toward recall — review is the quality gate. The reviewed SAR chips train a **separate** `best_sar.pt` (a SAR-specific model, never the optical `best.pt`).

`review_server.py` (stage 3) is a minimal, hardened, single-user **localhost** FastAPI UI (`--run <run dir>`, `--port`). If the run dir contains a `companion/` dir (from `sar_companion.py`), it renders a co-registered **optical image beside each (SAR) chip** as a static land reference — SAR speckle makes land/water ambiguous, and land is static so the S1/S2 date mismatch is irrelevant; the same detection boxes overlay both panes. It overlays `detections.json` on each chip and lets a human mark boxes correct/incorrect and drag missed boats, persisting verdicts to `reviews.json` (atomic). Security posture (single-user local tool): binds `127.0.0.1` only; a Host-header allowlist middleware blocks DNS-rebinding; a random startup token (printed in the URL) gates every route; chips are served by opaque manifest index, never a client path (defense-in-depth: `resolve()` + `is_relative_to`); strict per-request-nonce CSP; the page is fully self-contained (inline canvas UI, zero external assets); POST bodies are pydantic-validated (`extra="forbid"`, `Literal` verdicts). "Mark reviewed" is gated on every detection being judged — the *complete-labels* rule that keeps the export trainable.

`export_labels.py` (stage 3) turns reviewed chips into a YOLO training set under `data/processed/training_exports/<tag>/` (`images/*.png` + `labels/*.txt`, `cls cx cy w h` normalized; class 0 = boat). It keeps boxes verdicted `correct` or `missed`, drops `incorrect`, and writes an *empty* label for a reviewed chip with no boats — a hard negative that suppresses false positives. Only fully-reviewed chips are exported (complete-labels rule). Then `build_dataset.py` layers the accumulated exports onto the base dataset in `data/training/` and `train.py` fine-tunes from `best.pt` (or trains fresh); re-run stage 2 with the new weights — the active-learning loop, now fully in-repo. Full loop + data contracts in `docs/PIPELINE.md`.

`aisstream_fetch.py` (AIS ingest, ROADMAP Phase 3 step 2) is the **AISStream.io adapter** for the AIS position stream — the live source that feeds the matcher. It connects to the free AISStream websocket (`wss://stream.aisstream.io/v0/stream`, needs a registered `AISSTREAM_API_KEY` in `.env`), subscribes to a bbox (`--bbox` or an `--wdpa-id`/`--wdpa-pid` MPA lookup, mirroring `sat_fetch.py`), collects `PositionReport` messages for a bounded `--duration`/`--max-messages` (Ctrl-C stops early and still writes), and **normalizes each into the neutral AIS contract**, writing `data/raw/ais/ais_positions_<UTC>_<label>.json` — the file `fuse_violations.py --ais` consumes. The risky wire-format handling is factored into pure, offline-tested functions and covers AISStream's three footguns: bounding boxes are `[lat, lon]` corner order (reverse of GeoJSON — `bbox_to_aisstream` flips it), `MetaData.time_utc` is a Go-format nanosecond string (`2023-05-10 11:46:52.509865357 +0000 UTC`, truncated to microseconds), and `Sog`/`Cog` not-available sentinels (`102.3`/`360.0`) map to `null` (correctly triggering the matcher's max-speed fallback). **`--dry-run`** resolves the bbox + prints the subscription without a key or network. **Limitation:** AISStream is a *live* feed (positions from now forward only), so it fits a near-real-time workflow (fetch imagery + capture AIS in the same window, then fuse); a historical scene needs an archive/Spire/GFW source — a future sibling ingester. Its **only extra dep** is `websocket-client` (pinned in `requirements-ais.txt`, lazy-imported, kept out of the gfw cron). **House rule:** this is the AISStream adapter only — it never imports `fuse_violations.py`; other AIS sources (Spire, an archive) or identity sources (GFW Vessels/Insights) are each their own sibling ingester normalizing into the same contract.

`gfw_vessels.py` (GFW per-vessel ingest, ROADMAP Phase 3 step 3) is the GFW **per-vessel** identity/activity ingester — its own data stream, distinct from `gfw_fetch.py` (which pulls the coarse *gridded* 4WINGS effort density, not per-vessel data). Given a set of MMSIs (`--mmsi`, or pulled from a `violation_events.json` via `--from-events`, or an AIS file via `--from-ais`), it resolves each to GFW **identity** (Vessels search → `vessel_id`/`shipname`/`flag`/`geartype`/`imo`; `ssvid`==MMSI for AIS), summarizes recent **fishing** activity (Events API → `fishing_hours` + a coarse `fishing_probability`: 1.0 if the vessel had any fishing event in the `--start`/`--end` window, so bracket the scene date to keep it relevant; pass **`--region-bbox`/`--wdpa-id`** to filter fishing events to the MPA so `fishing_probability` means "fished *in* the MPA" — client-side position filtering, with a graceful fallback to the window prior if events carry no position), and — with `--insights` — pulls GFW **Vessel Insights** (IUU-list flag, AIS-off/gap events, apparent fishing in no-take MPAs). It writes the normalized vessel-records file `fuse_violations.py --vessels` reads. The risky GFW response handling is factored into pure, offline-tested normalizers; the HTTP layer is a thin `requests`/Bearer-token shell mirroring `gfw_fetch.py` (`GFW_API_TOKEN` in `.env`). **Live-verified 2026-07-26** (three v3 corrections landed on the first real run): `/v3/events` requires `offset` whenever `limit` is sent; an MMSI can resolve to **several identity clusters**, so `normalize_identity` picks the freshest/busiest matching block (`transmissionDateTo`, then counters) and reads `geartype` from the entry-level `combinedSourcesInfo[].geartypes[].name`; `/v3/insights/vessels` takes include `VESSEL-IDENTITY-IUU-VESSEL-LIST` (not `VESSEL-IDENTITY-IUU`) and answers **201** with one flat object (`gap`/`apparentFishing` `periodSelectedCounters`, `vesselIdentity.iuuVesselList`). Real captured responses are committed as offline fixtures (`scripts/tests/fixtures/gfw_search_response.json`, `gfw_insights_response.json`), so the parsers are regression-locked to the live shapes; `--dry-run` still prints the planned requests with neither a token nor the network. **House rule:** its own ingester; it never imports `fuse_violations.py`.

`fuse_violations.py` (stage 4, ROADMAP Phase 3 — the AIS dark-vessel reframe, the product payoff) reads a `detect_boats.py`/`sar_seed.py` run dir (`detections.json` + the sibling `manifest.json`) plus a **normalized AIS positions file** (`--ais`, `.json` or `.csv`) and decides, per **in-MPA** detection, **dark vs matched**. It optionally takes a **GFW vessel-records file** (`--vessels`, from `gfw_vessels.py`) that enriches *matched* events by MMSI: it fills `vessel_name`/`flag`/`gear_type`/`imo` and `fishing_hours`/`fishing_probability`, adds `"GFW"` to `evidence`, and lifts `violation_score` — the matched weight interpolates from `--matched-weight` (fishing_probability 0) up to `--fishing-weight` (fishing_probability 1, default 1.0), and an IUU-listed vessel escalates to full weight. **With no `--vessels` the verdicts and scores are unchanged from the AIS-only path** (the GFW/fishing fields stay null; the event record just gains those null keys). It is **pure stdlib** (no torch/rasterio/shapely) — `inside_mpa`, `centroid_wgs84`, and `scene.datetime` are already in `detections.json`, so the matcher runs anywhere and is testable offline (`scripts/tests/test_fuse_violations.py`). A satellite acquisition is instantaneous and almost never coincides with an AIS ping, so it does **not** use a fixed match radius (which over-flags fast vessels and under-flags during AIS gaps). Instead, **per vessel only the ping nearest in time** to `scene.datetime` (within `--dt-minutes`) is admissible — a fresher ping supersedes stale ones, so a moored vessel's old ping cannot build an envelope its newer ping refutes (found on the first real fusion, 2026-07-26) — and that ping gets a **feasible-movement envelope** `reach = sog·|t_scene − t_ping| + --slack-m` (falling back to `--max-speed-kn` when `sog` is missing/zero), qualifying iff the great-circle (haversine) distance ≤ `reach` — the "feasible movement envelope" the operational systems use (Skylight; SAR↔AIS coincident-match ~1 km). Among qualifying vessels the detection is **attributed to the nearest** (envelope decides *whether*, distance decides *which* — so an anchored vessel's big fallback envelope cannot out-bid an underway vessel 200 m away; found in the lanes adjudication 2026-07-26). An **unmatched in-MPA detection is a DARK candidate** — the product signal (`detection − AIS`); a **matched** one is an identified (cooperative) vessel, still a candidate violation in a no-take IUCN Ia/Ib MPA but down-weighted (`--matched-weight`, default 0.25). Output is one `violation_events.json` (the `CLAUDE.md` schema; `fishing_probability`/`fishing_hours` are `null` until the future GFW fishing-effort join fills them) + `violation_events.geojson` (dark points for QGIS), atomic (`.tmp`→`replace`). **House rule:** this is the FUSION stage only — live AIS ingestion (AISStream websocket / GFW Vessels+Insights / Skylight) belongs in its **own** ingester module that normalizes into the AIS contract, never in here. See `docs/ROADMAP.md` Phase 3.

`report_violations.py` (stage 4 operator output, P3) is a **read-only** view over the fusion output: it aggregates one or more `violation_events.json` files (or run dirs) into a **ranked** CSV + a self-contained HTML report ordered by `violation_score` (dark first at equal score), with `--dark-only`/`--min-score`/`--top` filters. Pure stdlib (no fusion or ingestion of its own); the HTML has zero external assets (same offline posture as `review_server.py`). Consumes the `violation_events.json` schema pinned in `docs/PIPELINE.md`.

`scan_violations.py` (stage 4 batch driver, P3) is a **thin orchestrator**, not a data stream: given a run dir + `--ais` file it sequences `fuse_violations` → (with `--gfw`) `gfw_vessels --from-events` + re-fuse `--vessels` → `report_violations`, each via that stage's `main(argv)`. Optional `--detect` runs `detect_boats` first (lazy-imported ML deps; do-from-home). `run_pipeline()` is pure control-flow (unit-tested by monkeypatching the stage wrappers); the fuse/report path runs offline over the committed fixtures. It reimplements no stage — just ordering + argv plumbing + a per-stage log.

`marinecadastre_fetch.py` (historical AIS ingest, P3) is a **sibling AIS ingester** (house rule; never imports `fuse_violations.py`) for the *historical* case AISStream can't cover (it's live-only). It normalizes a NOAA MarineCadastre archive AIS CSV (`MMSI`/`BaseDateTime`/`LAT`/`LON`/`SOG`/`COG`/`VesselName`) into the same neutral AIS contract, filtered to an AOI (`--bbox`/`--wdpa-id`) and optional time window (`--start`/`--end`) — essential since the national daily files are huge. Column matching is case-insensitive + aliased; AIS not-available sentinels (SOG `102.3` kn / COG `360`°) map to `null` (same as the AISStream adapter — NOAA archives carry them too); the row/time/filter logic is pure and offline-tested (`--csv <local file>` is the here-runnable path). The archive **download** (`--date`, `coast.noaa.gov` zips) is a thin do-from-home shell. Writes to `data/raw/ais/` with `source: "marinecadastre"`.

The **normalized AIS input contract** (`--ais`): JSON (a bare list, or `{"positions": [...]}`) or CSV (header row) of records `{mmsi, lon, lat, timestamp, sog?, cog?}` — WGS84 decimal degrees, `timestamp` ISO-8601 UTC (`Z`/offset) or epoch seconds, `sog` in **knots**, `cog` in degrees (both optional; `cog` is carried, not used in the gate). Field aliases are tolerated (`longitude`/`Longitude`, `time_utc`, `Sog`, …). Mapping from **AISStream** `PositionReport`: `mmsi←MetaData.MMSI`, `lat/lon←MetaData.latitude/longitude`, `timestamp←MetaData.time_utc`, `sog←Message.PositionReport.Sog` (knots), `cog←Message.PositionReport.Cog` (degrees).

The WDPA split-shapefile format is documented in `data/raw/WDPA/Shapefile_splitting_README.txt`.

---

# Design proposal — north-star vision vs. what's built

**The goal:** a real-time + batch pipeline that flags likely fishing violations inside no-take MPAs by fusing **AIS**, **GFW fishing-effort/identity**, and **Sentinel-1 SAR + Sentinel-2 optical** imagery into a ranked list of intrusion events with evidence and confidence. The built system is a **$0-budget, local, pure-stdlib + free-API** implementation; the full aspirational stack (cloud-native infra — Kubernetes/Kafka/PostGIS/Airflow/Prometheus — plus AIS track reconstruction and an alerting UI) is the **north-star, deliberately out of scope**, tracked as TODOs in `docs/ROADMAP.md` → "Remaining TODO items", not what runs today.

**Capability → code status** (every unbuilt item is a ROADMAP TODO):

| Executive-summary capability | Status | Where / note |
|---|---|---|
| Ingest AIS feeds | ✅ built | `aisstream_fetch.py` (live), `marinecadastre_fetch.py` (historical archive) |
| Ingest VMS | ⛔ out of scope | not public (partnership-only) |
| Ingest GFW APIs | ✅ built | `gfw_fetch.py` (gridded effort), `gfw_vessels.py` (per-vessel identity/fishing/Insights) |
| Preprocess AIS *tracks* (clean/interpolate/index) | ⬜ not built | superseded by the **detection-centric** reframe — we fuse instantaneous *positions*, not reconstructed tracks (ROADMAP TODO) |
| Intersect positions with MPA geofences | ◑ partial | detections tagged `inside_mpa` (`detect_boats.py`); geofencing *AIS-reported* vessels is a ROADMAP TODO |
| Infer fishing activity | ✅ (GFW) | `gfw_vessels.py` `fishing_hours` + `fishing_probability`; a local speed-based heuristic is a ROADMAP TODO |
| Cross-reference vessel identity | ✅ built | `gfw_vessels.py` (MMSI→name/flag/gear/imo, IUU flag) |
| Score each violation event | ✅ built | `fuse_violations.py` `violation_score` (see the corrected formula under *Violation scoring* below) |
| Validate with satellite imagery | ✅ built | `sat_fetch.py` + `detect_boats.py` (S2 optical live; S1 SAR provider built, detector `best_sar.pt` pending); cross-modality S1×S2 is a ROADMAP TODO |
| Ranked list of events + evidence/confidence | ✅ built | `report_violations.py` (ranked CSV + self-contained HTML); one-shot via `scan_violations.py` |
| Alerting interface | ⬜ not built | ROADMAP TODO (feasible offline over `violation_events.json`) |
| Cloud-native deploy / monitoring / scale | ⛔ out of scope | deliberately not pursued ($0-budget, local); the cloud stack is aspirational (see *Satellite validation & remaining build-out* below) |

**Live-validation** of the AISStream/GFW calls and the **SAR detector** are 🏠 do-from-home (need credentials / GPU); see `docs/PHASE3_PLAN.md` and `docs/ROADMAP.md` → "Remaining TODO items".

---

# Pipeline shape

`DETECT (S2 optical live · S1 SAR built) → CORRELATE vs AIS/GFW → CONFIRM (dark? S1×S2 + human review) → ranked violation_events`. The per-stream data sources (AIS/VMS/GFW/Sentinel-1/Sentinel-2/MPA geofences) and their access methods are covered by the **Script Architecture** table and **Data Layout** section above; the phased target architecture is in `docs/ROADMAP.md`.

---

# Violation scoring (as implemented)

Each violation candidate carries the fields pinned in the `docs/PIPELINE.md` `violation_events.json` schema (`vessel_id`/`MMSI`/`vessel_name`/`flag`/`gear_type`, `mpa_id`/`mpa_name`, `presence_confidence`, `fishing_hours`/`fishing_probability`, `violation_score`, `evidence`).

**Scoring — original proposal formula (kept for the record):**
```
violation_score = presence_confidence * fishing_probability * (fishing_hours / (duration_hours + ε))
```

**Scoring — as actually implemented in `fuse_violations.py`.** A satellite acquisition is instantaneous, so there is no track `duration_hours` (it is `null`) and the formula above is **not computable** here — it is *not* what runs. Instead:
- **dark** (in-MPA detection with no AIS match) → `violation_score = presence_confidence` (the product signal).
- **matched** (AIS-identified) → `presence_confidence × w`, where `w` interpolates `--matched-weight` (fishing_probability 0) → `--fishing-weight` (fishing_probability 1); an IUU-listed vessel escalates to `--fishing-weight`.

`fishing_probability` / `fishing_hours` are filled only by the optional GFW join (`gfw_vessels.py` → `fuse_violations.py --vessels`); without it they are `null` and matched events score at `matched_weight`. The authoritative emitted schema is pinned in `docs/PIPELINE.md`.

---

# Satellite validation & remaining build-out

**Two validation modalities:** Sentinel-1 SAR is all-weather day/night (~6-day revisit), catches dark/large metallic hulls but carries no vessel ID; Sentinel-2 optical is daytime + cloud-limited (5-day revisit), confirms tracked vessels + smaller boats in daylight. Both are ~10 m GSD (marginal below ~15 m) — details in the `sat_fetch.py` provider notes above.

The aspirational parts of the original proposal — the SQL `violation_events` table (the authoritative emitted schema is the `violation_events.json` pinned in `docs/PIPELINE.md`), the **cloud-native infra** (Kubernetes/Kafka/PostGIS/Airflow/Prometheus), and the **implementation checklist** — are the north-star / remaining build-out, tracked in `docs/ROADMAP.md` → "Remaining TODO items". They are not restated here; git history has the full original proposal.
