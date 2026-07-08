# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AntiAngler is a geospatial data processing and MPA violation detection project. It fuses AIS tracking, Global Fishing Watch (GFW) fishing-effort data, and satellite imagery (Sentinel-1 SAR and Sentinel-2 optical) to identify vessels entering Marine Protected Areas (MPAs) and assess their behavior.

The initial pipeline filters the World Database of Protected Areas (WDPA) for MPAs with IUCN category Ia (Strict Nature Reserve) or Ib (Wilderness Area). The full system then ingests AIS/VMS feeds and GFW APIs, preprocesses tracks, intersects vessel positions with MPA geofences, infers fishing activity, cross-references vessel identity with registries, and scores each violation event.

The end-to-end satellite ship-detection loop — collect imagery over MPAs (`sat_fetch.py`) → scan chips for boats (`detect_boats.py`) → report detections + human-in-the-loop review (`review_server.py`) → export corrected labels to retrain the model (`export_labels.py`) — is built and specified in `docs/PIPELINE.md` (the build spec with all inter-stage data contracts). `docs/AUDIT.md` tracks known antipatterns/inefficiencies in priority order.

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
```

No build system or test suite yet. Runtime dependencies are pinned in `requirements.txt` at the repo root — install with `pip install -r requirements.txt`. The pinned set covers `gfw_fetch.py` (`requests`, `python-dotenv`) and `sat_fetch.py` (`numpy`, `rasterio`, `geopandas`, `pystac-client`, `planetary-computer` — a heavier stack; `rasterio`/`geopandas` bundle GDAL/GEOS/PROJ, installed from PyPI wheels on Windows). The stage 2–3 loop (`detect_boats.py`, `review_server.py`, `export_labels.py`) needs the heavier ML/web stack pinned separately in `requirements-ml.txt` (`ultralytics`→torch, `fastapi`, `uvicorn`, `pillow`) — install it *in addition* (`pip install -r requirements-ml.txt`); it is kept out of `requirements.txt` so the daily gfw cron does not pull torch. `prep_polygons.py` and `view_polygons.py` additionally need `pandas`, `geodatasets`, and `matplotlib` (matplotlib now arrives transitively via `ultralytics`), not yet pinned in `requirements.txt`. A `venv/` is present locally but uncommitted.

GFW API requires a `GFW_API_TOKEN` in a `.env` file at the project root or inside `scripts/` (`gfw_fetch.py` loads both locations). `sat_fetch.py` needs no credentials (Planetary Computer is open); an optional `PC_SDK_SUBSCRIPTION_KEY` only raises rate limits.

**TLS note (corporate/AV proxy environments):** if an antivirus or proxy intercepts TLS (e.g. Avast), `sat_fetch.py`'s Python HTTPS calls need the interceptor's CA — set `REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE` to a bundle containing it (certifi + the AV root). GDAL's `/vsicurl` COG reads use the Windows schannel backend, which may still fail with a partial-chain error under interception; that is an environment/cert-store matter, not a code issue.

## CI / Automation

`.github/workflows/gfw.yml` runs `scripts/gfw_fetch.py` on:
- `workflow_dispatch` (manual trigger)
- `schedule: cron '0 0 * * *'` (daily at 00:00 UTC)

The workflow uses Python 3.11 with `actions/setup-python@v5` pip caching (keyed on `requirements.txt`), installs deps via `pip install -r requirements.txt`, reads `GFW_API_TOKEN` from GitHub secrets, and uploads `data/raw/gfw/` as the `gfw-fishing-effort-report` artifact with `if-no-files-found: error` (CI fails loudly if no file was written).

## Data Layout

- `data/raw/WDPA/` — Raw WDPA September 2025 shapefiles, split into 3 parts (`WDPA_Sep2025_Public_shp_0`, `_1`, `_2`) due to size. These are gitignored.
- `data/raw/gfw/` — Per-run GFW 4WINGS API responses written by `gfw_fetch.py` as `gfw_report_<UTC-timestamp>.json`. Gitignored; surfaced via the CI artifact.
- `data/raw/sentinel2/<UTC-timestamp>_<label>/` — Per-run Sentinel-2 imagery from `sat_fetch.py`: a `chips/` dir of georeferenced RGB GeoTIFF tiles plus a `manifest.json` (chosen scene id/datetime/cloud + its AOI coverage & AOI cloud, the `selection` block with the thresholds used and whether a best-effort `fallback_used` scene was accepted, probed `candidates`, AOI bbox, scene CRS, stretch params, and per-chip pixel window + WGS84/UTM bounds). Gitignored. The `manifest.json` is the seam `detect_boats.py` reads (schema in `docs/PIPELINE.md`). Stage 2–3 then write three more files *into the same run dir*: `detections.json` + `detections.geojson` (stage 2 output — geolocated boxes, the QGIS-loadable GeoJSON is points) and `reviews.json` (stage 3 human verdicts, keyed by chip).
- `data/processed/training_exports/<tag>/` — YOLO training sets from `export_labels.py`: `images/` (chip PNGs) + `labels/` (`cls cx cy w h`, empty file = hard negative) + `export_log.json` provenance. `<tag>` defaults to the UTC date and accumulates runs. Gitignored. Consumed by `build_dataset.py`.
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
| `review_server.py` | Human review (stage 3) | Local hardened FastAPI UI to mark detections correct/incorrect and draw missed boats → `reviews.json` |
| `export_labels.py` | Active learning (stage 3) | Turns reviewed chips into a YOLO training set to fine-tune the detector |
| `build_dataset.py` | Model training (stage 4) | Layers reviewed exports onto the base dataset in `data/training/` (idempotent) |
| `train.py` | Model training (stage 4) | Trains/fine-tunes the YOLOv8 boat detector in-repo; writes `data/training/runs/<name>/` |

`prep_polygons.py` reads each of the 3 split shapefiles, concatenates them into a single GeoDataFrame, filters rows where `MARINE` is `1` or `2` and `IUCN_CAT` is `Ia` or `Ib`, reprojects to EPSG:4326, and writes to a GeoPackage.

`view_polygons.py` loads the processed GeoPackage and overlays it on a Natural Earth world background via `geodatasets`. Defaults to interactive `plt.show()`; pass `--save PATH.png` to switch matplotlib to the `Agg` backend and write a PNG instead (headless-safe for CI).

`gfw_fetch.py` posts to the GFW `/v3/4wings/report` endpoint, requesting fishing effort grouped by vessel ID within a region. All eight query knobs (`--region-id`, `--region-dataset`, `--start`, `--end`, `--gear`, `--group-by`, `--spatial-resolution`, `--temporal-resolution`, `--format`, `--dataset`) are exposed as CLI flags with matching `GFW_*` env-var fallbacks — CLI overrides env. Defaults: EEZ `8466` / `public-eez-areas`, gear `trawlers`, `VESSEL_ID` / `LOW` / `ENTIRE` / `JSON`, dataset `public-global-fishing-effort:latest`. If `--start`/`--end` are omitted, the window is yesterday UTC (single day), so the daily cron pulls a rolling window. On 2xx, the JSON response is written atomically to `data/raw/gfw/gfw_report_<UTC-timestamp>.json` (write to `.tmp`, then `Path.replace()`); non-2xx responses raise before any file is written, so the raw dir stays clean.

`sat_fetch.py` takes an AOI as either an explicit `--bbox minLon minLat maxLon maxLat` or an MPA lookup (`--wdpa-id` / `--wdpa-pid`, resolved against `wdpa_marine_ia_ib.gpkg`). It searches Planetary Computer's `sentinel-2-l2a` STAC in the window (default: ~30 days ending yesterday, scene-wide `eo:cloud_cover < --max-cloud`), then **selects AOI-aware**: it probes up to `--max-search` candidates (least scene-cloud first) with the SCL cloud mask read *over the AOI itself*, and picks the first that meets `--min-coverage` (AOI overlap) and `--max-aoi-cloud` (cloud/shadow over the AOI) — because scene-wide cloud % is misleading for a small AOI. If none pass, it uses the best effort and warns; `--dry-run` prints the ranked candidate table and writes nothing (use it to find a good `--start`/`--end`/`--max-cloud` for a new location). The chosen scene's assets are signed and B04/B03/B02 windowed-read in the scene's **native UTM** via `rasterio`; surface reflectance is stretched to 8-bit RGB once over the whole AOI (`--stretch linear|percentile`; match this to the detection model's training preprocessing), then tiled into `--tile-size` (default 640) chips with `--overlap` (default 64), skipping blank/nodata chips. Every knob has a `SAT_*` env-var fallback (CLI > env > default). Output is written atomically (temp run dir → `Path.replace()`). Each chip GeoTIFF is self-describing (carries its own CRS + affine), so downstream code needs only `rasterio.open(chip)` to map pixels → lat/lon. Known limitation: Sentinel-2's 10 m GSD is marginal for vessels smaller than ~15 m; SAR (Sentinel-1) would suit small open-ocean targets and can be added behind the same `open_catalog`/`search_candidates`/`select_scene`/`read_aoi` seam. Downstream, a model-runner (see `C:\gtest\BoatDetection`; full spec in `docs/PIPELINE.md`) consumes `manifest.json`, runs inference per chip, and maps detections back to lat/lon via each chip's transform.

`detect_boats.py` (stage 2) reads a `sat_fetch.py` run dir and runs the YOLOv8 boat detector (`--weights`, default the sibling `C:\gtest\BoatDetection` `best.pt`) on each chip. Each chip GeoTIFF is self-describing, so a box's pixels map chip→UTM (chip affine)→WGS84 directly. Detections are pooled in the scene's UTM and **deduplicated by merging centroids within `--merge-dist` metres** (IoU is unstable for 1–5 px vessels, so dedup is metric, not pixel/degree). For WDPA runs it tags each detection `inside_mpa` (shapely point-in-polygon; `--inside-mpa-only` drops the rest). Writes `detections.json` (+ point `detections.geojson`) atomically. Keep `--conf` low (~0.25) for recall; the model was trained on oblique photos not overhead imagery, so treat first-pass hits as *candidates to review*, not answers.

`review_server.py` (stage 3) is a minimal, hardened, single-user **localhost** FastAPI UI (`--run <run dir>`, `--port`). It overlays `detections.json` on each chip and lets a human mark boxes correct/incorrect and drag missed boats, persisting verdicts to `reviews.json` (atomic). Security posture (single-user local tool): binds `127.0.0.1` only; a Host-header allowlist middleware blocks DNS-rebinding; a random startup token (printed in the URL) gates every route; chips are served by opaque manifest index, never a client path (defense-in-depth: `resolve()` + `is_relative_to`); strict per-request-nonce CSP; the page is fully self-contained (inline canvas UI, zero external assets); POST bodies are pydantic-validated (`extra="forbid"`, `Literal` verdicts). "Mark reviewed" is gated on every detection being judged — the *complete-labels* rule that keeps the export trainable.

`export_labels.py` (stage 3) turns reviewed chips into a YOLO training set under `data/processed/training_exports/<tag>/` (`images/*.png` + `labels/*.txt`, `cls cx cy w h` normalized; class 0 = boat). It keeps boxes verdicted `correct` or `missed`, drops `incorrect`, and writes an *empty* label for a reviewed chip with no boats — a hard negative that suppresses false positives. Only fully-reviewed chips are exported (complete-labels rule). Then `build_dataset.py` layers the accumulated exports onto the base dataset in `data/training/` and `train.py` fine-tunes from `best.pt` (or trains fresh); re-run stage 2 with the new weights — the active-learning loop, now fully in-repo. Full loop + data contracts in `docs/PIPELINE.md`.

The WDPA split-shapefile format is documented in `data/raw/WDPA/Shapefile_splitting_README.txt`.

---

# Executive Summary

We propose a **real-time and batch pipeline** to detect likely fishing violations inside Marine Protected Areas (MPAs). The system fuses **AIS tracking**, **Global Fishing Watch (GFW) fishing-effort data**, and **satellite imagery (Sentinel-1 SAR and Sentinel-2 optical)** to identify vessels entering MPAs and assess their behavior. Key components: ingest AIS/VMS feeds and GFW APIs; preprocess tracks (cleaning, interpolation, indexing); intersect vessel positions with MPA geofences; infer fishing activity via GFW or local ML; cross-reference vessel identity with registries; score each "violation event" (presence confidence, fishing probability, violation likelihood); and optionally validate suspicious events with satellite images. The output is a ranked list of MPA intrusion events (with evidence and confidence) and an alerting interface. The design supports cloud-native deployment (containers, scalable storage), automatic monitoring, and extensibility (e.g. adding SAR-based dark-vessel detection later).

**Assumptions:** We assume global AIS visibility (satellite + terrestrial) with typical delays, as well as access to open Sentinel data. We assume processing scale on the order of millions of AIS messages/day and dozens of MPA polygons. We assume MPAs and vessel registries are static or slowly updating. Retention is flexible; e.g., raw AIS ~1 year, aggregated events ~5–10 years.

---

# Data Sources and Access Methods

| **Data Stream** | **Description & Content** | **Access Method** | **Use in Pipeline** |
|---|---|---|---|
| **AIS** | Vessel-broadcast GPS (lat,lon), course, speed, identity (MMSI/IMO/call sign). High-frequency "breadcrumbs" of vessel movement. | Real-time feeds (MarineCadastre NOAA, ORCA network, AIS-Hub) or downloaded archives (hourly CSVs) | Core trajectory data. Ingest raw tracks; build vessel routes. Input to geofence and behavior models. |
| **VMS** | Government-tracked vessel locations (typically large fishing vessels) with ID. | Available only via partnerships (NOAA-Fisheries, RFMOs). Not public. | Augments AIS for compliant fleets. Can directly flag fishing vessels. |
| **GFW AIS-derived Fishing Effort** | Inferred fishing activity (hours, events) and vessel presence from AIS patterns. Includes gear type, timestamps. | GFW 4WINGS API (`/v3/4wings/report`). `gfwr` Python package or REST (requires token). | Provides fishing probability/effort for segments. Integrated into event scoring. |
| **GFW Vessel Registry & Insights** | Aggregated vessel metadata (IMO/MMSI, name, flag, type, gear, owner) from ~40 registries; AIS-off events via Insights API. | GFW Vessel API (`/v3/vessels`) and Insights API (`/v3/insights/vessels`). Requires token. | Resolve vessel identity (MMSI→IMO/name), detect AIS-dark events, flag known IUU listings. |
| **Sentinel-2 (Optical)** | 10 m multispectral optical satellite images (twin satellites, 5-day revisit). Cloud-limited. | ESA Copernicus APIs, Sentinel Hub, AWS/GCP. Python: `sentinelsat`, `sentinelhub-py`, or Google Earth Engine. | Validate events: spot actual vessels in MPA. Provides visual evidence (with ML-based vessel detection). |
| **Sentinel-1 (SAR)** | C-band SAR images (all-weather, day/night, global 6-day revisit). Detects large metallic objects on surface. | ESA / NASA (ASF DAAC) via APIs. AWS Open SAR (GDAL). | (Future) Detect AIS-dark ships. After events, search SAR footprints to verify unreported ships. |
| **Oceanographic Data** | Environmental layers: SST, chlorophyll-a, bathymetry, etc. Satellite-derived or model (MODIS, VIIRS, NOAA). | NOAA ERDDAP, Copernicus. APIs or bulk NetCDF downloads. | Auxiliary features: enrich behavior model or risk scores. |
| **MPA Geofences** | Polygons defining boundaries of MPAs. Includes level of protection. | GFW `public-mpa-all` dataset via API or WDPA (protectedplanet.net). Download as GeoJSON/Parquet. | Used for point-in-polygon checks. Tag vessel trajectories/events with which MPA they intersect. |

---

# System Architecture and Data Flow

```
Data Ingestion → Processing → Event Detection → Satellite Validation → Output
```

- **Data Ingestion:** AIS/VMS streams → raw datastore (message queue or data lake). Satellite imagery fetched on-demand or scheduled batch. GFW API outputs pulled periodically (daily/weekly). Oceanographic data loaded into geo-database.
- **Processing:** Raw AIS is cleaned (remove invalid, duplicate, interpolate gaps), then segmented into vessel trajectories. Tracks feed into GeofenceCheck (point-in-polygon against MPA shapes) and into a BehaviorModel that uses AIS features (e.g. speed patterns) to detect fishing, or directly queries GFW fishing-effort.
- **Event Detection:** When a track intersects an MPA, an Event is created (with start/end times). Features computed: time inside, distance from shore, AIS-off events, etc. Each event is scored (presence_confidence, fishing_probability, overall violation_score).
- **Satellite Validation:** SAR and optical image detectors run concurrently. Detections are matched to AIS tracks by time/space. This can confirm the vessel's presence or reveal a "ghost ship" (no AIS).
- **Output:** Violation events (with vessel info, MPA, times, scores, evidence tags) stored and exposed via an API or dashboard.

---

# Geofence Intersection & Event Detection

**Event Schema — each violation candidate includes:**
- `vessel_id`, `MMSI/IMO`, `vessel_name`, `flag`, `gear_type`
- `mpa_id`, `mpa_name`
- `entry_time`, `exit_time`, `duration_hours`
- `distance_from_port_km`, `presence_confidence`
- `fishing_hours`, `fishing_probability`, `violation_score`
- `evidence` — list of data sources confirming the event (e.g. `["AIS", "GFW", "Sentinel-2"]`)

**Scoring:**
```
violation_score = presence_confidence * fishing_probability * (fishing_hours / (duration_hours + ε))
```

---

# Satellite Validation

| Feature | Sentinel-1 (SAR) | Sentinel-2 (Optical) |
|---|---|---|
| **Coverage** | All-weather, day/night. ~6-day revisit (global). | 5-day revisit (twin satellites), but daytime only and no cloud penetration. |
| **Resolution** | ~10 m (C-band) – suitable for large vessels. | 10 m (RGB/NIR bands); can detect smaller vessels (~5–10 m) with AI. |
| **Detects** | Reflective objects (ships), including dark ones, but no ID info. | Bright hulls in visible spectrum. |
| **Best For** | Confirming untracked/dark vessels, large trawlers at night. | Confirming tracked vessels, small boats in daylight, creating heatmaps. |

---

# Output Schema

```sql
CREATE TABLE violation_events (
  event_id SERIAL PRIMARY KEY,
  vessel_id UUID REFERENCES vessels(vessel_id),
  mpa_id INTEGER,
  entry_time TIMESTAMP,
  exit_time TIMESTAMP,
  duration_hours FLOAT,
  presence_confidence FLOAT,
  fishing_hours FLOAT,
  fishing_probability FLOAT,
  violation_score FLOAT,
  evidence JSONB,
  created_at TIMESTAMP DEFAULT now()
);
```

---

# Infrastructure Recommendations

- **Compute:** Containerized microservices (Python) on Kubernetes or serverless. AWS EKS or GCP GKE for scalable tasks.
- **Storage:** Object storage (S3/GCS) for raw AIS dumps and satellite images. PostgreSQL+PostGIS for spatial queries. Parquet on object storage for batch analysis.
- **Streaming:** Kafka/Kinesis for AIS ingestion at scale.
- **Orchestration:** Apache Airflow / Prefect / Dagster for ETL flows (daily GFW pulls, AIS batch jobs).
- **Monitoring:** Prometheus/Grafana for pipeline health (ingestion lag, event rates, error rates). Structured JSON logging.

---

# Implementation Checklist

- [ ] **Data Integration:** Set up AIS ingestion (stream or batch) into raw store. Load MPA boundaries into spatial DB.
- [ ] **Preprocessing Pipeline:** Implement AIS cleaning/dedup logic and track reconstruction.
- [ ] **Geofence Logic:** Code MPA intersection and entry/exit detection. Unit test with synthetic trajectories.
- [ ] **Behavior Scoring:** Hook in GFW API calls to fetch fishing hours by vessel. Integrate speed-based fishing heuristic.
- [ ] **Identity Service:** Implement `IdentityResolver` with GFW Vessel API lookup.
- [ ] **Event Database:** Design and create `violation_events` table.
- [ ] **Output & Alerts:** Develop API or dashboard to browse events. Configure alert triggers for high scores.
- [ ] **Sentinel Proof-of-Concept:** Fetch Sentinel-2 for a test MPA/time, run a simple ship detection, validate against known vessel.
- [ ] **SAR Integration (Future):** Prototype dark-vessel detection by matching AIS gap events to SAR detections.
- [ ] **Monitoring & Testing:** Add logging/metrics to each component. Write end-to-end tests.
