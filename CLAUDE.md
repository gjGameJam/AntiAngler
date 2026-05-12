# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AntiAngler is a geospatial data processing and MPA violation detection project. It fuses AIS tracking, Global Fishing Watch (GFW) fishing-effort data, and satellite imagery (Sentinel-1 SAR and Sentinel-2 optical) to identify vessels entering Marine Protected Areas (MPAs) and assess their behavior.

The initial pipeline filters the World Database of Protected Areas (WDPA) for MPAs with IUCN category Ia (Strict Nature Reserve) or Ib (Wilderness Area). The full system then ingests AIS/VMS feeds and GFW APIs, preprocesses tracks, intersects vessel positions with MPA geofences, infers fishing activity, cross-references vessel identity with registries, and scores each violation event.

## Running Scripts

```bash
# Step 1: Filter raw WDPA shapefiles → processed GeoPackage
python scripts/prep_polygons.py

# Step 2: Visualize filtered MPAs on a world map
python scripts/view_polygons.py

# Step 3: Fetch GFW fishing-effort data via 4WINGS API
python scripts/gfw_fetch.py
```

There is no build system, test suite, or requirements file. Dependencies must be installed manually: `geopandas`, `pandas`, `geodatasets`, `matplotlib`, `requests`, `python-dotenv`.

GFW API requires a `GFW_API_TOKEN` in a `.env` file at the project root or inside `scripts/`.

## Data Layout

- `data/raw/WDPA/` — Raw WDPA September 2025 shapefiles, split into 3 parts (`WDPA_Sep2025_Public_shp_0`, `_1`, `_2`) due to size. These are gitignored.
- `data/processed/wdpa_marine_ia_ib.gpkg` — Output: filtered marine MPAs in WGS84 (EPSG:4326).
- `data/raw/WDPA/Resources_in_English/` — Official WDPA attribute/metadata PDFs for reference.

## Script Architecture (Compartmentalized by Data Stream)

Each data stream has its own script. Do not mix data-stream concerns across scripts.

| Script | Data Stream | Description |
|--------|-------------|-------------|
| `prep_polygons.py` | WDPA / MPA boundaries | Filters raw WDPA shapefiles → GeoPackage |
| `view_polygons.py` | Visualization | Overlays processed MPAs on a world map |
| `gfw_fetch.py` | Global Fishing Watch API | Queries GFW 4WINGS API for fishing effort by vessel |

`prep_polygons.py` reads each of the 3 split shapefiles, concatenates them into a single GeoDataFrame, filters rows where `MARINE` is `1` or `2` and `IUCN_CAT` is `Ia` or `Ib`, reprojects to EPSG:4326, and writes to a GeoPackage.

`view_polygons.py` loads the processed GeoPackage and overlays it on a Natural Earth world background via `geodatasets`.

`gfw_fetch.py` posts to the GFW `/v3/4wings/report` endpoint, requesting fishing effort grouped by vessel ID within a region (currently EEZ ID 8466) and date range.

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
