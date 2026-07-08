# AntiAngler Detection Pipeline — Build Spec

**Goal:** collect satellite imagery over Marine Protected Areas (MPAs), scan it for boats,
report the boats, and let a human mark each detection correct/incorrect so the model can be
retrained. This document is the implementation spec for the whole loop: what exists, what is
left to build, and — most importantly — the **data contracts (seams)** between stages so each
piece can be built and tested in isolation.

```
 ┌───────────┐   manifest.json   ┌───────────┐  detections.json  ┌───────────┐  reviews.json   ┌────────────┐
 │ 1 COLLECT │ ────────────────► │ 2 SCAN    │ ────────────────► │ 3 REPORT  │ ──────────────► │ RETRAIN    │
 │ sat_fetch │  georef RGB chips │detect_boats│  boats + lat/lon  │  + REVIEW │  human verdicts │ (BoatDetect)│
 └───────────┘                   └───────────┘                   └───────────┘                  └────────────┘
      built                          built                           built                     exists; wire up
                                                                        └──── active-learning loop ────┘
```

Everything for one scan lives in a single run directory
(`data/raw/sentinel2/<UTC-ts>_<label>/`) so a scan is self-contained and reproducible:

```
data/raw/sentinel2/<UTC-ts>_<label>/
  chips/chip_r00000_c00000.tif ...   # stage 1 — georeferenced 8-bit RGB, self-describing
  manifest.json                      # stage 1 → 2 seam  (EXISTS)
  detections.json                    # stage 2 → 3 seam  (NEW)
  detections.geojson                 # optional, for QGIS overlay (NEW)
  reviews.json                       # stage 3 human verdicts (NEW)
data/processed/training_exports/<date>/
  images/*.png  labels/*.txt         # stage 3 → retrain seam, YOLO format (NEW)
```

---

## Table of contents

1. [Stage 1 — Collect (built)](#stage-1--collect-built)
2. [Stage 2 — Scan (built)](#stage-2--scan-to-build)
3. [Stage 3 — Report + Review (built)](#stage-3--report--review-to-build)
4. [The active-learning loop → retrain](#the-active-learning-loop--retrain)
5. [Data contracts (all schemas in one place)](#data-contracts)
6. [New files, dependencies, gitignore](#new-files-dependencies-gitignore)
7. [Build order / checklist](#build-order--checklist)
8. [Cross-cutting concerns](#cross-cutting-concerns)

---

## Stage 1 — Collect (built)

`scripts/sat_fetch.py` turns a bounding box (explicit `--bbox`, or an MPA via `--wdpa-id` /
`--wdpa-pid` looked up in `data/processed/wdpa_marine_ia_ib.gpkg`) into a run directory of
640×640 georeferenced 8-bit RGB GeoTIFF chips plus `manifest.json`. It selects a recent
Sentinel-2 L2A scene from Microsoft Planetary Computer (no credentials) **AOI-aware**: it probes
up to `--max-search` candidates (least scene-cloud first) using the SCL cloud mask read *over the
AOI itself* and picks the first meeting `--min-coverage` and `--max-aoi-cloud`, because scene-wide
cloud % is misleading for a small AOI. It then windowed-reads B04/B03/B02 in the scene's native
UTM, stretches reflectance → 8-bit once over the whole AOI, and tiles with a 64-px overlap.
`--dry-run` prints the ranked candidate table (scene% / AOI-coverage / AOI-cloud) and writes
nothing — the fast way to find good `--start`/`--end`/`--max-cloud` for a new location anywhere on
Earth. Verified across hemispheres/UTM zones (Singapore 48N, False Bay 34S). See `CLAUDE.md` for
the full flag list and the Avast/TLS note (`--insecure-tls` + CA-bundle env vars on this machine).

**Provider/sensor seam** (add SAR or another catalog behind these): `open_catalog` →
`search_candidates` → `select_scene`/`assess_aoi` (SCL probe) → `read_aoi`. **Antimeridian** AOIs
are rejected with guidance to fetch each side as its own run (the halves are in different UTM
zones, so one scene can't span them).

**The key property downstream code relies on:** *each chip GeoTIFF is self-describing.* Opening
it with rasterio gives `src.crs` (a UTM zone) and `src.transform` (pixel → UTM affine). That is
all stage 2 needs to convert a pixel box to lat/lon — the manifest is a convenience/index, the
GeoTIFF is authoritative.

### Gap: collecting over *all* MPAs (the "marine protected areas" plural)

`sat_fetch.py` handles **one AOI, one scene** per run. To sweep every MPA:

- Large MPAs exceed `--max-bbox-deg 1.0` and can span multiple S2 tiles and **UTM zones**; a
  single windowed read won't cover them. Antimeridian-crossing AOIs are also rejected today.
- **Batch driver to build — `scripts/scan_mpas.py`:** iterate selected `WDPAID`s from the gpkg;
  for each, split its `total_bounds` into ≤1° sub-AOIs on a regular grid; call the fetch logic
  per sub-AOI (reuse `sat_fetch.py`'s functions — import them, don't shell out); collect the
  run dirs. Each sub-AOI stays within one scene/UTM zone, sidestepping the multi-zone problem.
- **Interim:** for MPAs that fit in 1°, just loop `--wdpa-id` over a shortlist. The Ia/Ib marine
  set is small; a shell/PowerShell loop is fine until `scan_mpas.py` exists.

This gap does **not** block stages 2–3 — build those against a single run dir first, then add the
batch driver to fan out.

---

## Stage 2 — Scan (built)

**`scripts/detect_boats.py`** reads a stage-1 run dir, runs the BoatDetection YOLOv8 model on
every chip, maps each box back to lat/lon, deduplicates across chip overlaps, optionally keeps
only boats inside the MPA polygon, and writes `detections.json` + `detections.geojson`.

```bash
python scripts/detect_boats.py --run data/raw/sentinel2/<run>/ \
    [--weights <path/to/best.pt>] [--conf 0.25] [--merge-dist 30] [--inside-mpa-only]
```

**As built** (validated on a Singapore run: 312 raw → 122 deduped, all centroids inside the AOI):
dedup is **centroid-distance merging in UTM metres** (`--merge-dist`, default 30 m), *not* IoU-NMS
— research confirmed IoU is numerically unstable for 1–5 px vessels. Detections are pooled in the
scene's single UTM CRS and merged there, then converted to WGS84. The model is fed a **PIL RGB
image** (Ultralytics reads a numpy array as BGR, which would swap R/B on our true-color chips). No
SAHI dependency: our 640/64px chips already *are* correct SAHI-style slices, so only the merge step
was needed. Default `--conf 0.25` for recall.

### "Coarse grid → cells" — yes, and here's the precise mapping

Your instinct is right, and stage 1 already implements it: **the 640×640 chips are the cells of a
coarse grid.** At 10 m GSD a chip is 6.4 km on a side, and they overlap by 64 px (640 m) so a
boat straddling a cell boundary still lands whole inside at least one neighbour. So stage 2 is
just "run the detector on each cell, then stitch results back into one map."

Two refinements worth knowing, neither required for a first version:

- **Optional coarse *reject* pass.** You could cheaply skip cells that are obviously empty water
  or cloud before spending a full detector pass on them. The clean signal for this is Sentinel-2's
  **SCL** (Scene Classification Layer) band — per-pixel water/cloud/land classes — which
  `sat_fetch.py` could read alongside RGB and record a per-chip `water_frac` / `cloud_frac` in the
  manifest; stage 2 then skips chips that are all cloud or all land. But YOLOv8n on 640 px is a
  few ms/chip on GPU, so brightness/blob pre-filters are premature optimization at MPA scale.
  **Recommendation:** don't build a coarse reject pass yet; rely on stage 1's existing
  `--min-valid-frac` nodata skip, and add SCL cloud gating only if cloud false-positives become a
  problem.
- **Fine pass would be super-resolution / SAR**, not a finer grid — see the domain-gap note below.

### Pixel box → lat/lon (the crux; get this exactly right)

Ultralytics returns boxes as `xyxy` in the chip's **pixel** frame, with `y` increasing downward —
which is exactly rasterio's row convention, so **no vertical flip is needed.** For a chip opened
as `src` (rasterio) and a detection box `[x1, y1, x2, y2]` in chip pixels:

```python
from rasterio.warp import transform as warp_transform

cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0        # centroid, chip pixels
ux, uy = src.transform * (cx, cy)                 # → scene UTM (Affine applied to a tuple)
lon, lat = warp_transform(src.crs, "EPSG:4326", [ux], [uy])   # → WGS84 (returns lists)
lon, lat = lon[0], lat[0]

# bbox corners the same way, then min/max → bbox_wgs84
uxs, uys = zip(src.transform * (x1, y1), src.transform * (x2, y2))
lons, lats = warp_transform(src.crs, "EPSG:4326", list(uxs), list(uys))
bbox_wgs84 = [min(lons), min(lats), max(lons), max(lats)]
```

Because the chip is self-describing, you need `src.transform` / `src.crs` only — the manifest's
`utm_bounds`/`wgs84_bounds` are a cross-check, not a dependency.

### Dedup across overlapping cells

A boat in the 64-px seam appears in 2–4 chips. Collect **all** detections in one global list in
world coordinates, then deduplicate:

- **Simple + robust:** greedy by descending confidence; drop any later detection whose centroid
  is within a small distance (e.g. **< 20 m ≈ 2 px**, roughly half a vessel) of a kept one.
- **Stricter:** global NMS on the WGS84/UTM boxes (IoU > `--iou`). Distance-on-centroid is usually
  enough because vessels are sparse and small; use box-IoU if boxes are large or crowded.

### Point-in-MPA filter

When the run came from a `--wdpa-id` (the manifest's `wdpa` block is non-null), load that MPA
polygon from the gpkg and keep only detections whose centroid falls inside it — a boat 5 km
offshore of the AOI bbox isn't a violation. `shapely` ships with `geopandas` (already a dep):

```python
import geopandas as gpd
from shapely.geometry import Point

mpa = gpd.read_file(gpkg)
poly = mpa.loc[mpa["WDPAID"] == manifest["wdpa"]["WDPAID"], "geometry"].iloc[0]
det["inside_mpa"] = poly.contains(Point(lon, lat))
```

Set `inside_mpa` on every detection; `--inside-mpa-only` drops the outside ones. For bbox-only
runs (no WDPA) leave `inside_mpa: null`.

### Output

Write `detections.json` into the run dir (atomic `.tmp` + `replace()`, matching house style) and,
with `--geojson`, a `detections.geojson` of point features for a one-click QGIS overlay against
the MPA polygon. Schema in [Data contracts](#data-contracts).

### Inference sketch

```python
from ultralytics import YOLO
import rasterio, numpy as np

model = YOLO(weights)
for chip in manifest["chips"]:
    with rasterio.open(run_dir / chip["filename"]) as src:
        rgb = np.transpose(src.read(), (1, 2, 0))          # (H,W,3) uint8, model-native
        res = model.predict(rgb, imgsz=640, conf=conf, verbose=False)[0]
        for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
            ...  # map to lat/lon as above, append to global list
# then: dedup → point-in-MPA → write detections.json
```

---

## Stage 3 — Report + Review (built)

Two responsibilities: **show** the detections to a human, and **capture** their verdict
(correct / incorrect / missed) in a form that can retrain the model. The verdicts are the whole
point — see [the loop](#the-active-learning-loop--retrain).

**As built** (chosen over FiftyOne/Label Studio for smallest footprint + full control):
`scripts/review_server.py` is a single-file **FastAPI + inline-canvas** app, `python
scripts/review_server.py --run <run dir>` → it prints a tokenized `127.0.0.1` URL to open.
**Security posture** (verified: no-token/bad-Host/wrong-token → 403; malformed bodies → 422):
binds `127.0.0.1` only; a **Host-header allowlist** middleware blocks DNS-rebinding; a **random
startup token** gates every route; chips are served by **opaque manifest index**, never a
client-supplied path (defense-in-depth `resolve()` + `is_relative_to`); **strict per-request-nonce
CSP**; the page is fully self-contained (zero external assets); POST bodies are **pydantic-validated**
(`extra="forbid"`, `Literal` verdicts); `reviews.json` is written atomically. "Mark reviewed" is
gated on no detection being left pending — the complete-labels rule below.

### Design notes (as implemented)

The app is a self-hosted FastAPI + one inline HTML/JS/canvas page. No cloud, no accounts, `localhost`.

- **Serve chips as PNG.** Convert each uint8 RGB GeoTIFF to PNG on the fly (`pillow` /
  `rasterio.read` → encode) so the browser can display it. Draw the `detections.json` boxes over
  the image (client-side `<canvas>` from the JSON, so the underlying chip stays clean).
- **Per-detection verdict:** each box gets **Correct** (✓, true positive) / **Incorrect** (✗,
  false positive) buttons.
- **Add missed boats:** let the reviewer drag a new box on the canvas for a boat the model missed
  (false negative). This is essential — see the "complete labels" gotcha below.
- **Mark chip reviewed:** a per-chip "done" toggle. Only fully-reviewed chips are exportable.
- **Save** → append/overwrite the chip's entry in `reviews.json` (atomic write).

Keyboard-first (←/→ between chips, keys for ✓/✗) makes labeling hundreds of chips bearable.

### Alternative: Label Studio with pre-annotations

If you'd rather not build UI, **Label Studio** (open-source, `pip install label-studio`) imports
images with pre-drawn boxes from a predictions JSON. Convert `detections.json` → Label Studio
predictions, review/correct in its mature UI, export corrected annotations, convert back to YOLO.
Trade-off: a heavier dependency and two format adapters vs. a ~200-line purpose-built server that
writes YOLO directly. **Recommendation:** build the small server — it closes the loop with fewer
moving parts and no format round-trips, and the export *is* the training data.

### The one gotcha that will silently wreck retraining: complete labels

YOLO training assumes **every** object in an image is boxed. Any real boat left unboxed in a
training image teaches the model to *suppress* real detections. Therefore:

- A chip is only exportable **once the reviewer has confirmed every true box AND added every
  missed box** — i.e. the chip is *fully* labeled, not just spot-checked.
- The "mark chip reviewed" toggle is the gate. Export (stage → retrain) pulls **only** fully
  reviewed chips.
- An **Incorrect** verdict removes that box; if a chip ends up with zero real boats it becomes a
  valid **negative** example (empty `.txt` label file) — which is exactly what teaches the model
  to stop firing on that kind of clutter (whitecaps, rocks, cloud edges).

Store all three verdict types in `reviews.json` so the export step can turn them into labels:
correct → keep box, incorrect → drop box (chip may become negative), missed → add box.

---

## The active-learning loop → retrain

The verdicts are training signal. This is how a model trained on *oblique photos* (see the
domain-gap note) becomes one that actually works on *overhead satellite* imagery — you bootstrap
a satellite-domain dataset from your own reviewed scans.

**New script: `scripts/export_labels.py`** — reads a run's `manifest.json` + `reviews.json`,
takes every fully-reviewed chip, and writes a YOLO-format dataset:

```
data/processed/training_exports/<date>/
  images/<run>__chip_r00000_c00000.png      # chip as PNG (run-prefixed to avoid name clashes)
  labels/<run>__chip_r00000_c00000.txt      # one line per kept box:  0 cx cy w h   (normalized)
```

- Class id is `0` (BoatDetection is single-class `boat`).
- Line format `class cx cy w h`, all normalized to [0,1] by chip width/height — **identical** to
  what `BoatDetection/scripts/helpers/PascalToYolo.py:43-47` produces, so it drops straight in.
- A reviewed chip with no surviving boxes → an **empty** `.txt` (a hard negative). Keep it.

**Wire into BoatDetection** (`C:\gtest\BoatDetection`, the sibling repo — its model contract is in
project memory):

1. Point `BoatDetection/scripts/helpers/splitData.py` at `training_exports/<date>/` (it does the
   70/20/10 split into `images/{train,val,test}` + `labels/...`). Optionally *merge* the satellite
   exports with the original photo dataset, or train satellite-only once enough chips accumulate.
2. **Fine-tune from the current weights, not from scratch:** in `BoatDetection/scripts/train.py`
   change `YOLO("yolov8n.pt")` → `YOLO(".../best.pt")` so each round builds on the last. Keep
   `imgsz=640` (matches the chips exactly).
3. New `best.pt` → point stage 2's `--weights` at it → rescan → review again. Each loop the
   detections get better and the reviewer's job gets lighter. That convergence *is* the product.

**Housekeeping blocker (from memory):** `BoatDetection/data/config.yml`, `train.py`, and
`splitData.py` hardcode absolute paths to another machine
(`C:\Users\bgern\Desktop\AIEng\BoatDetection\...`). Fix those to the local repo before retraining;
inference (stage 2) only needs `best.pt` and is unaffected.

---

## Data contracts

### `manifest.json` — stage 1 → 2 (EXISTS)

Written by `sat_fetch.py`. Top level: `generated_utc`, `provider`, `collection`,
`aoi_bbox_wgs84`, `wdpa` (`{WDPAID, WDPA_PID, NAME}` or `null` for bbox runs), `scene_crs`,
`tile_size`, `overlap`, `resolution_m`, `bands`, `stretch`, `known_limitations`,
`selection` (`{max_cloud, max_search, min_coverage, max_aoi_cloud, fallback_used}` — the AOI-aware
thresholds used; **`fallback_used: true` means no scene met them and a best-effort scene was
accepted, so expect cloud/partial coverage**), `chosen_item`
(`{id, datetime, eo:cloud_cover, aoi_coverage, aoi_cloud}` — `aoi_cloud` is cloud/shadow fraction
*over the AOI* from SCL, or `null` if the collection has no SCL), `candidates[]` (probed records,
same per-item shape; only those probed before early-stop selection appear), and `chips[]`. Each chip:

```json
{ "filename": "chips/chip_r00000_c00000.tif", "item_id": "S2C_...",
  "pixel_window": {"row_off": 0, "col_off": 0, "width": 640, "height": 640},
  "utm_bounds": [357580.0, 130700.0, 363980.0, 137100.0],
  "wgs84_bounds": [103.7199167, 1.1821852, 103.7774574, 1.2401006],
  "valid_frac": 1.0 }
```

### `detections.json` — stage 2 → 3 (NEW)

```json
{
  "generated_utc": "2026-07-07T12:00:00Z",
  "run_dir": "2026-07-07T11-19-33Z_bbox",
  "weights": ".../best.pt",
  "model": "yolov8n",
  "conf_threshold": 0.25,
  "iou_threshold": 0.5,
  "scene": { "item_id": "S2C_...", "datetime": "2026-06-13T03:15:41Z", "cloud": 15.98 },
  "wdpa": { "WDPAID": 555622064, "NAME": "Seal Rocks" },
  "count": 3,
  "detections": [
    {
      "detection_id": "det_00001",
      "chip": "chips/chip_r00000_c00000.tif",
      "class": "boat",
      "confidence": 0.83,
      "bbox_pixel": [412, 155, 431, 178],
      "centroid_wgs84": [103.7461, 1.2033],
      "bbox_wgs84": [103.7459, 1.2031, 103.7463, 1.2035],
      "inside_mpa": true,
      "review_status": "unreviewed"
    }
  ]
}
```

`review_status` starts `"unreviewed"`; stage 3 rewrites it (or, cleaner, records verdicts in a
separate `reviews.json` keyed by `detection_id` so `detections.json` stays an immutable model
output). `inside_mpa` is `null` for bbox-only runs.

### `reviews.json` — stage 3 human verdicts (NEW)

Keyed by chip so the exporter can emit one complete label file per chip:

```json
{
  "reviewed_by": "grant",
  "chips": {
    "chips/chip_r00000_c00000.tif": {
      "reviewed": true,
      "boxes": [
        { "bbox_pixel": [412,155,431,178], "source": "detection", "detection_id": "det_00001", "verdict": "correct" },
        { "bbox_pixel": [absent for FP],   "source": "detection", "detection_id": "det_00002", "verdict": "incorrect" },
        { "bbox_pixel": [88,510,109,533],  "source": "human",      "verdict": "missed" }
      ]
    }
  }
}
```

Export rule: for each `reviewed: true` chip, write a label line for every box with
`verdict ∈ {correct, missed}`; drop `incorrect`; if none remain, write an empty `.txt` (negative).

### YOLO label — stage 3 → retrain (NEW)

`labels/<run>__<chip>.txt`, one line per kept box, normalized to chip size:

```
0 0.658 0.260 0.030 0.036
```

Matches `BoatDetection`'s existing `PascalToYolo.py` output exactly.

---

## New files, dependencies, gitignore

| New file | Stage | Purpose | New deps |
|---|---|---|---|
| `scripts/detect_boats.py` | 2 | YOLO over chips → `detections.json` (+geojson) | `ultralytics` (pulls `torch`) |
| `scripts/review_server.py` | 3 | Local review UI; writes `reviews.json` | `fastapi`+`uvicorn` (or `flask`), `pillow` |
| `scripts/export_labels.py` | 3 | Reviewed chips → YOLO dataset | none (stdlib + rasterio/PIL) |
| `scripts/scan_mpas.py` | 1+ | Batch driver: all MPAs → many run dirs | none (imports `sat_fetch`) |

`shapely` (point-in-MPA) and `geopandas` are already installed. Pin the new deps in
`requirements.txt` with `==` **after** verifying in the venv, per the repo's "pin once verified"
convention. `torch` is large — consider a separate `requirements-ml.txt` so the lightweight
fetch/CI path (`gfw.yml`) doesn't drag in the ML stack.

**`.gitignore`** — `data/raw/sentinel2/` is already ignored (covers `detections.json` /
`reviews.json`, which live inside run dirs). Add:

```
# Human-labeled training exports produced by scripts/export_labels.py
data/processed/training_exports/
```

(`data/processed/wdpa_marine_ia_ib.gpkg` stays committed; only the exports subdir is ignored.)

---

## Build order / checklist

Build against **one** stage-1 run dir end-to-end before scaling out.

- [x] **2a–2c. Detector** — `detect_boats.py` built & verified: per-chip inference, pixel→lat/lon,
      centroid-metre dedup (312→122 on Singapore, all centroids inside AOI), point-in-MPA tag,
      atomic `detections.json`/`.geojson`.
- [x] **3a. Review server** — `review_server.py` built & verified: chips-as-PNG, canvas overlay,
      ✓/✗ + drawn boxes, "mark reviewed" gate, atomic `reviews.json`; security controls tested.
- [x] **3b. Export** — `export_labels.py` built & verified: reviewed chips → `images/` + `labels/`
      (incl. empty-file hard negatives); label math matched a hand-computed value.
- [ ] **retrain** — fix BoatDetection's hardcoded paths (see `docs/AUDIT.md` / model memory);
      `splitData.py` on an export; fine-tune `train.py` from `best.pt`; rescan with the new weights;
      confirm detections improved. *(next real step to close the loop)*
- [ ] **1-batch (later)** — `scan_mpas.py` to sweep all Ia/Ib marine MPAs (≤1° sub-AOIs). Needs the
      `main()`-guard refactor from `docs/AUDIT.md` P1.1 so `sat_fetch` functions are importable.
- [ ] **violation events (later)** — join high-confidence in-MPA detections to the
      `violation_events` schema in `CLAUDE.md`; fuse with GFW fishing-effort + AIS for scoring.

---

## Cross-cutting concerns

- **Domain gap (read this before trusting round 1).** `best.pt` was trained on the Kaggle "Boats
  Boats Boats" set — oblique/side **photographs**, not top-down 10 m imagery. Expect poor
  first-pass recall/precision on satellite chips. That is not a bug; it's *why* stage 3 exists —
  the review UI doubles as a bootstrap labeler to build an overhead-domain dataset. Treat round-1
  detections as *candidates to label*, not answers. (Full note in project memory.)
- **10 m GSD limit.** A 30 m boat ≈ 3 px; sub-~15 m vessels are near-invisible. Sentinel-2 finds
  larger trawlers/cargo. The genuine "fine" pass is a different sensor — **Sentinel-1 SAR**
  (all-weather, detects dark/AIS-off vessels) — addable behind stage 1's existing
  `open_catalog`/`search_items`/`read_aoi` seam. Recorded in the manifest's `known_limitations`.
- **TLS on this machine (Avast).** Every HTTPS/COG read is MITM'd. Stage 1 needs the
  certifi+Avast CA bundle env vars and `--insecure-tls` for GDAL reads (see `CLAUDE.md` + memory).
  Stage 2's `ultralytics` weight load is local (`best.pt` on disk) so it's unaffected; if you ever
  auto-download base weights, that download hits the same wall.
- **Stretch consistency.** Chips are stretched to 8-bit with the `stretch` params recorded in the
  manifest. Whatever normalization the model was trained under must match — when you retrain on
  chips, the training images *are* post-stretch chips, so the loop is self-consistent as long as
  `--stretch*` is held fixed across a scan→train→rescan cycle. Don't change it mid-loop.
- **Reproducibility.** Keep the house conventions in every new script: `SCRIPT_DIR`/`REPO_ROOT`
  path constants, `env_or` CLI>env>default fallbacks, atomic `.tmp` + `Path.replace()` writes,
  UTC-timestamped names, `print()` logging, fail-fast `RuntimeError`. One script per data stream;
  don't mix concerns.
```
