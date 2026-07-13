# ROADMAP — improving the detector & building the MPA-violation product

_Snapshot: 2026-07-11. Durable handoff for Phases 0(remainder)–3. Read `docs/STATUS.md` first
(live model, per-run eval numbers, open problems), then this. Original research/rationale:
`~/.claude/plans/what-will-actually-improve-swirling-truffle.md` (user machine, not in repo)._

## Where we are

The detector (`data/training/weights/best.pt`, YOLOv8n-**P2** on Sentinel-2 10 m RGB) is the
**`finetune-openocean`** model, deployed at **imgsz 1024, conf 0.15**. Cross-validation exposed that
the Singapore-only P2 model had **~0 recall on fresh open-ocean scenes** (Port Said held-out: 0.00);
folding in real human-reviewed open-ocean data (Fujairah 131 + Gibraltar 29 boxes) lifted held-out
open-ocean recall to **0.71 @ conf 0.15** with no forgetting (Singapore held, base mAP50 up). The
dominant early lever was **inference resolution** (1–5 px → 2–10 px), then the **P2 stride-4 head
stacked**, then **real open-ocean training data** proved the highest-leverage remaining move. **All of
Phase 0 is done.** The active-learning data loop is now the proven engine; next is scaling its
diversity (small fishing vessels) plus the imagery/AIS phases below.

**The remaining ceiling is physical:** at 10 m GSD a 20 m boat ≈ 2 px, near the optical information
floor. Free model tricks are largely spent; the next real gains come from **better imagery** (Phases
1–2) and from **reframing the goal as AIS-mismatch detection** (Phase 3), which is the actual product.

### Target architecture — how the phases combine
```
DETECT                          CORRELATE                  CONFIRM
S2 optical (P0, live)     ┐
PlanetScope 3 m (P1)      ├─► vs AIS/GFW (P3) ─► dark? ─► tip-and-cue hi-res (P3, paid, per-alert)
Sentinel-1 SAR (P2)       ┘         │
                                    └─► violation_events (schema in CLAUDE.md)
```
Each phase ships value alone. **Recommended order (P0-remainder now done): P3 (or P1) → P2.** P3 is the
highest strategic value and works even with a modest-recall detector; P1 sees more boats; P2 adds
all-weather/night/dark coverage.

---

## Cross-cutting prerequisite (do before P1/P2): AUDIT P1.1 — the `main()`-guard refactor

`sat_fetch.py` runs its orchestration at **module top-level (from line 409)** — `parse_args()` and a
full fetch execute on *import*. The seam functions we need to reuse for a new imagery source —
`open_catalog` (:198), `search_candidates` (:204), `assess_aoi` (:249), `select_scene` (:280),
`read_aoi` (:305), `scale_to_uint8` (:333) — are therefore **not importable**. Any Phase 1/2
connector must first wrap lines 409-end in `def main():` under `if __name__ == "__main__":`, leaving
the helpers importable. This is AUDIT.md item #1 ("highest-leverage change for stages 2–3"); it
unblocks both imagery phases and unit testing. **Effort: low. Do it first.**

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

## Phase 1 — PlanetScope ~3 m imagery (free for research)

**Goal:** biggest per-pixel gain for sub-20 m boats — at 3 m a 20 m boat is ~7 px (vs ~2 px at 10 m).
**Not** a `--collection` flip: PlanetScope is **not on Planetary Computer**; it needs Planet's own API +
a research account, plus generalizing the S2-specific assumptions that leak past the seam.

**Prerequisites:** the P1.1 main-guard refactor (above); a Planet Education & Research account
(≤3,000 km²/mo free — https://www.planet.com/industries/education-and-research/); Planet API key in
`.env` (mirror the `GFW_API_TOKEN` / `SAT_*` env pattern).

**Steps:**
1. **New connector behind the existing seam names.** Add a Planet implementation of `open_catalog` /
   `search_candidates` / `select_scene` / `read_aoi` (Planet Orders/Data API + `planet` SDK, or STAC if
   using Sentinel Hub's Planet integration). Select the source with a `--provider {pc,planet}` flag.
2. **Generalize the S2 leaks past the seam:**
   - **Cloud gate:** `assess_aoi` uses the Sentinel-2 **SCL** codes (`SCL_OBSCURED = (1,3,8,9,10)`, :245).
     PlanetScope ships its own **UDM2** usable-data mask — swap the mask source per provider.
   - **STAC prefilter:** `search_candidates` queries `eo:cloud_cover` (:209) — Planet exposes
     `cloud_percent`; abstract the field name.
   - **Reflectance stretch:** `scale_to_uint8` assumes S2 surface reflectance (`stretch_max≈3000`, :333–340).
     PlanetScope SR has its own scaling — parameterize per provider so 8-bit RGB matches training preproc.
3. **Train a separate 3 m model** (different GSD → keep the S2 model separate; a fresh fine-tune, its own
   `best.pt`). Chips are still 640 px → the imgsz-1024 lesson likely still applies; re-sweep conf.

**Data contract:** unchanged — output is the same run dir (`chips/*.tif` self-describing + `manifest.json`),
so `detect_boats.py` / `review_server.py` / `export_labels.py` work as-is.
**Verification:** `--dry-run` the Planet source over a known AOI (candidate table) → fetch → eyeball chips
→ `detect_boats.py` → confirm the 3 m model beats S2 on the same-area targets.
**Effort: medium.** Sources: Kanjir survey (min detectable ~20–30 m @10 m), Planet E&R.

---

## Phase 2 — Sentinel-1 SAR (free, complementary modality)

**Goal:** all-weather, day/night, **dark metallic-hull** detection — the wide-area workhorse GFW/Skylight
use (ESA/Paolo 2024: ~75% of industrial fishing vessels untracked). SAR floor ~15–20 m, so it
*complements* optical (catches larger/metallic vessels in cloud/night), not replaces it.

**Prerequisites:** P1.1 refactor. **Sentinel-1 GRD IS on Planetary Computer** (`sentinel-1-grd`), so the
STAC/catalog side is close to a `--collection` change — but the sensor differs fundamentally.

**Steps:**
1. **Drop the cloud logic:** SAR has no cloud concept — bypass the `eo:cloud_cover` prefilter and the SCL
   gate (`assess_aoi` already has a graceful no-SCL fallback at ~:270, coverage-only from band nodata).
2. **Backscatter, not reflectance:** replace `scale_to_uint8`'s reflectance stretch with **dB
   normalization** of VV/VH backscatter; band model is **VV/VH (2-band)**, not RGB — adjust `read_aoi`
   band handling and the chip writer.
3. **Train a separate SAR detector.** Consider transfer from public SAR ship datasets —
   **xView3-SAR** (arXiv 2206.00897), HRSID, SAR-Ship — rather than the optical base weights.

**Data contract:** same run-dir shape; chips carry CRS+affine so geolocation is unchanged. Detections
feed the same Phase 3 fusion.
**Verification:** `--dry-run` `sentinel-1-grd` over an AOI → fetch a VV/VH scene → eyeball → run the SAR
model; confirm it flags vessels optical missed under cloud/night.
**Effort: medium-hard** (new normalization + band model + a separate model).

---

## Phase 3 — AIS dark-vessel reframe (the operational payoff, greenfield)

**Goal / the product:** **dark-vessel = detection − AIS**. An in-MPA detection with no matching AIS
track = a candidate violation. This converts "impossible perfect recall" into achievable "mismatch
precision" — and it *filters the detector's false positives* (the open-water FP problem from conf-0.20),
so it makes the current model good enough. This is the project's stated purpose.

**The detection side is already join-ready.** `detect_boats.py` emits, per detection: `confidence`,
`centroid_wgs84 [lon,lat]`, `bbox_wgs84`, `inside_mpa`, `detection_id`, `chip`, plus a `scene.datetime`
in `detections.json` (:122–:123, :206, :229). The fusion layer is greenfield.

**Steps:**
1. **New `scripts/fuse_violations.py`** — join `detections.json` against per-vessel AIS positions within a
   spatiotemporal tolerance. Must be **velocity-aware**: the SAR/optical acquisition time and the nearest
   AIS ping differ, and a moving vessel travels between them — match within a distance that scales with
   `Δt × plausible_speed`, not a fixed radius. Unmatched in-MPA detections → the **`violation_events`**
   schema already specified in `CLAUDE.md` (vessel/mpa/times/scores/evidence).
2. **AIS source** (pick fastest-to-value first):
   - **Skylight (Ai2)** — a free, ready-made S1+S2+VIIRS **dark-vessel alert feed** that explicitly
     supports MPAs (https://skylight.global/platform). Consume it first for validation / tip-and-cue
     before building the matcher — fastest path to a working product.
   - **GFW Vessels / Insights API** (`/v3/vessels`, `/v3/insights/vessels`) for per-vessel identity + AIS-off
     events. NOTE: current `gfw_fetch.py` only calls the **coarse gridded** `/v3/4wings/report`
     (group-by `VESSEL_ID`) — that's effort density, **not** per-vessel positions; a new per-vessel call
     is needed.
   - Or a raw AIS feed (AISStream / Spire) for position-level tracks.
3. **Tip-and-cue hook:** this layer is where a confirmed dark alert would trigger **paid sub-meter**
   imagery (WorldView/SkySat/Pléiades/ICEYE/Capella, ~$12–60/km²) — per-alert only, never blanket
   (blanketing a 10,000 km² MPA is ~$150k–600k/pass). Attach here, don't build wide-area.

**Verification:** seed one detection known to match an AIS track (must NOT flag) and one known dark
(must flag); validate against Skylight's feed for the same AOI/time.
**Effort: medium-hard (greenfield).**

---

## Quick reference — commands & files

| Task | Command / file |
|---|---|
| Evaluate candidate @1024 | `eval_holdout.py --old .../best.pt --old-imgsz 1024 --new <run>/best.pt --new-imgsz 1024` |
| Pick operating point | `conf_sweep.py --weights <run>/best.pt --imgsz 1024` |
| Re-apply scene holdout after `build_dataset.py` | snippet in `docs/STATUS.md` (Operational recipes) |
| Promote | `cp <run>/weights/best.pt data/training/weights/best.pt` (back up first) |
| Detector deploy defaults | `detect_boats.py`: imgsz 1024, conf 0.20, weights → in-repo `best.pt` |
| Imagery seams to reuse | `sat_fetch.py`: `open_catalog`/`search_candidates`/`assess_aoi`/`select_scene`/`read_aoi`/`scale_to_uint8` |
| Refactor prerequisite | wrap `sat_fetch.py` :409-end in `main()` (AUDIT.md #1) |
| Violation schema | `CLAUDE.md` → "Output Schema" / `violation_events` |

## Key sources
Small-object: SAHI 2202.06934, copy-paste 1902.07296, NWD 2110.13389, P2/SOD-YOLOv8 2408.04786.
Imagery: Kanjir survey (PMC5877374), S2-YOLO RSE 2025, Planet E&R. Dark vessels: ESA/GFW (75% untracked),
Skylight (skylight.global/platform), xView3-SAR 2206.00897.
