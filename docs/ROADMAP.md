# ROADMAP — improving the detector & building the MPA-violation product

_Snapshot: **2026-08-01** (verified against code, not memory). The forward plan. Written to be read
**cold**, with no prior conversation context._

**Read in this order:**
1. **`docs/STATUS.md`** — live model, per-run eval numbers, model progression, open problems.
2. **this file** — what is built, what is next, and the exact command + acceptance test for each.
3. **`docs/IMPROVEMENT_PLAN.md`** — the accuracy levers (optical O1–O7, SAR S1–S7) in detail.
4. **`docs/PHASE3_PLAN.md`** — the AIS-fusion product checklist (its definition of done is **met**).
5. **`docs/AUDIT.md`** — code-health findings (all P1/P2 closed).
6. **`CLAUDE.md`** — per-script contracts, data layout, house rules, first-run setup.

_Original research/rationale: `~/.claude/plans/what-will-actually-improve-swirling-truffle.md` (user
machine, not in repo)._

## Where we are

| | Current state |
|---|---|
| **Optical detector** | `data/training/weights/best.pt` = **`finetune-diverse`** (promoted 2026-07-26). YOLOv8n-**P2** on Sentinel-2 10 m RGB. **Deploy: `imgsz 1024`, `conf 0.25`** — baked into `detect_boats.py` defaults. |
| **SAR detector** | `data/training/weights/best_sar.pt` = **`sar-deep`** (promoted 2026-07-26). Thin first model (206 boxes / 1 region-family). **`sar-deep2` was NOT promoted** — it regressed the Gibraltar gate. |
| **Phase 3 (AIS fusion)** | **All stages built, offline-tested, AND live-validated** (2026-07-26). Definition of done **met**. |
| **Offline test suite** | **203 tests**, pure stdlib, no network / keys / ML deps: `python -m unittest discover -s scripts/tests` |
| **Code health** | All `docs/AUDIT.md` P1/P2 findings closed. The P3 items are conscious house-style deferrals, not debt. |

**Four holdout scenes anchor every optical gate.** These are the numbers a candidate model must beat
(all at `imgsz 1024 / conf 0.25`, from the `finetune-diverse` promotion eval, 2026-07-26):

| Holdout | Type | Boxes | P | R | Note |
|---|---|---|---|---|---|
| **Kornati** (Adriatic) | cross-region small-vessel | 156 | 0.57 | **0.51** | ⬅ **the weakest link — open gap #1** |
| Alonnisos (Aegean) | cross-type small-vessel | 76 | — | 0.80 | relabeled 2026-07-16 → honest baseline |
| Port Said | big-ship dense anchorage | 42 | — | 0.79 | `--imgsz 1280 --augment` profile reaches 0.81 |
| Singapore Jul-5 | cross-date | 26 | — | 0.77 | the original gate |

### The story so far — why the levers are what they are

The dominant early lever was **inference resolution** (imgsz 640→1024 turned 1–5 px specks into
2–10 px). Then the **P2 stride-4 head stacked** on top of that. Then **real, diverse training data**
proved the highest-leverage remaining move — **four times running**: cross-*region* (open-ocean
Fujairah+Gibraltar lifted Port Said 0.00→0.71), cross-*type* (small-vessel Med data lifted Alonnisos
0.31→0.67), big-ship (the Gulf-of-Finland S2 set), and broad diversity (the 13-set label harvest).
**Phase 0 is complete and the active-learning loop is the proven engine — when in doubt, feed it more
diverse real data.**

**The remaining optical ceiling is physical.** At 10 m GSD a 20 m boat is ~2 px, near the information
floor, and there is **no free global higher-resolution optical source**. The cheap model/inference
tricks are spent and measured (see "Do not re-attempt" below). So the real remaining gains come from
three places: **more diverse real training data** (the engine), **free Sentinel-1 SAR** (all-weather,
night, dark metallic hulls), and **the AIS-mismatch reframe** (Phase 3) — which is the actual product,
and is already built and live-validated.

### Target architecture — how the phases combine
```
DETECT                       CORRELATE                  CONFIRM
S2 optical  (P0, live)     ┐
Sentinel-1 SAR (P2, live)  ├─► vs AIS/GFW (P3) ─► dark? ─► S1×S2 cross-check + human review (free)
                           ┘         │
                                     └─► violation_events ──► report_violations (record)
                                                          └─► alert_violations  (what needs attention)
```
All three phases are now built and each ships value alone. Both free imagery providers (`pc` =
Sentinel-2 optical, `s1` = Sentinel-1 SAR) live behind the `sat_fetch.py` **Provider seam**, so adding
a source is a subclass + one `register()` line.

---

# THE WORK QUEUE — what to do next

_This is the actionable part of the file — everything after the **HISTORICAL RECORD** banner further
down is reference only. Each item states **why**, the **exact commands**, the **acceptance test**, and
the **traps**._

**Legend:** 🏠 = needs the user's machine (GPU / human review / credentials) · 🟢 = buildable in any
sandbox, no credentials · ⛔ = blocked on an external party.

**Status in one line: the buildable backlog is nearly empty.** All AUDIT P1/P2 findings are closed,
Phase 3's definition of done is met, and the three offline product stages (`ais_fishing.py`,
`geofence_ais.py`, `alert_violations.py`) landed 2026-07-29. What remains is mostly **data work** that
needs a GPU and a human labelling chips — the loop is proven, it just has to be turned.

| # | Item | Attacks | Effort | Blocker |
|---|---|---|---|---|
| **W1** | Adriatic optical scenes → retrain → re-gate | recall (Kornati gap) | M (~1.5 h/cycle) | 🏠 fetch + review + GPU |
| **W2** | SAR region diversity **or** xView3 transfer | SAR generalization | M–L | 🏠 review + GPU (xView3 also gated download) |
| **W3** | Live operational loop over a trafficked MPA | product proof | S–M | 🏠 `AISSTREAM_API_KEY` |
| **W4** | O3 same-region hard negatives | precision | M | 🏠 — **and do W1 first** |
| **W5** | `eval_holdout.py --train-dir` (SAR gating harness) | trust / unblocks W2 | S | 🟢 buildable now |
| **W6** | S6 speckle-filter A/B | SAR precision/recall | S | 🏠 GPU |
| **W7** | S7 cross-modality S1×S2 check | trust | S–M | 🏠 paired scenes |
| **W8** | Websocket reconnect for `aisstream_fetch.py` | W3 robustness | S | 🟢 — but decide semantics first |
| **W9** | Skylight dark-call cross-check | external validation | M | ⛔ API registration |

---

## W1 — Close the Kornati gap (Adriatic optical data) 🏠 **← start here**

**Why.** Kornati is the sharpest number in the repo: **R 0.51 / P 0.57** versus 0.80 on Alonnisos.
It is a *cross-region* small-vessel holdout and the model has **zero Adriatic/karst-coast data in
TRAIN by design**. This is the same gap the loop has closed four times before, so the fix is known
and the engine is proven. Highest impact per hour of any open item.

**Pick 2–3 scenes of the same class, but NOT Kornati** — Kornati must stay a pure holdout or the
gate becomes meaningless. Good candidates: **Croatian islands** (Šibenik/Zadar archipelago, Hvar,
Korčula, Mljet), **Greek Ionian** (Corfu, Lefkada, Zakynthos), **Turkish Aegean/Med coast** (Bodrum,
Göcek, Fethiye). What matters is the *class*: karst coastline, many small islets, dense small-craft
and day-boat traffic, clear shallow water.

```bash
# 0. TLS: under Avast interception, export the CA bundle vars first (see docs/STATUS.md recipe)

# 1. Preview availability before spending a fetch (ranked by cloud OVER the AOI)
python scripts/sat_fetch.py --bbox <minLon> <minLat> <maxLon> <maxLat> --dry-run
# 2. Fetch
python scripts/sat_fetch.py --bbox <...> --min-coverage 0.6 --label <slug> --insecure-tls
# 3. Pre-detect so review has candidate boxes to correct (deploy point is the default)
python scripts/detect_boats.py --run data/raw/sentinel2/<run>/
# 4. Human review — the actual bottleneck, ~40 min/scene
python scripts/review_server.py --run data/raw/sentinel2/<run>/
# 5. Export → assemble → train (Kornati STAYS held out)
python scripts/export_labels.py --run data/raw/sentinel2/<run>/
python scripts/build_dataset.py \
    --holdout-scene fallbacktest portsaid alonnisos kornati-adriatic --max-neg-ratio 1.0
python scripts/train.py --weights data/training/yolov8n-p2.yaml --pretrained best \
    --imgsz 1024 --batch 4 --mosaic 0.5 --close-mosaic 15 --epochs 30 --name finetune-adriatic
# 6. Gate on all four holdouts, then pick the operating point
python scripts/eval_holdout.py --old data/training/weights/best.pt --old-imgsz 1024 \
    --new data/training/runs/finetune-adriatic/weights/best.pt --new-imgsz 1024
python scripts/conf_sweep.py --weights data/training/runs/finetune-adriatic/weights/best.pt --imgsz 1024
# 7. Promote ONLY if the gate passes (back up the current best.pt first!)
cp data/training/weights/best.pt data/training/weights/best_finetune-diverse.pt
cp data/training/runs/finetune-adriatic/weights/best.pt data/training/weights/best.pt
```

**Acceptance:** Kornati recall clears **0.51** with its 90% bootstrap CI separated from the current
one, and **no other holdout regresses** (Alonnisos 0.80 / Port Said 0.79 / Singapore 0.77).

**Traps:**
- **`--batch 4`, not 16.** Batch 16 at imgsz 1024 with the P2 head exhausts memory and dies as a
  *silent* torch C++ abort mid-epoch. This has bitten before.
- **Kornati precision is the standing watch item** (FP/chip 3.81 @0.25). If precision craters while
  recall climbs, that is the W4 (hard negatives) signal, not a reason to reject the model.
- **Keep `--max-neg-ratio 1.0`.** This guards the `finetune-reef` counter-lesson (see below).
- **Deploy conf has moved +0.05 at each of the last two promotions** (0.15→0.20→0.25) as recall
  improved. Let `conf_sweep.py` decide; do not assume 0.25 carries over.
- Re-apply the scene-aware holdout after `build_dataset.py` if you bypass `--holdout-scene`
  (snippet in `docs/STATUS.md` → Operational recipes).

---

## W2 — SAR needs *regions*, not one more scene 🏠

**Why.** `best_sar.pt` (`sar-deep`) is trained on 206 boxes from one region-family (Fujairah /
Gibraltar / Hormuz — all warm, deep, Gulf-like). The diagnostic that matters: **it scored 0 hits at
conf 0.15 on a fresh Singapore Strait scene.** Because `providers/s1.py` applies a **per-scene dB
percentile stretch**, every new region is literally a different pixel distribution — so a thin model
is out-of-domain immediately.

**The already-tried path that failed:** adding the Singapore scene alone (`sar-deep2`, +267 boxes)
**regressed the Gibraltar gate** — P 0.392→0.285, R 0.478→0.435, mAP50 0.335→0.298 — and was not
promoted. Caveat: that gate is thin (23 test boxes), so the noise is large. Singapore data remains
staged in `data/training_sar/` TRAIN/VAL for the next attempt. **The conclusion is that one more
scene will not fix this.** Two real options:

**Option A — more regions (incremental, free).** Add **2+ more deep-water regions** before the next
gate attempt. Seed the review with the classical detector, since `best_sar.pt` will score ~0 on
anything unseen:
```bash
python scripts/sat_fetch.py --provider s1 --bbox <...> --label s1deep-<slug> --dry-run
python scripts/sat_fetch.py --provider s1 --bbox <...> --label s1deep-<slug>   # → data/raw/sentinel1/
# Optional but strongly recommended for review: a co-registered OPTICAL pane so land is unambiguous
# under speckle. Needs a separate S2 fetch over the same AOI (any date — land is static).
python scripts/sat_fetch.py --bbox <same AOI> --label <slug>-optical
python scripts/sar_companion.py --sar-run data/raw/sentinel1/<run>/ \
                                --optical-run data/raw/sentinel2/<s2run>/
# --water-only reads a CACHED mask, so build it first (ports/buildings are very bright in SAR)
python scripts/water_mask.py --run data/raw/sentinel1/<run>/
python scripts/sar_seed.py --run data/raw/sentinel1/<run>/ --water-only --min-depth 12
python scripts/review_server.py --run data/raw/sentinel1/<run>/   # shows the companion pane if present
python scripts/export_labels.py --run data/raw/sentinel1/<run>/ \
    --out data/processed/sar_training_exports --tag <region-slug>   # --out is the ROOT, --tag the subdir
    #  ^^ NEVER data/processed/training_exports/ — build_dataset.py sweeps every tag there into the
    #     OPTICAL data/training/, which would silently poison the optical set with SAR chips.
python scripts/build_dataset.py --exports data/processed/sar_training_exports \
    --data data/training_sar --holdout-scene s1deep-gibraltar
python scripts/train.py --weights data/training/yolov8n-p2.yaml --pretrained yolov8n \
    --data data/training_sar/config.yml --imgsz 1024 --batch 4 \
    --mosaic 0.5 --close-mosaic 15 --epochs 30 --name sar-deep3
```
Target **deep water** (SAR's home turf): deep anchorages and lane chokepoints. Shallow water returns
strong bathymetric clutter — that is what `--min-depth` exists for.

**Option B — xView3-SAR transfer (the big lever).** arXiv 2206.00897: Sentinel-1 dual-pol, maritime,
dark-vessel labels with vessel length. Free but **registration-gated** at iuu.xview.us. The
**make-or-break step is preprocessing alignment (S2 in `IMPROVEMENT_PLAN.md`)**: run xView3's VV/VH
through `providers/s1.py::scale_to_uint8` — the *same function* the pipeline deploys — so training and
inference see the same distribution. Training on raw amplitude or any other stretch will fail.

**Acceptance:** the Gibraltar gate does not regress (P ≥ 0.39 / R ≥ 0.48 / mAP50 ≥ 0.335) **and** the
new region scores non-zero on its own held-out chips.

**Blocker to clear first: W5.** `eval_holdout.py` hardcodes `data/training/`, so it **cannot evaluate
the SAR set at all** — there is currently no equivalent tiny-object gate for SAR. Do W5 first or the
gate stays eyeball-grade.

**Never warm-start SAR from the optical `best.pt`** — wrong texture priors. Use `--pretrained yolov8n`
or a SAR-pretrained backbone.

---

## W3 — Run the live operational loop 🏠

**Why.** Every stage is live-validated *individually*, and the historical path produced real
violations (Anacapa: 2 dark, one a visually-confirmed wake vessel; SB-Channel lanes: 36 matched / 7
dark). What has never run is the **near-real-time** mode: a rolling AIS buffer over a trafficked MPA
fused against a fresh overpass.

**This got substantially cheaper on 2026-07-29.** `geofence_ais.py` + `ais_fishing.py` need **only an
AIS capture** — no imagery, no overpass timing, no GFW token. That is the fastest path to a live
result, so do it first:

```bash
# A. AIS-only product loop — needs just AISSTREAM_API_KEY. No imagery, no timing constraint.
python scripts/aisstream_fetch.py --wdpa-id <WDPAID> --duration 1800
python scripts/ais_fishing.py --ais data/raw/ais/<file>.json
python scripts/geofence_ais.py --ais data/raw/ais/<file>.json --wdpa-id <WDPAID> \
    --vessels data/raw/ais_fishing/<file>.json
python scripts/alert_violations.py --events data/raw/ais_geofence/<file>.json

# B. Full fusion — additionally needs a fresh overpass inside the AIS capture window
python scripts/scan_violations.py --run data/raw/sentinel2/<run>/ --ais data/raw/ais/<file>.json \
    --gfw --wdpa-id <WDPAID>
```

**The timing constraint is the whole difficulty of B.** Imagery is a *past* overpass; AISStream is
*live-only*. Fusion is valid only when `scene.datetime` falls inside the AIS capture window. So
either run `aisstream_fetch` as a rolling buffer and fuse when an overpass lands, or use the
historical path (`marinecadastre_fetch.py`, US waters only). Widen `--dt-minutes` only as far as
vessel speeds actually justify.

**Acceptance:** an alert digest over a real MPA where at least one event is independently
sanity-checkable (a vessel loitering in a no-take reserve, or a dark detection you can see on the chip).

**Trap:** pick an MPA with actual traffic. A strict IUCN Ia/Ib reserve is often empty water, which is
the point of the product but makes for a boring first demo. See W8 — a rolling buffer is exactly the
case where a dropped websocket costs you the overpass.

---

## W4 — O3 same-region hard negatives 🏠 *(do W1 first)*

**Why.** On-water false alarms on small-vessel scenes are the dominant precision problem (~84% of all
FPs per the Step-0 audit; land is only ~16%). The fix is reviewed empty-water / glint / swell /
reef-edge chips **from the deployment regions**, exported as hard negatives (empty labels).

**Sequence matters: W1 before W4.** New Adriatic positives change the FP mix, so negatives mined
before W1 would target a distribution that no longer exists. Current watch signals: Kornati FP/chip
**3.81** @0.25, False Bay open-water probe **7→34**, GBR probe **15→19**.

Same loop as W1, but the reviewed chips come out with **no boxes** (an empty label file = a hard
negative). Keep `--max-neg-ratio 1.0` and **re-check holdout recall after every retrain that adds
negatives** — that discipline matters far more than the specific ratio.

**Acceptance:** precision up on Kornati/Alonnisos with **recall held** on all four holdouts.

**Trap — the `finetune-reef` counter-lesson:** piling on hard negatives once made the detector
over-conservative and dropped *both* recall and base mAP. Negatives are a scalpel, not a hammer.

---

## W5 — Give `eval_holdout.py` a `--train-dir` 🟢 *(buildable now, unblocks W2)*

**Why.** `eval_holdout.py` hardcodes `TRAIN = REPO/"data"/"training"` (line ~35), so it can only ever
evaluate the **optical** set. The SAR model lives at `data/training_sar/`, which means **the SAR gate
has no tiny-object harness** — no center-distance P/R, no FP/chip, no bootstrap CI, no size strata.
The `sar-deep2` promote/reject decision was made on thinner evidence than any optical decision has been.

**Do:** add `--train-dir` (default `data/training`) to `eval_holdout.py`, and the same to
`conf_sweep.py` if it shares the assumption. Small, mechanical, no new deps. Running it needs
ultralytics + a GPU, but the change itself is sandbox-buildable.

**Acceptance:** `eval_holdout.py --train-dir data/training_sar --holdout s1deep-gibraltar` reproduces
the Gibraltar gate numbers with CIs.

---

## W6–W7 — SAR polish 🏠

- **W6 speckle A/B (S6):** try Lee / Refined-Lee before tiling and measure recall on 1–5 px targets
  **both ways**. Speckle filters can smooth away the very targets we want — test, do not assume.
- **W7 cross-modality S1×S2 (S7):** run `best_sar.pt` on S1 chips and compare against optical
  detections over the same AOI. Confirms SAR flags large/dark vessels optical missed under
  cloud/night — the entire justification for the modality. Paired scenes already exist on disk
  (Fujairah, Singapore, Cyclades), and the 2026-07-15 Fujairah validation already showed backscatter
  blobs coinciding with optical detections plus extras.

---

## W8 — Websocket reconnect for `aisstream_fetch.py` 🟢 *(decide semantics first)*

**Deliberately left open** during the AUDIT #4 retry round. Every other network call site got the
shared `scripts/_http.py` policy, but a websocket "retry" means **reconnect-and-resume mid-capture**,
which changes *capture semantics* — duplicate positions across the reconnect, partial windows, an
ambiguous `duration`. That is a design decision, not a transport setting, so it was left alone rather
than half-done.

**It is coupled to W3.** A rolling buffer over a fresh overpass is exactly the scenario where a
dropped connection costs you the acquisition. Decide before running W3 in earnest:
- **Dedup on reconnect?** (`(mmsi, timestamp)` is the natural key)
- **Does `--duration` mean wall-clock or connected-time?**
- **Should the output record connection gaps** so a downstream consumer knows the capture is holey?
  (`ais_fishing.py` already treats gaps as *unobserved* rather than idle — the same honesty property
  belongs here.)

---

## W9 — Skylight dark-call cross-check ⛔

Free ready-made ground truth for our dark calls (Ai2, skylight.global/platform). A sibling ingester
+ a comparison harness scoring our dark calls against Skylight's for the same AOI/time. **Gated on
API registration** — research access first. Not a pipeline dependency; a validation harness.

---

# DO NOT RE-ATTEMPT — measured dead ends

_Each of these was built or measured, cost real time, and did **not** work. They are recorded so a
future session does not rediscover them. Do not undo these conclusions without new evidence._

| Thing | Verdict | Why |
|---|---|---|
| **imgsz 1280 (plain, no TTA)** | ❌ non-win | Port Said identical; Alonnisos recall **0.70→0.49**. Upscaling past the 1024 *train* size loses the 1–5 px vessels it was meant to enlarge. 1536 not worth testing. The big-ship lift comes from **`--augment` TTA**, not raw imgsz. |
| **NDWI / NIR open-water filter for precision** | ❌ calibrated non-win | `ndwi_mask.py` + `--open-water` are built, correct, and **off by default**. The premise ("FPs sit on blank water") is FALSE: glint/foam/swell reflect NIR *more* than 1–5 px boats (FP min-NDWI median −0.122 vs TP −0.027). At −0.10 it drops 7/20 FP but **60/63 TP**. A water index cannot reject real spectral water *features*. |
| **Piling on hard negatives** (`finetune-reef`) | ⚠️ counter-lesson | Recall *and* base mAP both dropped — the detector went over-conservative. Hence `--max-neg-ratio`. Add **positives** to fix recall; use negatives surgically. |
| **Textbook CFAR / top-hat for SAR seeding** | ❌ fails by construction | `providers/s1.py`'s dB-percentile stretch is tuned for *display*: water sits mid-bright (median ~136/255) and vessels saturate at 255. CFAR assumes dark water. `sar_seed.py` instead thresholds the dual-pol geometric mean `sqrt(VV·VH)` at an adaptive `median + k·MAD` — robust and stretch-invariant. |
| **Warm-starting SAR from the optical `best.pt`** | ❌ wrong priors | Different sensor, different texture statistics. Use `--pretrained yolov8n` or a SAR-pretrained backbone. |
| **One more SAR scene to fix generalization** (`sar-deep2`) | ❌ regressed | +Singapore alone dropped the Gibraltar gate P 0.392→0.285. SAR needs *regions* (W2-A) or *transfer* (W2-B). |
| **`sentinel-1-grd` as the SAR collection** | ❌ incompatible | PC's GRD assets are in radar geometry: `crs=None`, identity transform, geolocation only via 189 GCPs → `transform_bounds` crashes. Use **`sentinel-1-rtc`** (real UTM grid, clean 10 m affine). GRD needs a GCP/warp reader first. |
| **Raising `--conf` to fix precision** | ❌ wrong tool | Trades away the recall the 10 m GSD already caps. The architecture is deliberately *recall-favouring candidate generation, then filter downstream* (water mask → AIS fusion). |
| **Paid/commercial imagery (Planet, VHR)** | ⛔ out of scope | Hard $0-budget constraint. Refs were removed deliberately; do not reintroduce. |
| **batch 16 @ imgsz 1024 with the P2 head** | 💥 crashes | Silent torch C++ abort mid-epoch. Use **`--batch 4`** (plateaus ~7 GB). |

---

# HISTORICAL RECORD — how each phase was built

_Everything below is **reference**, not a to-do list: the phase-by-phase build history, the seam
tables a new provider composes, and the reproduction detail for each promoted model. Forward-looking
work is in **THE WORK QUEUE** above._

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

## Phase 2 — Sentinel-1 SAR (free, complementary modality) — provider ✅, real-chip validation ✅, first detector ✅ (region generalization = **W2**)

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
4. ✅ **DONE 2026-07-26 — first `best_sar.pt` trained + promoted** (`runs/sar-deep`: P2 arch, yolov8n backbone, 30 epochs on the RTX 4070 in 76 s; honest test split incl. the reviewed Gibraltar holdout P 0.392 / R 0.478 / mAP50 0.335). Thin-data first model — growing the set + the xView3 transfer below remain the levers. *(original step text kept:)* Needs local GPU/compute with the machine kept
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

## Phase 3 — AIS dark-vessel reframe (the operational payoff) — ✅ **COMPLETE** (built + live-validated 2026-07-26; definition of done met)

**Goal / the product:** **dark-vessel = detection − AIS**. An in-MPA detection with no matching AIS
track = a candidate violation. This converts "impossible perfect recall" into achievable "mismatch
precision" — and it *filters the detector's false positives* (the open-water FP problem from conf-0.20),
so it makes the current model good enough. This is the project's stated purpose.

**Status: COMPLETE.** The pipeline was built + offline-tested 2026-07-23, **live-validated 2026-07-26**
(AISStream connected first try — 41/41 positions over Singapore; three GFW v3 corrections landed with
real captured-response fixtures; first real violations produced over Anacapa), and extended
2026-07-29 with three more offline stages. `docs/PHASE3_PLAN.md` holds the checklist — **every box is
ticked** except the optional Skylight cross-check (**W9**). Per-script contracts + first-run setup are
in `CLAUDE.md`. The only remaining Phase-3 work is **W3** (the near-real-time operational mode) and
**W8** (websocket reconnect, which W3 depends on).

Built — **nine** scripts, each its own module (ingestion never imports the matcher — house rule):
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
- **`ais_fishing.py`** (2026-07-29) — infers fishing from AIS **movement** (speed band + sinuosity) and
  emits the **same vessel-records contract** as `gfw_vessels.py` with `source: "ais-heuristic"`, so
  `--vessels` reads it unchanged. The fishing term in `violation_score` now works with **no GFW token
  and no network**. Thin evidence yields `null`, never `0.0`; AIS gaps are excluded from *both*
  numerator and denominator. Coarse by design — a prior to rank on, not proof of fishing.
- **`geofence_ais.py`** (2026-07-29) — flags cooperative vessels transmitting from **inside** the MPA,
  giving a third status beside the fusion's two: `dark` / `matched` / **`reported`**. Same
  `violation_events` schema. **Scored by behaviour, not presence** (transit floor 0.10 → fishing
  weight; IUU escalates), so legal transit does not drown the signal. Pure-stdlib ray-casting
  point-in-polygon (holes + concave shapes correct); only the WDPA *lookup* needs geopandas.
- **`alert_violations.py`** (2026-07-29) — the design proposal's promised **alerting interface**:
  severity digest (critical/high/medium/low; an **IUU listing forces critical**, an unscored event is
  never paged), MPA-grouped worst-first, console + markdown, `--exit-code` cron gate. Reads fusion
  **and** geofence events in one digest.

The `violation_events.json` schema is pinned in `docs/PIPELINE.md`. **203 offline tests** suite-wide
(synthetic fixtures + a committed end-to-end contract guard + real captured GFW responses; no network /
key / ML deps). ✅ **Live-validated 2026-07-26** — see `docs/PHASE3_PLAN.md` Workstream 1 for the three
GFW v3 corrections and the three matcher/ingest defects the first real fusion caught and fixed.

**Why it makes the modest detector good enough:** dark-vessel = detection − AIS *inverts the metric* — a
low-conf false positive sitting on a matching AIS track is filtered out, a real dark vessel with no AIS
survives. The product cares about *mismatch precision* (which fusion improves), not the raw *recall* the
10 m GSD caps. This is the payoff that justifies Phases 0–2.

---

## Completed-work index (what landed, and when)

_Forward-looking work lives in **THE WORK QUEUE** above — this section is the closed-items record, kept
so a future session can see what was already tried and does not redo it._

### 🏠 Do-from-home (need credentials / GPU — cannot run in the remote sandbox)
- ✅ **ALL CLOSED 2026-07-26** (single at-home session): AISStream + GFW live-verified, first real
  violations produced (Anacapa 2 dark; SB-Channel 36 matched/7 dark), `scan_violations.py` +
  `marinecadastre_fetch.py --date` run live, `best_sar.pt` trained + promoted (`sar-deep`), CUDA torch
  installed. Full detail: `docs/PHASE3_PLAN.md` + `docs/STATUS.md`. Note: the follow-up SAR retrain
  `sar-deep2` (+Singapore 267 boxes) was NOT promoted (Gibraltar gate regressed) — see STATUS.

### 🟢 Built offline in the sandbox (no credentials needed) — all closed
- ✅ **DONE (2026-07-29)** — **alerting interface**: `scripts/alert_violations.py`, the design
  proposal's final promised output. Severity-buckets `violation_events` (critical/high/medium/low by
  `violation_score`; an **IUU listing forces critical**, an unscored event is never paged), groups by
  MPA worst-first, and prints a console + optional markdown digest. Reads **fusion and geofence events
  together**, so all three statuses rank in one digest. `--exit-code` gives a cron/CI gate, off by
  default. Read-only — it computes no scores of its own. 29 offline tests.
- ✅ **DONE (2026-07-29)** — **AIS-track MPA geofencing**: `scripts/geofence_ais.py`. Flags cooperative
  vessels transmitting from inside a no-take MPA, giving a third status beside the fusion's two —
  `dark` / `matched` / **`reported`**. Emits the same `violation_events.json` schema, so
  `report_violations.py` ranks AIS-only events unchanged. **Scored by behaviour, not presence**
  (`--transit-weight` 0.10 → `--fishing-weight`, driven by a `--vessels` file from `ais_fishing.py` or
  `gfw_vessels.py`; IUU escalates), per the emit-and-rank house pattern. Point-in-polygon is **pure
  stdlib** ray casting (holes + concave shapes correct); only the WDPA lookup needs geopandas, and the
  `--bbox` path needs nothing. `duration_hours` is populated here — AIS gives multiple timestamps.
  Verified against the real Anacapa SMR polygon (1,291 vertices). 26 offline tests.
- ✅ **DONE (2026-07-29)** — **local speed-based fishing heuristic**: `scripts/ais_fishing.py`. Infers
  fishing-like behaviour from AIS movement (speed band + sinuosity) and emits the **same vessel-records
  contract** as `gfw_vessels.py` with `source: "ais-heuristic"`, so `fuse_violations.py --vessels` reads
  it unchanged — the fishing term in `violation_score` now works with no GFW token and no network.
  `fuse_violations.vessel_evidence_label()` tags such events `"AIS-heuristic"` rather than `"GFW"` (a
  provenance bug the old hardcoded label would have introduced). Thin evidence yields `null`, not `0.0`;
  AIS gaps are excluded from both numerator and denominator. 19 offline tests. Coarse by design — a
  prior to rank on, not proof of fishing.
- **Still open, now tracked in THE WORK QUEUE above:** cross-modality S1×S2 (**W7** / `IMPROVEMENT_PLAN.md`
  S7), Skylight cross-check (**W9** / `PHASE3_PLAN.md` 3B), the detector-accuracy levers **O3** →
  **W4** and **O4** → **W1**, the SAR gating harness (**W5**), and websocket reconnect (**W8**).
  (**O5** inference/tiling tuning and **O6** eval-harness hardening were **measured/exercised
  2026-07-25** and are closed — imgsz 1024 is confirmed and 1280 is a non-win; see "Do not re-attempt".)

### 🧹 Repo hygiene / reproducibility (`docs/AUDIT.md`) — all P1/P2 findings closed
- ✅ **DONE (2026-07-23)** — a cheap **CI job** (`.github/workflows/tests.yml`): `compileall` + the
  offline suite (now 203 tests) on every push/PR. (`gfw.yml` still only runs `gfw_fetch.py` on schedule.)
- ✅ **DONE (2026-07-23)** — committed **`.env.example`** (credentials + grouped per-script env-var legend)
  and **pinned `pandas`/`geodatasets`/`matplotlib`** in `requirements.txt` (resolver-verified).
- ✅ **DONE (2026-07-25)** — `main()`-guard + input/atomic-write hardening for `gfw_fetch.py` /
  `prep_polygons.py` / `view_polygons.py` (all wrap orchestration in `main(argv=None)`; fail-fast input
  validation; atomic `.tmp`→`replace` writes with cleanup; `gfw_fetch` now stdlib-importable + offline-tested).
- ✅ **DONE (2026-07-29)** — **network retry/backoff (AUDIT #4)**. One shared policy in
  `scripts/_http.py` (3 attempts, 0s/2s/4s backoff, 429+5xx and connect/read only; POST retryable,
  non-429 4xx never retried, `raise_on_status=False` so callers keep their existing error flow), wired
  into `gfw_fetch.py` (+ `--retries`/`--retry-backoff`), `gfw_vessels.py`, `marinecadastre_fetch.py`,
  `bathymetry.py` and `ingest_s2ships_finland.py`; `GDAL_HTTP_MAX_RETRY=3`/`RETRY_DELAY=2` added to
  `providers/_raster.py` `gdal_env()` for the `/vsicurl` COG reads. 16 new offline tests.
  **Still open:** `aisstream_fetch.py` is a websocket — reconnect-and-resume changes capture semantics
  (duplicate positions, partial windows), so it was left as a design decision, not half-done.
- ✅ **DONE (2026-07-29)** — **`data/raw/sentinel1/` output dir**. `Provider.output_subdir` (`sentinel2`
  for `pc`, `sentinel1` for `s1`) now picks the run root, so SAR and optical runs never share a
  directory. Purely additive: existing run dirs are untouched (downstream stages take an explicit
  `--run`), and `--output-dir`/`SAT_OUTPUT_DIR` still override.
- ✅ **DONE (2026-07-29)** — stale legacy-`BoatDetection` pointers removed from `export_labels.py`
  (header) and `detect_boats.py` (the weights-not-found error now names `data/training/weights/`).

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
| **Detector deploy defaults** | `detect_boats.py`: **imgsz 1024, conf 0.25**, weights → in-repo `best.pt` (`finetune-diverse`, promoted 2026-07-26). imgsz 1280 is a non-win — use `--imgsz 1280 --augment` only for the opt-in big-ship profile |
| Train the OPTICAL model | `train.py --weights data/training/yolov8n-p2.yaml --pretrained best --imgsz 1024 --batch 4 --mosaic 0.5 --close-mosaic 15 --epochs 30 --name <n>` (**`--batch 4`** — 16 crashes) |
| Train the SAR model | same, but `--pretrained yolov8n --data data/training_sar/config.yml` (**never** `--pretrained best` — wrong texture priors). Runs still land in `data/training/runs/<name>/` |
| Assemble the SAR dataset | `build_dataset.py --exports data/processed/sar_training_exports --data data/training_sar --holdout-scene s1deep-gibraltar` |
| Seed a NEW SAR region | `sar_seed.py --run <s1 run> --water-only --min-depth 12` — `best_sar.pt` scores ~0 on unseen regions (per-scene dB stretch), so seed classically |
| Add an imagery source | subclass `Provider` (`scripts/providers/base.py`), `register()` it (`providers/__init__.py`); copy `providers/pc.py` (optical) or `providers/s1.py` (SAR). Set `output_subdir` to the modality's run root under `data/raw/` |
| Select a source | `sat_fetch.py --provider {pc,s1}` (default `pc`; `SAT_PROVIDER` env). `pc`=S2 optical, `s1`=S1 SAR — both free, no credentials |
| Fetch a SAR scene | `sat_fetch.py --provider s1 --bbox … --dry-run` then without `--dry-run` (free, all-weather; needs its own `best_sar.pt`). Lands in `data/raw/sentinel1/` |
| Refactor prerequisite ✅ | `main()`-guard + provider dispatch both done (AUDIT.md #1) |
| Ingest live AIS (P3) ✅ | `aisstream_fetch.py --bbox … --duration 120` (or `--wdpa-id`) → `data/raw/ais/ais_positions_*.json`; needs `AISSTREAM_API_KEY`, `--dry-run` to test wiring |
| Fuse AIS → dark vessels (P3) ✅ | `fuse_violations.py --run <run> --ais data/raw/ais/<file>.json` → `violation_events.json` (+ dark `.geojson`) |
| GFW identity/fishing (P3) ✅ | `gfw_vessels.py --from-events <violation_events.json> --start … --end …` → `data/raw/gfw_vessels/*.json`; needs `GFW_API_TOKEN`, `--dry-run` to test wiring |
| Enrich matched events (P3) ✅ | re-run `fuse_violations.py … --vessels data/raw/gfw_vessels/<file>.json` (fills identity + fishing, lifts score for confirmed-fishing / IUU) |
| Historical AIS (P3) ✅ | `marinecadastre_fetch.py --csv AIS_YYYY_MM_DD.csv --wdpa-id <id> --start … --end …` → `data/raw/ais/…` (fuse past US-waters scenes) |
| Rank / report (P3) ✅ | `report_violations.py --run <run>` → ranked `violations_report_*.csv` + self-contained `.html` |
| Local fishing signal (P3) ✅ | `ais_fishing.py --ais <file>.json` → vessel records with `source: ais-heuristic`; feed to any `--vessels`. No key, no network |
| Geofence AIS × MPA (P3) ✅ | `geofence_ais.py --ais <file>.json --wdpa-id <id> --vessels <fishing>.json` → `"reported"` events (+ `.geojson`) |
| Alert digest (P3) ✅ | `alert_violations.py --run <run> --events <geofence>.json [--min-severity critical] [--out-markdown a.md] [--exit-code]` |
| One-shot driver (P3) ✅ | `scan_violations.py --run <run> --ais <file>.json [--gfw --wdpa-id <id>]` (fuse → enrich → report) |
| Test the P3 stages offline | `python -m unittest discover -s scripts/tests -p 'test_*.py'` (203 tests; synthetic, no network / key / ML deps) |
| Violation schema | `docs/PIPELINE.md` → `violation_events.json` (pinned) |

## Key sources
Small-object: SAHI 2202.06934, copy-paste 1902.07296, NWD 2110.13389, P2/SOD-YOLOv8 2408.04786.
Imagery: Kanjir survey (PMC5877374), S2-YOLO RSE 2025. Dark vessels: ESA/GFW (75% untracked),
Skylight (skylight.global/platform), xView3-SAR 2206.00897.
