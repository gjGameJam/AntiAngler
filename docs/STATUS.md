# STATUS — boat detector / active-learning loop

_Snapshot: 2026-07-17 (eval refresh 2026-07-25). Point-in-time; verify against code before trusting specifics._

This tracks the live state of the satellite ship-detection model and its active-learning
loop. For the build spec + data contracts see `PIPELINE.md`; for known antipatterns see
`AUDIT.md`; for the daily GFW/WDPA streams see `../CLAUDE.md`.

---

## TL;DR

- **Phase 3 (AIS dark-vessel fusion) — the whole sandbox-buildable pipeline is BUILT & offline-tested (2026-07-23).** Six scripts, each its own module (ingestion never imports the matcher): **`fuse_violations.py`** (the matcher — dark-vs-matched via a velocity-aware feasible-movement envelope; optional `--vessels` GFW enrichment), **`aisstream_fetch.py`** (live AIS → normalized contract), **`marinecadastre_fetch.py`** (historical AIS archive CSV → same contract), **`gfw_vessels.py`** (GFW identity/fishing/Insights; `--region-bbox`/`--wdpa-id` makes fishing_probability MPA-specific), **`report_violations.py`** (ranked CSV + self-contained HTML), and **`scan_violations.py`** (thin batch driver). The `violation_events.json` schema is pinned in `docs/PIPELINE.md`; a committed end-to-end fixture test (`test_pipeline_e2e.py`) guards the contract. **104 offline tests**, no network/key/ML deps. ⚠️ **The AISStream + GFW calls are NOT yet live-verified** — this remote sandbox blocks those hosts (egress policy) and has no keys/token, so live validation must run elsewhere (an env that allowlists the hosts, with the key + token); parsers are tolerant and every ingester has a `--dry-run`/local-file path. **All that remains is 🏠 do-from-home:** live validation + a first real violation (`docs/PHASE3_PLAN.md` Workstream 1), the live runs of the driver/MarineCadastre download, and optional Skylight cross-check (3B). Remaining work + definition of done: `docs/PHASE3_PLAN.md`; per-script contracts + first-run setup: `CLAUDE.md`.
- **Live model:** `data/training/weights/best.pt` = the **`finetune-smallvessel`** run (promoted 2026-07-14), **deployed at `imgsz 1024`, `conf 0.15`** — the YOLOv8n-**P2** arch (stride-4 head, strides [4,8,16,32]) retrained after folding in real human-reviewed **small-fishing-vessel** Mediterranean scenes (Cyclades + Egadi), warm-started from the prior `finetune-openocean` backbone. `detect_boats.py` defaults bake this in (in-repo `best.pt`, conf 0.15). Backups: `best_finetune-openocean.pt` (prior live, cross-region big-ship), `best_finetune-p2-1024.pt` (P2 Singapore-only), `best_finetune-aug-1024.pt`, `best_finetune-aug-640.pt`, `best_multiscene.pt`, `best_baseline_pre-multiscene.pt`.
- **The whole in-repo active-learning loop works** end-to-end: `sat_fetch → detect_boats → review_server → export_labels → build_dataset → train → promote → rescan`.
- **Large-ship RECALL is the current gap, not precision (2026-07-16).** Re-measured live `best.pt` on the Port Said holdout stratified by GT box size: at the deploy point (conf 0.15 / imgsz 1024) **precision 0.97 but recall only 0.69 — large ships (≥20px) 0.79 (22/28), mid (8–20px) 0.50 (7/14)**. Some misses are unmistakable bright-ship-with-wake signatures, not GSD-limited specks; conf 0.05 recovers only 1 more large ship while precision → 0.70 ⇒ a **training-coverage gap** (big ships learned from just 2 scenes: Fujairah 131 + Gibraltar 29 boxes), not a threshold. This corrects the earlier "big-ship precision is solved" framing — precision was solved, recall was never the headline number. **Free lever found:** `imgsz 1280 + augment=True` (TTA) lifts Port Said recall 0.69→**0.81** (large 0.79→0.86, mid 0.50→0.71) at **unchanged precision 0.97 / FP-chip 0.08** — but it is **big-ship-only**: on small-vessel Alonnisos it drops precision 0.40→0.34 and on Singapore FP/chip 2.50→5.75, so it must NOT be the global deploy default (offer as an opt-in big-ship inference profile). **Durable fix = scale the big-ship AL loop** (more dense-anchorage scenes → TRAIN, keep Port Said holdout).
- **Alonnisos holdout relabeled (2026-07-16) — trustworthy baseline.** Full human re-review via `review_server.py`: **76 boats** (was 49 — a ~55% undercount; 63 model boxes confirmed + 13 hand-added, 20 rejected). Corrected labels folded into TEST via `build_dataset.py --holdout-scene fallbacktest portsaid alonnisos` (train 659 / val 189 / test 143). Honest deploy-point number on the corrected set: **precision 0.40 / recall 0.70** (was reported 0.25/0.67 on the stale labels) — confirming many "FPs" were real unlabeled boats. This is the baseline O3 (hard negatives) is measured against.
- **O5/O6 eval refresh (2026-07-25) — deploy point re-confirmed; plain imgsz 1280 is a NON-WIN.** Re-ran `eval_holdout.py` + `conf_sweep.py` on the live `best.pt` (finetune-smallvessel) at the deploy point (conf 0.15 / imgsz 1024) over the relabeled holdouts, and swept imgsz 1024-vs-1280 on the *same* weights (IMPROVEMENT_PLAN O5/O6). **Fresh deploy-point numbers:** Alonnisos (small-vessel, 64 chips / 76 boxes) **P 0.40 / R 0.70, FP/chip 1.23, recall 90% CI [0.59, 0.79]**; Port Said (big-ship, 42 boxes) **P 0.97 / R 0.69, FP/chip 0.08, CI [0.57, 0.79]** — both reproduce the post-relabel baseline (Alonnisos FP/chip is 1.23 now vs the pre-relabel 1.55, since relabeling reclassified real boats). **imgsz 1280 (plain, no TTA) does not help:** Port Said 1024→1280 is **identical** (R 0.69→0.69, P 0.97, FP/chip 0.08) and Alonnisos gets **worse** (R 0.70→0.49, >5px 0.69→0.47) — upscaling past the 1024 train size loses the 1–5 px vessels. **So the Port Said 0.69→0.81 lift below is entirely from `--augment` (TTA), NOT raw imgsz** — raising tile/inference size alone buys nothing. `conf_sweep.py` puts the **F1 peak at conf 0.15 (F1 0.510)** on the relabeled Alonnisos → operating point unchanged. **Conclusion: keep imgsz 1024 / conf 0.15; 1536 not worth testing.** The remaining optical levers are the data-driven O3 (same-region hard negatives) + O4 (diverse positives + a 2nd small-vessel holdout), both 🏠 do-from-home.
- **Dual S1+S2 "200 each" warm-water sets assembled (2026-07-17) — for review, then their training sets.** Fetched 6 warm-water big-ship anchorages (RasTanura, Suez Gulf, Durban, Santos, Jebel Ali, Kaohsiung) in BOTH modalities to diversify away from the Baltic skew. **S2: 271 chips** (`*_s2warm-*`), seeded with the big-ship profile (`detect_boats --imgsz 1280 --augment`). **S1 SAR: 258 chips** (`*_s1warm-*`, `--provider s1` RTC), seeded with the new **`scripts/sar_seed.py`** — the classical SAR vessel detector that stands in for the not-yet-trained `best_sar.pt`. Seeding lands ~457 candidates over the ship-bearing chips (open-water AOIs like RasTanura are correctly ~empty). **`sar_seed.py` design (hard-won):** textbook CFAR/top-hat fail because `s1.py`'s display stretch leaves water mid-bright (median ~136/255, measured) and saturates vessels — so it uses despeckled **dual-pol geo-mean sqrt(VV·VH)** + a **per-chip adaptive median+k·MAD** threshold (stretch-invariant across scenes) + coherent-blob area filter + nodata-edge erosion + **`--water-only`** (WorldCover mask — essential: port cities like Jebel Ali are 45% land and buildings are very bright in SAR). Residual FPs on artificial structures (Jebel Ali Palm/breakwaters) are left for the human to reject. **SAR review shows a co-registered OPTICAL companion beside each SAR chip** (`scripts/sar_companion.py` reprojects the paired S2 scene into each SAR chip's grid → `companion/<chip>.png`; `review_server.py` auto-detects it) so land is unambiguous under speckle — land is static, so the S1/S2 date gap doesn't matter. **DEEP-WATER PIVOT (user directive):** shallow water returns strong bathymetric SAR clutter, and GEBCO showed **78% of the warm-anchorage SAR candidates were shallow** (anchorages are shallow by nature). So SAR refetched over deep-water vessel areas — **Fujairah (deep anchorage), Gibraltar & Hormuz (deep lanes)** — gated by `sar_seed.py --min-depth` (GEBCO via `bathymetry.py`; land reads as positive elevation so the depth gate also drops it). Deep set: `*_s1deep-*` / `*_s2deep-*`, **271 clean deep-water candidates** (Fujairah 207 the workhorse; Gibraltar 56, Hormuz 8 — deep water is vessel-sparse, that's the tradeoff). Seeder fix: cap the adaptive threshold below the ~255 vessel-saturation so rough-sea/high-MAD scenes don't miss every ship (Fujairah 0→207). The shallow `*_s1warm-*` set is superseded.
- **DEEP SAR REVIEWED + EXPORTED (2026-07-17) — the first `best_sar.pt` training set now exists.** All 3 deep SAR runs fully human-reviewed via the companion-enabled `review_server.py`: **Fujairah 42 chips (155 correct + 27 hand-added missed − 52 rejected) → 182 boxes; Gibraltar 35 chips → 23 boxes; Hormuz 72 chips → 1 box** (deep lanes are vessel-sparse, as expected — Fujairah the deep *anchorage* is the workhorse). Exported with `export_labels.py --out data/processed/sar_training_exports --tag deep` → **`data/processed/sar_training_exports/deep/`: 149 chips (37 positive / 112 empty hard-negative) / 206 SAR vessel boxes**. ⚠️ **This export lives OUTSIDE `data/processed/training_exports/` on purpose** — `build_dataset.py` sweeps *every* tag under `training_exports/` onto the OPTICAL `data/training/`, so a SAR tag there would poison the optical set with pseudo-RGB SAR chips. **SAR train flow BUILT + SMOKE-VALIDATED (2026-07-18).** No new script was needed — the generic flags already suffice: `build_dataset.py --data data/training_sar --exports data/processed/sar_training_exports [--holdout-scene s1deep-gibraltar] [--max-neg-ratio R]` assembles a SAR-only dataset (built: **train 48 / val 26 / Gibraltar-holdout test 35**; the new `--max-neg-ratio` tames the 3:1 neg-heavy SAR set — capped 2:1 here), then `train.py --data data/training_sar/config.yml --weights data/training/yolov8n-p2.yaml --pretrained yolov8n ...` trains it (P2 arch, ImageNet backbone — **never the optical `best.pt`**, wrong texture priors). A 1-epoch smoke ran clean at **batch 4** (no P2 OOM), producing a valid YOLOv8n-p2 `best.pt` → promote a real run to `data/training/weights/best_sar.pt`. **The full run is not done yet** — needs 30 epochs with the machine kept awake (the smoke's "7.6 h" was pure sleep-inflation; real compute for 48 imgs is ~30–60 min awake). 206 boxes + 112 deep-water hard negatives is a viable seed, but thin (esp. large-vessel variety) — grow it with more deep-anchorage scenes (more Fujairah-class) before expecting a strong SAR model. The paired `*_s2deep-*` chips (149) were companions-only (no independent detections); they're clean deep-water big-ship OPTICAL if wanted for `best.pt` later, but were not reviewed as an optical set.
- **HUGE S2-exact dataset assembled (2026-07-17) — STAGED, NOT YET TRAINED (held at user request).** To fix big-ship recall at scale without the review bottleneck, ingested the **Gulf-of-Finland Sentinel-2 vessel dataset** (Zenodo 15019034, CC-BY-4.0, 8,866 AIS-labeled boxes) through our own pipeline via `scripts/ingest_s2ships_finland.py`: resolves each product on Planetary Computer, downloads B04/B03/B02 (**via `requests`+CA-bundle — Avast throttles GDAL `/vsicurl` to uselessness**), same `linear/3000` stretch + 640 tiling as `sat_fetch`, keeps chips with a ≥60 m anchor vessel and labels **every** vessel in them (complete labels). Yield: **1,034 chips / 7,984 labels (658 large ≥150 m, 1,751 ≥100 m)**. Plus the reviewed **LA/San Pedro (86)** and **Panama (63)** anchorage scenes. `build_dataset.py --holdout-scene fallbacktest portsaid alonnisos` → **train 1,475 / val 422 / test 143** (Finnish in train/val only, 0 in test). **⚠️ CAVEAT: Finnish is now ~71% of train boxes and is cold-Baltic small-craft-skewed — domain-dominance risk; the 3 holdouts will catch it.** `best.pt` (live) predates this data. **Next when ready: `train.py --weights data/training/yolov8n-p2.yaml --pretrained best --imgsz 1024 --batch 4 --mosaic 0.5 --close-mosaic 15 --epochs 30 --name finetune-bigship`, then eval on Port Said** (big-ship recall, currently large 0.79 / mid 0.50 @ deploy). Rebalance to a ≥150 m anchor if Baltic dominance regresses warm-water holdouts.
- **Best result (imgsz 1024, plain-n):** on the clean held-out scene, deployment before/after — old (OLD@640, conf 0.25) → aug-1024 (NEW@1024, conf 0.20): instance **recall 0.31 → 0.73**, **precision 0.53 → 0.76**, **FP/chip 1.75 → 1.50**. The recall jump is the first statistically clean one (bootstrap CIs fully separate).
- **P2 head (Phase 0 remainder, promoted 2026-07-12) — the last free detector lever, and it STACKED.** A stride-4 P2 detection head trained fresh at 1024 (warm-started backbone) beats the prior plain-n `aug-1024` model on the same holdout at the shared deploy point conf 0.20: **recall 0.73 → 0.77** (20 vs 19 TP), **precision 0.76 → 0.87**, **FP/chip 1.50 → 0.75** (half the false positives). At its F1-peak conf 0.15 it reaches **recall 0.85 / precision 0.85 / FP-chip 1.0** — beating live @0.20 on every axis. No base-domain regression (base mAP50 0.634 → 0.658, base R 0.618 → 0.649) and open-water FP probe `testrun` **37 → 7**. Chose conf 0.20 for a conservative FP posture. **Caveat unchanged:** same thin 26-box single-location (Singapore) holdout — recall CIs still overlap ([0.44,0.68] vs [0.43,0.88]), so magnitude is uncertain; a fresh cross-location scene is still needed to confirm. Cost: P2 is ~1.5× slower (12.2 vs 8.1 GFLOPs).
- **Open-ocean retrain (promoted 2026-07-12, LIVE) — the data-collection loop, decisively validated.** Cross-validating the P2 model on FRESH open-ocean scenes exposed a catastrophic generalization gap: on a held-out dense-anchorage scene (Port Said, 42 boxes) it scored **recall 0.00** (0/42 @ conf 0.25) — the 0.77 Singapore number was wildly optimistic. Fix: fetched + human-reviewed real open-ocean scenes (Fujairah **131** boxes + Gibraltar **29**) through the in-repo loop, folded them into TRAIN (Port Said + Singapore held out via the new `build_dataset.py --holdout-scene`), retrained the P2 arch (warm-started, batch 4, 30 ep / 7 h). On the Port Said holdout (never trained on): **recall 0.00 → 0.67** @ conf 0.25 (**0.71 @ conf 0.15**, the swept F1 peak) at **precision 0.94–1.00, FP/chip 0–0.17**; recall CIs cleanly separated ([0,0] vs [0.52,0.79]). **No forgetting:** Singapore recall flat within noise (0.65→0.62) with precision UP (0.85→0.94), FP/chip DOWN (0.75→0.25); base mAP50 0.59→0.68. Two open-ocean scenes taught the model to detect a third in a new region. **Caveats:** Port Said is dense-anchored (same vessel *type* as Fujairah-in-train) → proves cross-*region*, not yet cross-*type* (small fishing boats untested); still one 42-box scene. **Next: scale the loop — more diverse open-ocean scenes, esp. small fishing vessels.**
- **Small-vessel retrain (promoted 2026-07-14, LIVE) — cross-TYPE gap closed.** The open-ocean model was all big anchored ships; small fishing vessels were untested. Measured the honest cross-type baseline on a clean **Alonnisos** (Aegean) holdout — the openocean model scored **recall 0.31 / precision 0.25 @ conf 0.15** (a real capability gap, *not* the GSD ceiling: conf 0.05 recovered recall to 0.63 → the model saw the boats but under-scored them). Fix: fetched + human-reviewed two MPA-adjacent Mediterranean scenes (**Cyclades 219** + **Egadi 223** boat boxes) through the in-repo loop, folded into TRAIN (Alonnisos + Singapore + Port Said held out), retrained the P2 arch (warm-started from openocean, batch 4, 30 ep / 6.8 h). On the Alonnisos holdout (never trained on): **recall 0.31 → 0.67** @ conf 0.15 (TP 15→33, FN 34→16), **precision held at 0.25**, recall 90% CIs cleanly separated ([0.19,0.45] vs [0.54,0.80]). The lift is concentrated on the >5px boats (15/48 → 32/48). **No forgetting:** base-test mAP50 0.629→0.626, base R 0.568→0.556 (noise). Cost: false positives roughly doubled (FP/chip 0.70→1.55) — but `conf_sweep.py` confirms **conf 0.15 is F1-optimal** (F1 0.365, peak) and the new model **Pareto-dominates** the old (at equal precision 2.2× the recall; at equal recall, better precision), so no operating-point or deploy-conf change. **Caveats:** one 49-box scene; precision is inherently low at 10 m GSD → this is a candidate-generation feed for HITL review, where recall is the metric that matters. **Cross-region AND cross-type now both demonstrated. Next: more diverse scenes (small artisanal boats, other regions) + the AIS-fusion reframe.**
- **The key finding — inference resolution is the dominant lever, not the retrain.** Just running the *existing* model at `imgsz 1024` (no training) lifted holdout recall **0.31 → 0.73** (1–5 px vessels → 2–10 px stop being sub-cell). The 5-h retrain at 1024 then bought a *cleaner frontier*: at matched recall (0.73) it has **~45% fewer false positives** than the raw-upscaled old model (FP/chip 1.5 vs 2.75). A conf sweep (`scripts/conf_sweep.py`) put the operating point at 0.20 (F1 near-peak; recall-vs-conf is steep, FP-vs-conf shallow).
- **Earlier win (augmentation, superseded as live):** `scripts/augment_positives.py` water-gated copy-paste fixed the 11-vs-58 boat/empty imbalance → 113 boat chips / 722 boxes; that model is now the `_640` backup.
- **Next levers:** (1) **scale the open-ocean data loop** — more diverse scenes, especially *small fishing vessels* (cross-type generalization is still unproven); (2) **Sentinel-1 SAR** (Phase 2 — provider built + validated on real chips (2026-07-15); classical `sar_seed.py` + human review now yielded the **first SAR training set** (`sar_training_exports/deep/`, 206 boxes / 112 hard-neg, 2026-07-17); next = a SAR base + build/train flow → `best_sar.pt`) for free all-weather / dark-vessel coverage; (3) the AIS dark-vessel reframe (Phase 3, the operational payoff). The P2 head + real open-ocean data closed the biggest detector gaps; remaining gains are more/diverse data + the SAR and AIS phases (no free global higher-res optical exists, so 10 m is the optical floor). See `docs/ROADMAP.md`.
- **Sentinel-1 SAR retrieval validated (2026-07-15).** A real S1 fetch over the Fujairah anchorage (same AOI as the existing S2 scene, ~1 week apart) confirms the SAR path end to end: bright backscatter blobs land on the optically-detected vessels, and SAR shows extra vessels the optical detector missed (comparison PNGs in the SAR run dir `2026-07-16T02-45-16Z_fujairah-sar/`). Two fixes made the real read work: **(a)** `_raster.py` `/vsicurl` allowlist `.tif,.TIF` → `.tif,.TIF,.tiff,.TIFF` (S1 assets are `.tiff`); **(b)** S1 default collection **GRD → RTC** — GRD has no CRS (radar geometry + GCPs), incompatible with the bbox windowed-read path; RTC is on a UTM grid. Caveat: the chosen RTC scene covered ~82% of the AOI (east no-data wedge). Next: train `best_sar.pt` (ROADMAP Phase 2 step 4).

---

## What works (functionality)

1. **End-to-end loop, all in-repo** (see `../CLAUDE.md` "Running Scripts" for exact commands):
   `sat_fetch.py` (fetch S2 chips) → `detect_boats.py` (YOLO inference + geolocate + dedup) →
   `review_server.py` (localhost HITL labelling) → `export_labels.py` (→ YOLO exports) →
   `build_dataset.py` (layer onto base set) → `train.py` (fine-tune) → promote → rescan.
2. **CPU fine-tuning** via the AntiAngler venv (`venv/Scripts/python.exe`), ultralytics 8.4.90 / torch 2.12.1+cpu. ~5 min/epoch; a 30-epoch run early-stops in ~1.3 h **if the machine stays awake** (sleep inflates wall-clock massively).
3. **Trustworthy tiny-object eval** — `scripts/eval_holdout.py` reports instance-level P/R via center-distance matching (IoU-mAP is unstable at 1–5 px), FP/chip, recall stratified by GT pixel size, and a bootstrap recall CI. Isolates satellite recall from the base-photo-dominated `val`. The only reliable way to read progress. Per-model **`--imgsz` / `--old-imgsz` / `--new-imgsz`** flags compare at different inference sizes (e.g. old@640 vs new@1024) and isolate test-time upscaling from retraining.
   - **Operating-point sweep** — `scripts/conf_sweep.py` runs one low-floor inference pass then re-thresholds across conf values, printing an instance-level PR curve (P/R/F1/FP-per-chip) plus an open-water false-alarm column. Used to pick the deployment conf per model (chose 0.20 for the 1024 model).
4. **Positive-synthesis augmentation** — `scripts/augment_positives.py` water-gated copy-paste: crops labeled boats, isolates them by brightness mask, pastes onto the reviewed empty MPA-water/reef chips (deployment domain), excludes the holdout scene (no leakage). Writes an `augmented` export tag that `build_dataset.py` layers. This is how the imbalance was fixed without new review.
5. **Small-object training knobs** — `train.py` `--mosaic/--close-mosaic/--copy-paste/--degrees` (default None = Ultralytics default, so existing flows unchanged).
6. **Fetching under Avast TLS interception** — solved; see the recipe below and `[[sat-fetch-tls-avast]]` memory.

## Model progression

| Run (`data/training/runs/<name>`) | Data added | SG Jul-5 holdout mAP50 / recall | Promoted? |
|---|---|---|---|
| _pre-AL baseline_ | base ~621 overhead photos only | 0.035 / 0.077 | was baseline |
| `finetune-smoke` (07-09) | +4 SG Jun-13 chips (same-scene, leaky eval) | ~ (eval meaningless) | no |
| `finetune-multiscene` (07-10) | +SG Jun-13 + False Bay (30) | 0.167 / 0.269 | superseded |
| `finetune-reef` (07-11) | +GBR Cairns reefs (35) | 0.129 / 0.215 (regressed) | no |
| `finetune-aug` (07-11) | +80 synth + 22 oversampled positive chips (copy-paste) | 0.444 / 0.31¹ | superseded (→ `_640` backup) |
| `finetune-aug-1024` (07-11) | same data, retrained at **imgsz 1024** | 0.741 / 0.73² | superseded (→ `_aug-1024` backup) |
| `finetune-p2-1024` (07-12) | same data, **P2 head** (stride-4) @ imgsz 1024, warm-started backbone | 0.773 / 0.77³ | superseded (→ `_p2-1024` backup) |
| `finetune-openocean` (07-12) | +Fujairah(131)+Gibraltar(29) **real open-ocean** | SG R 0.62 (held); **Port Said open-ocean R 0.00→0.67**⁴ | superseded (→ `_openocean` backup) |
| **`finetune-smallvessel` (07-14)** | +Cyclades(219)+Egadi(223) **real small-vessel Med** | Alonnisos cross-type R **0.31→0.67**, base held⁵ | ✅ **LIVE** |

¹ `finetune-aug`/`finetune-aug-1024` numbers are from the rebuilt instance-level eval (mAP50 / instance-recall); earlier rows show the old IoU-mAP50 / mAP-recall. `finetune-aug` (imgsz 640, conf 0.25) beat the multiscene model on recall (0.19→0.31), precision (0.39→0.53), FP/chip (2.0→1.75). Trained from the *original baseline* on the rebalanced set, `--mosaic 0.5 --close-mosaic 15`.

² `finetune-aug-1024` = the same recipe retrained at **imgsz 1024**, evaluated + **deployed at imgsz 1024, conf 0.20**. Recall 0.73 (19/26) at P 0.76, FP/chip 1.5. The imgsz-1024 *inference* is the dominant lever (upscaling the old model alone gave recall 0.73, CIs fully separate from 640's 0.31); the retrain bought ~45% fewer FPs at matched recall. Operating point chosen via `scripts/conf_sweep.py` (F1 peaks ~0.15–0.20). **Requires imgsz-1024 inference to realize** — `detect_boats.py` defaults updated accordingly.

³ `finetune-p2-1024` = a **YOLOv8n-P2** arch (`data/training/yolov8n-p2.yaml`, stride-4 head added) trained **fresh** at imgsz 1024, 30 epochs, warm-started from the prior `best.pt` backbone via `train.py --pretrained best` (219/437 layers transferred; the new P2 branch + Detect heads init random). Command: `train.py --weights data/training/yolov8n-p2.yaml --pretrained best --imgsz 1024 --batch 4 --mosaic 0.5 --close-mosaic 15 --epochs 30`. **Note `--batch 4`** (not 16): batch 16 @1024 with the P2 head exhausted memory (~18 GB worker RSS → ~1 GB commit headroom → silent torch C++ abort mid-epoch-1, no traceback); batch 4 plateaus at ~7 GB. Ran 8.58 h on CPU. Holdout @conf 0.20: recall 0.77 (20/26) / P 0.87 / FP-chip 0.75; @conf 0.15 (F1 peak): 0.85 / 0.85 / 1.0. See `train.py --pretrained` (added for this: warm-start a fresh `.yaml` arch).

⁴ `finetune-openocean` = same P2 arch retrained after folding in **real human-reviewed open-ocean scenes** (Fujairah 131 + Gibraltar 29 boxes) fetched/reviewed through the in-repo loop. Port Said (dense-anchorage Med, 42 boxes) held out via `build_dataset.py --holdout-scene fallbacktest portsaid` (new flag; reproducible scene-aware split, no leakage). Instance recall on Port Said: **0.00 → 0.67** @ conf 0.25, **0.71 @ conf 0.15** (F1 peak per `conf_sweep.py --holdout portsaid`), precision 0.94–1.00, FP/chip 0–0.17. Singapore held (R 0.65→0.62, P 0.85→0.94, FP/chip 0.75→0.25); base mAP50 0.59→0.68. Command: `train.py --weights data/training/yolov8n-p2.yaml --pretrained best --imgsz 1024 --batch 4 --mosaic 0.5 --close-mosaic 15 --epochs 30`.

⁵ `finetune-smallvessel` = same P2 arch, warm-started from the **openocean** `best.pt`, after folding in two real human-reviewed **small-fishing-vessel** MPA-adjacent Mediterranean scenes: Cyclades/Paros (219 boxes) + Egadi/Trapani (223 boxes), both fetched/reviewed through the in-repo loop. **Alonnisos** (Aegean, WDPAID 349993, 49 boxes) held out via `build_dataset.py --holdout-scene fallbacktest portsaid alonnisos` (dataset train 659 / val 189 / test 143). Cross-*type* holdout (small fishing boats — a vessel type absent from all prior training). Instance recall on Alonnisos @ conf 0.15: **0.31 → 0.67** (TP 15→33, FN 34→16), precision held 0.25, recall CIs [0.19,0.45]→[0.54,0.80] (separated); >5px stratum 15/48→32/48. No forgetting (base-test mAP50 0.629→0.626, R 0.568→0.556). FP/chip 0.70→1.55 but `conf_sweep.py --holdout alonnisos` shows conf 0.15 is the F1 peak (0.365) and NEW Pareto-dominates OLD across the PR curve → deploy conf unchanged. Command: `train.py --weights data/training/yolov8n-p2.yaml --pretrained best --imgsz 1024 --batch 4 --mosaic 0.5 --close-mosaic 15 --epochs 30 --name finetune-smallvessel`. 6.8 h on CPU.

`finetune-reef` suppressed reef false positives but **lowered recall and base-domain mAP** — piling on hard negatives made the detector too conservative. That lesson drove Phase 0: fix the imbalance by adding *positives*, not more negatives. Left unpromoted in `runs/finetune-reef/`.

## Current training data (10 reviewed scenes)

Export tags under `data/processed/training_exports/` (`2026-07-08 … 2026-07-14`) layered by
`build_dataset.py`; scene-aware split via `--holdout-scene fallbacktest portsaid alonnisos`
→ **train 659 / val 189 / test 143**.

| Scene (run dir) | Location | Boat boxes | Role | Added by |
|---|---|---|---|---|
| `…11-36-29Z_bbox` | Singapore Jun-13 | ~153 | TRAIN (positives) | baseline AL |
| `…11-44-23Z_fallbacktest` | Singapore Jul-5 | 26 | **TEST holdout** (cross-date) | baseline AL |
| `…01-08-52Z_testrun` | False Bay ZA | 2 (+28 hard-neg) | TRAIN (open-water neg) | baseline AL |
| `…04-32-01Z_gbr-cairns-reefs` | GBR Cairns | 6 (+30 hard-neg) | TRAIN (reef neg) | baseline AL |
| Fujairah anchorage | UAE | 131 | TRAIN (open-ocean positives) | `finetune-openocean` |
| Gibraltar | Strait | 29 | TRAIN (open-ocean positives) | `finetune-openocean` |
| Port Said | Med (dense anchor) | 42 | **TEST holdout** (cross-region) | `finetune-openocean` |
| `…22-38-08Z_cyclades-paros` | Cyclades, Aegean | 219 | TRAIN (small-vessel positives) | `finetune-smallvessel` |
| `…22-38-35Z_egadi-trapani` | Egadi, Sicily | 223 | TRAIN (small-vessel positives) | `finetune-smallvessel` |
| `…11-14-36Z_alonnisos-boundary` | Alonnisos, Aegean | 49 | **TEST holdout** (cross-type) | `finetune-smallvessel` |

Three clean held-out scenes now anchor eval: Singapore Jul-5 (cross-date), Port Said (cross-region),
Alonnisos (cross-type small vessels). The original ~11-vs-58 boat/negative imbalance is long resolved —
current TRAIN carries **~750+ real reviewed boat boxes** across big-ship + small-vessel domains.

---

## Open problems (priority order)

1. **Negative-heavy set → recall bottleneck — PARTLY ADDRESSED (Phase 0).** `augment_positives.py` copy-paste rebalanced to 113 boat chips / 722 boxes and lifted holdout recall 0.19→0.31. Remaining: synthetic boats reuse 161 real stamps at fixed orientation on ~58 real backgrounds, so *real* boat/background diversity is still limited. **Durable fix: still fetch boat-DENSE real scenes** (busy harbours, MPA boundaries/transit lanes) to grow genuine positives. See `[[mpa-focused-training-objective]]`.
2. **Scene-aware split is a manual override.** `build_dataset.py` does a flat random split; the SG-Jul5-as-TEST holdout is re-applied by hand after every rebuild (a random split scatters adjacent tiles of one scene across train/test → leaky eval, and buries the few positive chips). **Fix: add a `--holdout-scene <name-filter>` option to `build_dataset.py`** so the holdout is reproducible. Until then, re-run the reassignment snippet (below) after any `build_dataset.py`.
3. **Low absolute recall (~27%).** Sentinel-2's 10 m GSD renders sub-15 m vessels as 1–5 px specks — a hard ceiling fine-tuning can't fully beat. **Structural fix: add Sentinel-1 SAR or commercial hi-res** behind the existing `open_catalog/search_candidates/select_scene/read_aoi` seam in `sat_fetch.py`.
4. **Thin, single-location eval.** The holdout is 4 chips / 26 boxes, and it's the *same location* (Singapore) on a different date — a temporal, not cross-location, holdout. We have no clean cross-location generalization number yet (GBR could serve, but it's now in TRAIN). **Fix: hold out the next fresh scene before folding it in.**
5. **CPU-only training** — practical only while the machine stays awake; a GPU makes runs trivial.
6. **TLS interception** — every fetch needs the CA-bundle + `--insecure-tls` recipe, and the auto-mode classifier blocks `--insecure-tls` until the user authorizes it in-session.
7. **Minor: stale hint.** `export_labels.py` prints a "Next: point BoatDetection/scripts/helpers/splitData.py …" message — obsolete (that flow is now `build_dataset.py` → `train.py`). Cosmetic.

## Recommended next steps

**Phase 0 is complete, and BOTH generalization gates have now been run.** The cross-*location* gate
exposed the Singapore-only model's ~0 open-ocean recall → fixed by real open-ocean data
(`finetune-openocean`, Port Said 0.00→0.71). The cross-*type* gate exposed the openocean model's low
recall on small fishing vessels (Alonnisos 0.31) → fixed by real small-vessel Med data
(`finetune-smallvessel`, LIVE at conf 0.15, Alonnisos 0.31→0.67). **The active-learning data loop is the
proven engine** and has now closed the two biggest generalization gaps end-to-end.
`build_dataset.py --holdout-scene` is implemented (reproducible scene-aware split). **The full forward
plan — Sentinel-1 SAR (Phase 2, provider built) and AIS dark-vessel fusion (Phase 3) — is in `docs/ROADMAP.md`.**
Short version, in impact order:
1. **Keep scaling the data loop — still highest leverage.** Cross-region and cross-type are now both
   demonstrated on single scenes each; the remaining diversity gaps are **small artisanal/pleasure boats**
   (Alonnisos-class but even smaller) and **other regions/sea-states** (Atlantic, tropical, high-latitude).
   Also worth a **second small-vessel holdout** to confirm the 0.67 magnitude (it's one 49-box scene).
   Each fresh scene: hold it out, eval the live model on it first (the honest number), then fold in and
   retrain. Keep at least one fresh scene as a rotating holdout. Watch the FP/chip trend — the
   smallvessel model doubled it (0.70→1.55); if it keeps climbing, add reviewed empty-water hard
   negatives from the *same* regions to rein in precision without costing recall.
2. **Phase 2 (Sentinel-1 SAR — provider + `sar_seed.py` + first reviewed training set (`sar_training_exports/deep/`, 149 chips / 206 boxes) + the `build_dataset`/`train` SAR flow all built & smoke-validated 2026-07-18; remaining = run the full 30-epoch train → promote `best_sar.pt`, then grow the thin SAR set)** for free all-weather / dark-vessel coverage, or **Phase 3 (AIS dark-vessel reframe)** —
   see the roadmap. Phase 3 is the operational payoff and makes even a modest-recall detector "good
   enough" by filtering the detector's false positives against AIS.

---

## Operational recipes

**Fetch imagery under Avast TLS interception** (public read-only COGs; `--insecure-tls` is user-authorized each session — see `[[sat-fetch-tls-avast]]`):
```bash
# 1. build combined CA bundle (certifi + Avast root) once, e.g. via PowerShell:
#    export Cert:\LocalMachine\Root\CECD313F523A651EB7675F0D0EF8D9F64DD0DA80 as PEM,
#    append to venv/Lib/site-packages/certifi/cacert.pem -> ca_avast_bundle.pem
# 2. point Python AND GDAL at it, and allow GDAL to skip verify on the byte-reads:
export CA=/path/to/ca_avast_bundle.pem
export REQUESTS_CA_BUNDLE="$CA" SSL_CERT_FILE="$CA" CURL_CA_BUNDLE="$CA" GDAL_HTTP_CAINFO="$CA"
venv/Scripts/python.exe scripts/sat_fetch.py --bbox <minLon minLat maxLon maxLat> \
    --min-coverage 0.6 --label <slug> --insecure-tls
```

**Re-apply the scene-aware holdout after any `build_dataset.py`** (moves the SG-Jul5 `fallbacktest` chips to TEST, all other AL chips to TRAIN):
```python
import shutil; from pathlib import Path
TRAIN=Path("data/training"); AL="__"
tgt=lambda n:"test" if "fallbacktest" in n else "train"
for split in ("train","val","test"):
    for img in list((TRAIN/"images"/split).glob(f"*{AL}*.png")):
        t=tgt(img.name)
        if t==split: continue
        lbl=TRAIN/"labels"/split/f"{img.stem}.txt"
        shutil.move(str(img), str(TRAIN/"images"/t/img.name))
        if lbl.exists(): shutil.move(str(lbl), str(TRAIN/"labels"/t/lbl.name))
```

**Evaluate any candidate weights on the holdout + FP probes** (the live model runs at imgsz 1024, so
eval it there; a fresh candidate trained at 1024 → `--new-imgsz 1024`):
```bash
venv/Scripts/python.exe scripts/eval_holdout.py \
    --old data/training/weights/best.pt --old-imgsz 1024 \
    --new data/training/runs/<run>/weights/best.pt --new-imgsz 1024
# pick the deployment conf with the sweep:
venv/Scripts/python.exe scripts/conf_sweep.py \
    --weights data/training/runs/<run>/weights/best.pt --imgsz 1024
```

**Promote a run** (only after the holdout eval shows recall did not regress):
```bash
cp data/training/runs/<run>/weights/best.pt data/training/weights/best.pt
```
