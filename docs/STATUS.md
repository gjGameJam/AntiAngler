# STATUS — boat detector / active-learning loop

_Snapshot: 2026-07-26 (post label-harvest promotion). Point-in-time; verify against code before trusting specifics._

This tracks the live state of the satellite ship-detection model and its active-learning
loop. For the build spec + data contracts see `PIPELINE.md`; for known antipatterns see
`AUDIT.md`; for the daily GFW/WDPA streams see `../CLAUDE.md`; for the forward plan see `ROADMAP.md`.

---

## TL;DR — current state

- **Live model:** `data/training/weights/best.pt` = the **`finetune-diverse`** run (promoted 2026-07-26, same-day successor to `finetune-bigship`), **deployed at `imgsz 1024`, `conf 0.25`** — the P2 arch warm-started from `finetune-bigship` after the **13-set label harvest** (11 new reviewed scenes folded into TRAIN: Lofoten, Curaçao, SB-Channel lanes, Anacapa, Santos, Kaohsiung, Jebel Ali, Suez ×2, Durban, False Bay; train 659→2,512 images with `--max-neg-ratio 1.0` guarding the reef counter-lesson). `detect_boats.py` defaults bake this in. Backups: `best_finetune-bigship.pt` (prior live), `best_finetune-smallvessel.pt`, `best_finetune-openocean.pt`, `best_finetune-p2-1024.pt`, `best_finetune-aug-1024.pt`, `best_finetune-aug-640.pt`, `best_multiscene.pt`, `best_baseline_pre-multiscene.pt` (see the Model progression table).
- **Kornati (Adriatic) is the 2nd small-vessel holdout — and it broke the 0.67 story (2026-07-26).** Fresh never-trained region, 16 chips / **156 human boxes**: the then-live `finetune-bigship` scored only **P 0.73 / R 0.37** (CI [0.30, 0.45]) — the Alonnisos 0.67 does **not** transfer cross-region (open problem #4 was right to worry). `finetune-diverse` lifts it to R 0.51 @0.25 (CI [0.42, 0.60]) at F1-parity with the old model's best point (0.560 vs 0.573 — overlapping CIs). **Kornati precision is the named watch item**; no Adriatic scene is in TRAIN (Kornati stays a pure holdout), so the durable lever is more Adriatic/karst-coast data.
- **The whole in-repo active-learning loop works** end-to-end: `sat_fetch → detect_boats → review_server → export_labels → build_dataset → train → promote → rescan`.
- **Phase 3 (AIS dark-vessel fusion) — the whole sandbox-buildable pipeline is BUILT & offline-tested (2026-07-23).** Six scripts, each its own module (ingestion never imports the matcher): **`fuse_violations.py`** (the matcher — dark-vs-matched via a velocity-aware feasible-movement envelope; optional `--vessels` GFW enrichment), **`aisstream_fetch.py`** (live AIS → normalized contract), **`marinecadastre_fetch.py`** (historical AIS archive CSV → same contract), **`gfw_vessels.py`** (GFW identity/fishing/Insights; `--region-bbox`/`--wdpa-id` makes fishing_probability MPA-specific), **`report_violations.py`** (ranked CSV + self-contained HTML), and **`scan_violations.py`** (thin batch driver). Schema pinned in `docs/PIPELINE.md`; **98 offline P3 tests** (129 suite-wide), no network/key/ML deps. ✅ **GFW calls live-verified 2026-07-26 at home** — three v3 corrections landed (`events` needs `offset` with `limit`; identity picks the freshest cluster + gear from `combinedSourcesInfo`; insights include is `VESSEL-IDENTITY-IUU-VESSEL-LIST` → 201 flat object), each locked by a real captured-response fixture in `scripts/tests/fixtures/`. ✅ **First real violations landed 2026-07-26 (1C, historical path):** Anacapa Island SMR × MarineCadastre archive → 2 confirmed DARK (one wake vessel); the adjacent-lanes bbox run → 36 matched (all GFW-enriched live; best match 2 s / 362 m) / 7 dark. The first fusion caught + same-day-fixed three matcher/ingest defects (AIS sentinel SOG/COG passthrough, stale-ping admission, envelope-size attribution bias) — see `docs/PHASE3_PLAN.md` 1C. ✅ **AISStream live-verified 2026-07-26** (first connect: Singapore Strait, 41/41 positions kept from 37 vessels, 0 skipped/errors; file fuses cleanly). **Phase 3's definition of done is MET** — 1A+1B+1C all closed 2026-07-26; the only remaining W-item is the optional Skylight cross-check (3B) and the live-capture 1C variant (rolling AIS buffer over a fresh overpass) as an ongoing operational mode.
- **Deploy point & operating point** *(2026-07-25 record for the superseded `finetune-smallvessel`; the live `finetune-bigship` deploys at **conf 0.20** — see the first bullet + footnote ⁶)*. On the relabeled holdouts at conf 0.15 / imgsz 1024: **Alonnisos** (small-vessel, 76 boxes) **P 0.40 / R 0.70, FP/chip 1.23, recall 90% CI [0.59, 0.79]**; **Port Said** (big-ship, 42 boxes) **P 0.97 / R 0.69, FP/chip 0.08**. `conf_sweep.py` puts the **F1 peak at conf 0.15** → operating point unchanged. **Plain imgsz 1280 is a non-win:** Port Said identical, Alonnisos recall *drops* 0.70→0.49 (upscaling past the 1024 train size loses the 1–5 px vessels). The **big-ship recall lever is `--augment` (TTA)**, not raw imgsz: `--imgsz 1280 --augment` lifts Port Said recall 0.69→**0.81** (large 0.79→0.86) at unchanged precision — but it is **big-ship-only** (it worsens small-vessel/cluttered scenes), so it stays an opt-in profile, never the default. **Keep imgsz 1024 / conf 0.15; 1536 not worth testing.**
- **Large-ship recall (0.69) is a training-coverage gap, not a threshold or GSD limit** (2026-07-16): some Port Said misses are unmistakable bright-ship-with-wake signatures; conf 0.05 recovers only 1 more large ship while precision collapses to 0.70. Big ships were learned from just 2 scenes (Fujairah 131 + Gibraltar 29). **Durable fix = scale the big-ship AL loop** (more dense-anchorage scenes → the staged Finland set below targets exactly this).
- **Alonnisos holdout relabeled (2026-07-16) — trustworthy baseline.** Full human re-review: **76 boats** (was 49 — a ~55% undercount). Corrected labels in TEST via `build_dataset.py --holdout-scene fallbacktest portsaid alonnisos`. This is the honest baseline O3 (hard negatives) is measured against — many earlier "FPs" were real unlabeled boats.
- **Sentinel-1 SAR (Phase 2) — provider built + validated on real chips; first training set exists; full train pending.** `--provider s1` (RTC, dB-percentile pseudo-RGB `[VV,VH,VV/VH]`) validated on a real Fujairah fetch (2026-07-15: backscatter blobs coincide with optical detections, and SAR finds vessels optical missed). The classical **`sar_seed.py`** (dual-pol geo-mean + adaptive `median+k·MAD` + water/depth gates) + human review over deep-water runs (Fujairah/Gibraltar/Hormuz) yielded the **first SAR training set** — `data/processed/sar_training_exports/deep/`: **149 chips / 206 vessel boxes / 112 empty hard-negatives**. **`best_sar.pt` TRAINED + PROMOTED 2026-07-26** (`runs/sar-deep`, first GPU train — 30 epochs in 76 s): P2 arch, `yolov8n` backbone transfer (never the optical `best.pt`), `data/training_sar/` (train 48/val 26/test 35, `s1deep-gibraltar` held out). Honest test-split numbers (includes the reviewed Gibraltar holdout): **P 0.392 / R 0.478 / mAP50 0.335** — a thin-data first model; it replaces `sar_seed.py` as the SAR review loop's starting detector (`detect_boats.py --weights data/training/weights/best_sar.pt` on S1 runs). **Next: grow the thin set** (more Fujairah-class deep-anchorage scenes through seed/detect → review → export → retrain), then S6 speckle A/B + S7 cross-modality, and the xView3 transfer (S1/S2) as the big lever. **2026-07-26 update:** a fresh Singapore Strait S1 scene was reviewed (**267 vessel boxes over 18 chips** — `sar_training_exports/singapore/`; notably `best_sar.pt` scored **0 hits at conf 0.15** there, so the classical `sar_seed.py --water-only --min-depth 12` seeded the review — the per-scene dB stretch means each new region is out-of-domain for the thin model). The retrain **`sar-deep2` was NOT promoted**: Gibraltar-gate P 0.392→0.285 / R 0.478→0.435 / mAP50 0.335→0.298 (regression; though the gate is thin — 23 test boxes — so noise is large). Singapore data stays in `data/training_sar` TRAIN/VAL for the next round; the conclusion stands that SAR needs *more regions* (or xView3 transfer) before another gate attempt, not one more scene.
- **Gulf-of-Finland S2 dataset TRAINED + PROMOTED (2026-07-26, `finetune-bigship`).** The feared ~71% Baltic small-craft dominance did **not** regress the warm-water holdouts — it transferred: at the old deploy conf-equivalents, **Alonnisos** leapt (see deploy bullet), **Port Said** recall nudged up at near-perfect precision, **Singapore** improved (P 0.773→0.864, R 0.654→0.731 @0.25), base-test mAP50 0.629→0.645 (no forgetting). ⚠️ The one regression signal: the **False Bay open-water FP probe jumped 7→34 raw dets @0.25** (in-TRAIN, directional) — same-region open-water hard negatives (O3) are the follow-up if fresh-scene FP/chip climbs. GPU makes retrains ~25 min end-to-end.
- **Next levers, in impact order:** (1) **scale the data loop** — small artisanal boats + more regions/sea-states + a 2nd small-vessel holdout (highest recall leverage; the proven engine); (2) **finish Phase 2 SAR** (`best_sar.pt`) for free all-weather/dark-vessel coverage; (3) **Phase 3 AIS fusion** is the operational payoff — it filters the detector's false positives against AIS, making even a modest-recall detector "good enough." No free global higher-res optical exists, so 10 m is the optical floor. See `docs/ROADMAP.md`.

---

## What works (functionality)

1. **End-to-end loop, all in-repo** (see `../CLAUDE.md` "Running Scripts" for exact commands):
   `sat_fetch.py` → `detect_boats.py` → `review_server.py` → `export_labels.py` →
   `build_dataset.py` → `train.py` → promote → rescan.
2. **CPU fine-tuning** via the AntiAngler venv (`venv/Scripts/python.exe`), ultralytics 8.4.90 / torch 2.12.1+cpu. ~5 min/epoch; a 30-epoch run early-stops in ~1.3 h **if the machine stays awake** (sleep inflates wall-clock massively).
3. **Trustworthy tiny-object eval** — `scripts/eval_holdout.py`: instance-level P/R via center-distance matching (IoU-mAP is unstable at 1–5 px), FP/chip, recall stratified by GT pixel size, and a bootstrap recall CI. Isolates satellite recall from the base-photo-dominated `val`. Per-model `--imgsz`/`--old-imgsz`/`--new-imgsz` flags compare at different inference sizes.
   - **Operating-point sweep** — `scripts/conf_sweep.py` runs one low-floor inference pass then re-thresholds across conf values, printing an instance-level PR curve (P/R/F1/FP-per-chip) + an open-water false-alarm column. Picks the deployment conf per model.
4. **Positive-synthesis augmentation** — `scripts/augment_positives.py` water-gated copy-paste (crop labeled boats → paste onto reviewed empty MPA-water chips, holdout excluded). Fixed the original imbalance; **superseded** as the live recall lever by real diverse data.
5. **Small-object training knobs** — `train.py` `--mosaic/--close-mosaic/--copy-paste/--degrees` (default None = Ultralytics default).
6. **Fetching under Avast TLS interception** — solved; see the recipe below and `[[sat-fetch-tls-avast]]` memory.

## Model progression

_History collapsed here — each row is a promoted (or rejected) retrain; footnotes carry the eval detail._

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
| `finetune-smallvessel` (07-14) | +Cyclades(219)+Egadi(223) **real small-vessel Med** | Alonnisos cross-type R **0.31→0.67**, base held⁵ | superseded (→ `_smallvessel` backup) |
| `finetune-bigship` (07-26) | +Gulf-of-Finland S2 (1,034 chips / 7,984 boxes) | Alonnisos @0.20: P 0.48/R **0.76** (was 0.40/0.70 @0.15); Port Said P 0.94/R 0.71; SG improved⁶ | superseded same-day (→ `_bigship` backup) |
| **`finetune-diverse` (07-26)** | +11 reviewed scenes from the 13-set label harvest (~549 boxes + 159 hard-neg chips, neg-capped) | recall UP on **all 4 holdouts** @0.25: Kornati 0.37→**0.51**, Alonnisos 0.67→**0.80**, Port Said 0.69→**0.79**, SG 0.73→0.77; base mAP50 0.594→0.614⁷ | ✅ **LIVE** (deploy conf **0.25**) |

The lessons the table encodes: **inference resolution (imgsz 1024) was the dominant early lever** (upscaling the old model alone lifted holdout recall 0.31→0.73; the retrain then bought ~45% fewer FPs at matched recall); the **P2 stride-4 head stacked** on top; then **real, diverse training data** proved the highest-leverage move — twice (cross-region via open-ocean, cross-type via small-vessel Med). `finetune-reef` is the counter-lesson: piling on hard negatives made the detector too conservative (recall + base-mAP both dropped) → the fix is adding *positives*, not negatives.

¹ `finetune-aug` (imgsz 640, conf 0.25) beat the multiscene model on recall (0.19→0.31), precision (0.39→0.53), FP/chip (2.0→1.75). Trained from the *original baseline* on the rebalanced set, `--mosaic 0.5 --close-mosaic 15`.

² `finetune-aug-1024` = same recipe at **imgsz 1024**, deployed at conf 0.20. Recall 0.73 (19/26) at P 0.76, FP/chip 1.5. The imgsz-1024 *inference* is the dominant lever (upscaling the old model alone gave recall 0.73, CIs fully separate from 640's 0.31); the retrain bought ~45% fewer FPs at matched recall.

³ `finetune-p2-1024` = a **YOLOv8n-P2** arch (`data/training/yolov8n-p2.yaml`, stride-4 head) trained **fresh** at imgsz 1024, warm-started from the prior `best.pt` backbone via `train.py --pretrained best` (219/437 layers transferred; new P2 branch + Detect heads init random). **Note `--batch 4`** (not 16): batch 16 @1024 with the P2 head exhausts memory → silent torch C++ abort mid-epoch-1; batch 4 plateaus at ~7 GB. Holdout @conf 0.20: recall 0.77 (20/26) / P 0.87 / FP-chip 0.75; @conf 0.15 (F1 peak): 0.85 / 0.85 / 1.0. Cost: P2 is ~1.5× slower (12.2 vs 8.1 GFLOPs).

⁴ `finetune-openocean` = same P2 arch after folding in real reviewed open-ocean scenes (Fujairah 131 + Gibraltar 29). Port Said (dense-anchorage Med, 42 boxes) held out via `build_dataset.py --holdout-scene fallbacktest portsaid`. Port Said recall **0.00 → 0.67** @ conf 0.25, **0.71 @ conf 0.15**, precision 0.94–1.00, FP/chip 0–0.17; recall CIs cleanly separated ([0,0] vs [0.52,0.79]). Singapore held (R 0.65→0.62, P 0.85→0.94); base mAP50 0.59→0.68. Proves cross-*region* (not yet cross-*type*).

⁵ `finetune-smallvessel` (LIVE) = same P2 arch, warm-started from **openocean**, after folding in two reviewed small-fishing-vessel Med scenes: Cyclades/Paros (219) + Egadi/Trapani (223). **Alonnisos** (Aegean, WDPAID 349993) held out. Instance recall @ conf 0.15: **0.31 → 0.67** (TP 15→33), precision held 0.25, recall CIs [0.19,0.45]→[0.54,0.80] (separated). No forgetting (base-test mAP50 0.629→0.626). FP/chip 0.70→1.55, but `conf_sweep.py` shows conf 0.15 is the F1 peak and NEW Pareto-dominates OLD → deploy conf unchanged. (Numbers here are vs the 49-box pre-relabel Alonnisos; the current honest baseline on the 76-box relabel is P 0.40 / R 0.70 — see TL;DR.) Command: `train.py --weights data/training/yolov8n-p2.yaml --pretrained best --imgsz 1024 --batch 4 --mosaic 0.5 --close-mosaic 15 --epochs 30 --name finetune-smallvessel`. 6.8 h on CPU.

⁶ `finetune-bigship` (LIVE) = P2 arch warm-started from `smallvessel`, Finland-s2 folded in (train 1,475/val 422/test 143), **first GPU train** (RTX 4070, 30 epochs ≈ 25 min). Gated eval 2026-07-26, all three holdouts, old@1024 vs new@1024: **Singapore** instance P 0.773→0.864 / R 0.654→0.731 / FP-chip 1.25→0.75 (@0.25); **Port Said** R 0.667→0.690 @ P 1.00 (@0.25), and @0.20 P 0.938/R 0.714/FP-chip 0.17; **Alonnisos** @0.25 P 0.452→0.607 / R 0.184→**0.671** (recall CIs [0.10,0.32]→[0.58,0.80], separated). `conf_sweep` on Alonnisos: F1 peak **0.637 @ conf 0.25** (old model: 0.510 @ 0.15); **deploy conf 0.20** picked as the recall-leaning point that Pareto-dominates the old deploy on every holdout metric (Alonnisos 0.479/0.763/0.98 vs 0.40/0.70/1.23). Caveat: False Bay open-water probe 7→34 raw dets @0.25 → watch fresh-scene FP/chip, O3 if it climbs. Command: `train.py --weights data/training/yolov8n-p2.yaml --pretrained best --imgsz 1024 --batch 4 --mosaic 0.5 --close-mosaic 15 --epochs 30 --name finetune-bigship`.

⁷ `finetune-diverse` (LIVE) = P2 arch warm-started from `bigship`, after folding in the **2026-07-26 13-set label harvest**: 11 new reviewed scenes → TRAIN/VAL (Lofoten 30, Curaçao 34, SB-Channel lanes 40, Anacapa 4, Santos 155, Kaohsiung 122, Jebel Ali 97, Suez anchorage 36, Suez gulf 7, Durban 23, False Bay 1 — ~549 boxes + 159 hard-negative chips, capped by `build_dataset.py --max-neg-ratio 1.0`), **Kornati (156 boxes) held out as the 4th TEST holdout**. Gated eval old@1024 vs new@1024 (conf 0.25): recall up on all four holdouts (Kornati 0.372→**0.513**, Alonnisos 0.671→**0.803**, Port Said 0.690→**0.786**, SG 0.731→0.769); precision down at fixed conf everywhere (Kornati 0.734→0.567 the largest). `conf_sweep`: Alonnisos F1 peak 0.637→**0.704** (@0.30) and NEW@0.25 beats the old deploy on P *and* R *and* FP/chip there; Kornati F1 peak 0.560 vs old 0.573 (statistical parity, CIs overlap); **deploy conf moved 0.20→0.25** (the recall-leaning point, same +0.05 pattern as the last promotion). Watch items: Kornati precision (FP/chip 3.81 @0.25 vs old-deploy 3.31) and the gbr FP probe 15→19. No forgetting: base-test mAP50 0.594→0.614. Command: `train.py --weights data/training/yolov8n-p2.yaml --pretrained best --imgsz 1024 --batch 4 --mosaic 0.5 --close-mosaic 15 --epochs 30 --name finetune-diverse`.

## Current training data (the live model — 21 reviewed scenes + Finland-s2)

The live `best.pt` trained on the 10-scene set below **plus Finland-s2 plus the 2026-07-26 harvest**
(11 more reviewed scenes: Lofoten, Curaçao, SB-Channel lanes, Anacapa, Santos, Kaohsiung, Jebel Ali,
Suez anchorage, Suez gulf, Durban, False Bay — **train 2,512 / val 712 / test 159**), export tags under
`data/processed/training_exports/`, scene-aware split via
`build_dataset.py --holdout-scene fallbacktest portsaid alonnisos kornati-adriatic --max-neg-ratio 1.0`.
**Kornati (Adriatic, 156 boxes) is the 4th TEST holdout** — never in TRAIN.

| Scene (run dir) | Location | Boat boxes | Role |
|---|---|---|---|
| `…11-36-29Z_bbox` | Singapore Jun-13 | ~153 | TRAIN (positives) |
| `…11-44-23Z_fallbacktest` | Singapore Jul-5 | 26 | **TEST holdout** (cross-date) |
| `…01-08-52Z_testrun` | False Bay ZA | 2 (+28 hard-neg) | TRAIN (open-water neg) |
| `…04-32-01Z_gbr-cairns-reefs` | GBR Cairns | 6 (+30 hard-neg) | TRAIN (reef neg) |
| Fujairah anchorage | UAE | 131 | TRAIN (open-ocean positives) |
| Gibraltar | Strait | 29 | TRAIN (open-ocean positives) |
| Port Said | Med (dense anchor) | 42 | **TEST holdout** (cross-region) |
| `…22-38-08Z_cyclades-paros` | Cyclades, Aegean | 219 | TRAIN (small-vessel positives) |
| `…22-38-35Z_egadi-trapani` | Egadi, Sicily | 223 | TRAIN (small-vessel positives) |
| `…11-14-36Z_alonnisos-boundary` | Alonnisos, Aegean | 76 (relabeled) | **TEST holdout** (cross-type) |

Four clean held-out scenes anchor eval: Singapore Jul-5 (cross-date), Port Said (cross-region),
Alonnisos (cross-type small vessels), **Kornati (cross-region small vessels, added 2026-07-26)**.
TRAIN carries ~1,300+ real reviewed boat boxes across big-ship + small-vessel domains and 11 regions.

---

## Open problems (priority order)

1. **On-water false alarms on small-vessel scenes are the dominant precision problem** (Alonnisos P 0.40). Per `IMPROVEMENT_PLAN.md` Step 0, land is only ~16% of FPs; ~84% are on-water (glint/swell/reef-edge). **Fix = O3 same-region hard negatives + O7 AIS fusion (built)** — not raising conf (trades recall) or piling on negatives (made `finetune-reef` too conservative).
2. **Recall is GSD-capped.** Sentinel-2's 10 m renders sub-15 m vessels as 1–5 px specks — a physical floor. **Structural fix = Sentinel-1 SAR (Phase 2) + the AIS-mismatch reframe (Phase 3)**, both behind the built `sat_fetch.py` provider seam / fusion layer.
3. **Big-ship recall gap (0.69)** — a training-coverage gap (2 scenes), addressed by the staged Finland set + more dense-anchorage scenes; not the GSD limit.
4. ~~Thin eval — single scene per gate~~ **PARTIALLY RESOLVED 2026-07-26:** Kornati (156 boxes) is the 2nd small-vessel holdout — and it showed the worry was justified: the 0.67 Alonnisos recall did **not** transfer (bigship scored R 0.37 there; `finetune-diverse` lifts it to 0.51). **Successor problem: cross-region small-vessel recall (Kornati R 0.51, P 0.57 @0.25)** — the model has zero Adriatic/karst-coast TRAIN data by design; more same-class regions (Croatian/Greek/Turkish coasts NOT named Kornati) are the lever.
5. ~~CPU-only training~~ **RESOLVED 2026-07-26:** venv torch is now 2.12.1+cu130 on the RTX 4070 (`train.py --device` auto-detects CUDA); a 30-epoch 1024px P2 run takes ~25 min.
6. **TLS interception** — every fetch needs the CA-bundle + `--insecure-tls` recipe (below), and the auto-mode classifier blocks `--insecure-tls` until authorized in-session.
7. ~~Minor/cosmetic: stale `BoatDetection` pointers~~ **RESOLVED 2026-07-29:** `export_labels.py`'s header now names `build_dataset.py` → `train.py`, and `detect_boats.py`'s weights-not-found error points at `data/training/weights/` instead of the legacy sibling repo.

## Recommended next steps

**Everything runnable is run (2026-07-26): Phase 3 live-validated end-to-end, 13-set label harvest
reviewed + retrained + promoted, GPU unlocked.** In impact order:

1. **Attack the Kornati gap (cross-region small-vessel recall, R 0.51 / P 0.57).** Fetch + review 2–3
   more Adriatic-class scenes (Croatian islands, Greek Ionian, Turkish coast — NOT Kornati itself, it
   stays a pure holdout), fold in, retrain, re-gate on all 4 holdouts. Same loop as today, ~40 min of
   review per cycle.
2. **SAR needs regions, not one more scene:** `sar-deep2` (+Singapore) regressed the Gibraltar gate →
   next attempt only after 2+ more reviewed deep-water regions or the xView3 transfer (`docs/ROADMAP.md`).
   `best_sar.pt` = `sar-deep` unchanged; note it scores 0 on unseen regions (per-scene dB stretch), so
   seed new SAR reviews with `sar_seed.py --water-only --min-depth 12`.
3. **Operational mode:** run the live product loop — rolling `aisstream_fetch --wdpa-id` buffer over a
   trafficked MPA + fresh S2 overpass → `scan_violations.py` (the 1C-live variant). Optional: Skylight
   cross-check (3B, needs registration).
4. ~~Pending user action: merge the branch to main~~ **DONE 2026-07-29** — work continues on `main`.

**Hygiene round landed 2026-07-29 (before resuming the data loop):** AUDIT #4 (network retry/backoff) is
closed — one shared policy in `scripts/_http.py` wired into every `requests` call site, plus
`GDAL_HTTP_MAX_RETRY`/`RETRY_DELAY` on the `/vsicurl` COG reads, so the fetch-heavy work in steps 1–2
above stops losing a run to a transient 5xx. SAR runs now land in their own `data/raw/sentinel1/` root.
Suite is 129 offline tests. Not covered: websocket reconnect for `aisstream_fetch.py` (a capture-semantics
decision, deliberately left open), and retries do **not** help the Avast schannel wall.

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

**Evaluate any candidate weights on the holdout + FP probes** (the live model runs at imgsz 1024):
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
