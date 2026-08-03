# STATUS — boat detector / active-learning loop

_Snapshot: **2026-08-03** (W5 shipped: the eval harness is now modality-agnostic and gates SAR — which
promptly showed `best_sar.pt` has no usable operating point. Previous cycle: W1 Adriatic,
`finetune-adriatic` promoted at **conf 0.20**). Point-in-time; verify against code before trusting
specifics._

This tracks the live state of the satellite ship-detection model and its active-learning
loop. For the build spec + data contracts see `PIPELINE.md`; for known antipatterns see
`AUDIT.md`; for the daily GFW/WDPA streams see `../CLAUDE.md`.

**➡ For what to do next, read `ROADMAP.md` → "THE WORK QUEUE"** — it carries the prioritized items
(W1–W9) with exact commands, acceptance tests and traps, plus a **"Do not re-attempt"** table of
measured dead ends. This file is the *state*; that one is the *plan*.

---

## TL;DR — current state

- **Live model:** `data/training/weights/best.pt` = the **`finetune-adriatic`** run (promoted **2026-08-02**), **deployed at `imgsz 1024`, `conf 0.20`** — the P2 arch warm-started from `finetune-diverse` after the **W1 Adriatic harvest** (3 in-class scenes: Šibenik/Primošten + Hvar/Pakleni + Göcek/Fethiye, **1093 boat boxes**, train 2512→2725). ⚠️ **The deploy conf moved DOWN 0.25→0.20**, reversing the previous two promotions' +0.05 pattern — this model is more conservative per-box, and judging it at the inherited 0.25 understates it badly (Kornati R 0.468 @0.25 vs **0.590 @0.20**). **What it bought was precision, not recall:** open-water false alarms −45% (1.43→0.79 dets/chip @0.20), base mAP50 0.614→**0.640** (no forgetting), Kornati weakly dominating the old deploy point. What it did **not** buy: any lift in the Kornati recall *ceiling* — see the W1 outcome box in `ROADMAP.md`. Backup of the outgoing model: `best_finetune-diverse.pt`.
- _(superseded)_ `finetune-diverse` (promoted 2026-07-26, was **`conf 0.25`**) — the P2 arch warm-started from `finetune-bigship` after the **13-set label harvest** (11 new reviewed scenes folded into TRAIN: Lofoten, Curaçao, SB-Channel lanes, Anacapa, Santos, Kaohsiung, Jebel Ali, Suez ×2, Durban, False Bay; train 659→2,512 images with `--max-neg-ratio 1.0` guarding the reef counter-lesson). `detect_boats.py` defaults bake this in. Backups: `best_finetune-bigship.pt` (prior live), `best_finetune-smallvessel.pt`, `best_finetune-openocean.pt`, `best_finetune-p2-1024.pt`, `best_finetune-aug-1024.pt`, `best_finetune-aug-640.pt`, `best_multiscene.pt`, `best_baseline_pre-multiscene.pt` (see the Model progression table).
- **Kornati (Adriatic) is the 2nd small-vessel holdout — and it broke the 0.67 story (2026-07-26).** Fresh never-trained region, 16 chips / **156 human boxes**: the then-live `finetune-bigship` scored only **P 0.73 / R 0.37** (CI [0.30, 0.45]) — the Alonnisos 0.67 does **not** transfer cross-region (open problem #4 was right to worry). `finetune-diverse` lifts it to R 0.51 @0.25 (CI [0.42, 0.60]) at F1-parity with the old model's best point (0.560 vs 0.573 — overlapping CIs). **Kornati precision is the named watch item**; no Adriatic scene is in TRAIN (Kornati stays a pure holdout), so the durable lever is more Adriatic/karst-coast data.
- **The whole in-repo active-learning loop works** end-to-end: `sat_fetch → detect_boats → review_server → export_labels → build_dataset → train → promote → rescan`.
- **Phase 3 (AIS dark-vessel fusion) — the whole sandbox-buildable pipeline is BUILT & offline-tested (2026-07-23).** Six scripts, each its own module (ingestion never imports the matcher): **`fuse_violations.py`** (the matcher — dark-vs-matched via a velocity-aware feasible-movement envelope; optional `--vessels` GFW enrichment), **`aisstream_fetch.py`** (live AIS → normalized contract), **`marinecadastre_fetch.py`** (historical AIS archive CSV → same contract), **`gfw_vessels.py`** (GFW identity/fishing/Insights; `--region-bbox`/`--wdpa-id` makes fishing_probability MPA-specific), **`report_violations.py`** (ranked CSV + self-contained HTML), and **`scan_violations.py`** (thin batch driver). Schema pinned in `docs/PIPELINE.md`; **98 offline P3 tests** (203 suite-wide), no network/key/ML deps. ✅ **GFW calls live-verified 2026-07-26 at home** — three v3 corrections landed (`events` needs `offset` with `limit`; identity picks the freshest cluster + gear from `combinedSourcesInfo`; insights include is `VESSEL-IDENTITY-IUU-VESSEL-LIST` → 201 flat object), each locked by a real captured-response fixture in `scripts/tests/fixtures/`. ✅ **First real violations landed 2026-07-26 (1C, historical path):** Anacapa Island SMR × MarineCadastre archive → 2 confirmed DARK (one wake vessel); the adjacent-lanes bbox run → 36 matched (all GFW-enriched live; best match 2 s / 362 m) / 7 dark. The first fusion caught + same-day-fixed three matcher/ingest defects (AIS sentinel SOG/COG passthrough, stale-ping admission, envelope-size attribution bias) — see `docs/PHASE3_PLAN.md` 1C. ✅ **AISStream live-verified 2026-07-26** (first connect: Singapore Strait, 41/41 positions kept from 37 vessels, 0 skipped/errors; file fuses cleanly). **Phase 3's definition of done is MET** — 1A+1B+1C all closed 2026-07-26; the only remaining W-item is the optional Skylight cross-check (3B) and the live-capture 1C variant (rolling AIS buffer over a fresh overpass) as an ongoing operational mode.
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
3. **Trustworthy tiny-object eval** — `scripts/eval_holdout.py`: instance-level P/R via center-distance matching (IoU-mAP is unstable at 1–5 px), FP/chip, recall stratified by GT pixel size, and a bootstrap recall CI. Isolates satellite recall from the base-photo-dominated `val`. Per-model `--imgsz`/`--old-imgsz`/`--new-imgsz` flags compare at different inference sizes. **`--train-dir` (2026-08-03, W5) makes it modality-agnostic** — `data/training` (default, optical) or `data/training_sar`; `PROMOTED_BY_TREE` binds each tree to the checkpoint that gates it, and a zero-match `--holdout` or zero-chip FP probe now errors/warns instead of reporting a silent zero.
   - **Operating-point sweep** — `scripts/conf_sweep.py` runs one low-floor inference pass then re-thresholds across conf values, printing an instance-level PR curve (P/R/F1/FP-per-chip) + an open-water false-alarm column. Picks the deployment conf per model. Takes the same `--train-dir`.
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
| `finetune-diverse` (07-26) | +11 reviewed scenes from the 13-set label harvest (~549 boxes + 159 hard-neg chips, neg-capped) | recall UP on **all 4 holdouts** @0.25: Kornati 0.37→**0.51**, Alonnisos 0.67→**0.80**, Port Said 0.69→**0.79**, SG 0.73→0.77; base mAP50 0.594→0.614⁷ | superseded (→ `_diverse` backup) |
| `finetune-adriatic-60ep` (08-02) | same data, 60 ep, `--close-mosaic 15` unscaled | early-stopped @21, "best = epoch 1" — never reached the mosaic-free tail | ❌ no (config error) |
| `finetune-adriatic-60ep-b` (08-02) | same data, 60 ep, `--close-mosaic 30 --patience 0` | best Kornati R (0.635 @0.20) but base recall 0.585→**0.556** (forgetting) + Port Said 0.786→0.762 | ❌ not promoted |
| **`finetune-adriatic` (08-02)** | +3 Adriatic-class scenes (**1093 boxes**: Šibenik 262 / Hvar 483 / Göcek 348) | **precision win, not recall**: open-water FP −45%, base mAP50 0.614→**0.640**; Kornati 0.513→0.590 @0.20 (CI overlaps) but SG 0.769→0.731⁸ | ✅ **LIVE** (deploy conf **0.20**) |

The lessons the table encodes: **inference resolution (imgsz 1024) was the dominant early lever** (upscaling the old model alone lifted holdout recall 0.31→0.73; the retrain then bought ~45% fewer FPs at matched recall); the **P2 stride-4 head stacked** on top; then **real, diverse training data** proved the highest-leverage move — twice (cross-region via open-ocean, cross-type via small-vessel Med). `finetune-reef` is the counter-lesson: piling on hard negatives made the detector too conservative (recall + base-mAP both dropped) → the fix is adding *positives*, not negatives.

¹ `finetune-aug` (imgsz 640, conf 0.25) beat the multiscene model on recall (0.19→0.31), precision (0.39→0.53), FP/chip (2.0→1.75). Trained from the *original baseline* on the rebalanced set, `--mosaic 0.5 --close-mosaic 15`.

² `finetune-aug-1024` = same recipe at **imgsz 1024**, deployed at conf 0.20. Recall 0.73 (19/26) at P 0.76, FP/chip 1.5. The imgsz-1024 *inference* is the dominant lever (upscaling the old model alone gave recall 0.73, CIs fully separate from 640's 0.31); the retrain bought ~45% fewer FPs at matched recall.

³ `finetune-p2-1024` = a **YOLOv8n-P2** arch (`data/training/yolov8n-p2.yaml`, stride-4 head) trained **fresh** at imgsz 1024, warm-started from the prior `best.pt` backbone via `train.py --pretrained best` (219/437 layers transferred; new P2 branch + Detect heads init random). **Note `--batch 4`** (not 16): batch 16 @1024 with the P2 head exhausts memory → silent torch C++ abort mid-epoch-1; batch 4 plateaus at ~7 GB. Holdout @conf 0.20: recall 0.77 (20/26) / P 0.87 / FP-chip 0.75; @conf 0.15 (F1 peak): 0.85 / 0.85 / 1.0. Cost: P2 is ~1.5× slower (12.2 vs 8.1 GFLOPs).

⁴ `finetune-openocean` = same P2 arch after folding in real reviewed open-ocean scenes (Fujairah 131 + Gibraltar 29). Port Said (dense-anchorage Med, 42 boxes) held out via `build_dataset.py --holdout-scene fallbacktest portsaid`. Port Said recall **0.00 → 0.67** @ conf 0.25, **0.71 @ conf 0.15**, precision 0.94–1.00, FP/chip 0–0.17; recall CIs cleanly separated ([0,0] vs [0.52,0.79]). Singapore held (R 0.65→0.62, P 0.85→0.94); base mAP50 0.59→0.68. Proves cross-*region* (not yet cross-*type*).

⁵ `finetune-smallvessel` (LIVE) = same P2 arch, warm-started from **openocean**, after folding in two reviewed small-fishing-vessel Med scenes: Cyclades/Paros (219) + Egadi/Trapani (223). **Alonnisos** (Aegean, WDPAID 349993) held out. Instance recall @ conf 0.15: **0.31 → 0.67** (TP 15→33), precision held 0.25, recall CIs [0.19,0.45]→[0.54,0.80] (separated). No forgetting (base-test mAP50 0.629→0.626). FP/chip 0.70→1.55, but `conf_sweep.py` shows conf 0.15 is the F1 peak and NEW Pareto-dominates OLD → deploy conf unchanged. (Numbers here are vs the 49-box pre-relabel Alonnisos; the current honest baseline on the 76-box relabel is P 0.40 / R 0.70 — see TL;DR.) Command: `train.py --weights data/training/yolov8n-p2.yaml --pretrained best --imgsz 1024 --batch 4 --mosaic 0.5 --close-mosaic 15 --epochs 30 --name finetune-smallvessel`. 6.8 h on CPU.

⁶ `finetune-bigship` (LIVE) = P2 arch warm-started from `smallvessel`, Finland-s2 folded in (train 1,475/val 422/test 143), **first GPU train** (RTX 4070, 30 epochs ≈ 25 min). Gated eval 2026-07-26, all three holdouts, old@1024 vs new@1024: **Singapore** instance P 0.773→0.864 / R 0.654→0.731 / FP-chip 1.25→0.75 (@0.25); **Port Said** R 0.667→0.690 @ P 1.00 (@0.25), and @0.20 P 0.938/R 0.714/FP-chip 0.17; **Alonnisos** @0.25 P 0.452→0.607 / R 0.184→**0.671** (recall CIs [0.10,0.32]→[0.58,0.80], separated). `conf_sweep` on Alonnisos: F1 peak **0.637 @ conf 0.25** (old model: 0.510 @ 0.15); **deploy conf 0.20** picked as the recall-leaning point that Pareto-dominates the old deploy on every holdout metric (Alonnisos 0.479/0.763/0.98 vs 0.40/0.70/1.23). Caveat: False Bay open-water probe 7→34 raw dets @0.25 → watch fresh-scene FP/chip, O3 if it climbs. Command: `train.py --weights data/training/yolov8n-p2.yaml --pretrained best --imgsz 1024 --batch 4 --mosaic 0.5 --close-mosaic 15 --epochs 30 --name finetune-bigship`.

⁷ `finetune-diverse` (LIVE) = P2 arch warm-started from `bigship`, after folding in the **2026-07-26 13-set label harvest**: 11 new reviewed scenes → TRAIN/VAL (Lofoten 30, Curaçao 34, SB-Channel lanes 40, Anacapa 4, Santos 155, Kaohsiung 122, Jebel Ali 97, Suez anchorage 36, Suez gulf 7, Durban 23, False Bay 1 — ~549 boxes + 159 hard-negative chips, capped by `build_dataset.py --max-neg-ratio 1.0`), **Kornati (156 boxes) held out as the 4th TEST holdout**. Gated eval old@1024 vs new@1024 (conf 0.25): recall up on all four holdouts (Kornati 0.372→**0.513**, Alonnisos 0.671→**0.803**, Port Said 0.690→**0.786**, SG 0.731→0.769); precision down at fixed conf everywhere (Kornati 0.734→0.567 the largest). `conf_sweep`: Alonnisos F1 peak 0.637→**0.704** (@0.30) and NEW@0.25 beats the old deploy on P *and* R *and* FP/chip there; Kornati F1 peak 0.560 vs old 0.573 (statistical parity, CIs overlap); **deploy conf moved 0.20→0.25** (the recall-leaning point, same +0.05 pattern as the last promotion). Watch items: Kornati precision (FP/chip 3.81 @0.25 vs old-deploy 3.31) and the gbr FP probe 15→19. No forgetting: base-test mAP50 0.594→0.614. Command: `train.py --weights data/training/yolov8n-p2.yaml --pretrained best --imgsz 1024 --batch 4 --mosaic 0.5 --close-mosaic 15 --epochs 30 --name finetune-diverse`.

⁸ `finetune-adriatic` (LIVE) = P2 arch warm-started from `diverse`, after folding in the **2026-08-02 W1 Adriatic harvest**: 3 in-class scenes → TRAIN/VAL (Šibenik/Primošten 262 boxes, Hvar/Pakleni 483, Göcek/Fethiye 348 — **1093 boxes over 48 chips, 8 hard-negative**; train 2512→2725), Kornati still held out. **Deploy conf moved DOWN 0.25→0.20** (`conf_sweep.py`; the model is more conservative per-box). Gated on all four holdouts, old@0.25-deploy vs new@0.20-deploy: Kornati R 0.513→**0.590** (P 0.567→0.558), Alonnisos 0.803→**0.855**, Port Said 0.786→0.786, SG 0.769→0.731; open-water probe 1.43→**0.79** dets/chip; base mAP50 0.614→**0.640**, base R 0.585→0.590 (no forgetting). **The acceptance test was NOT met** — Kornati's CI [0.50, 0.68] overlaps the old [0.42, 0.60], and SG regressed — but it was promoted deliberately for the ~45% false-alarm reduction (STATUS open problem #1) plus the clean base-set retention. **The Kornati recall CEILING did not move** (conf-0.05 recall 0.904→0.891); three schedules moved aggregate 4-holdout F1 by ±0.002. Command: `train.py --weights data/training/yolov8n-p2.yaml --pretrained best --imgsz 1024 --batch 4 --mosaic 0.5 --close-mosaic 15 --epochs 30 --name finetune-adriatic`.

## Current training data (the live model — 24 reviewed scenes + Finland-s2)

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
| `2026-08-02T00-51-44Z_sibenik-primosten` | Šibenik, Croatia | 262 | TRAIN (W1 Adriatic-class) |
| `2026-08-02T00-52-02Z_hvar-pakleni` | Hvar/Pakleni, Croatia | 483 | TRAIN (W1 Adriatic-class) |
| `2026-08-02T00-52-13Z_gocek-fethiye` | Göcek/Fethiye, Turkey | 348 | TRAIN (W1 Adriatic-class) |

Four clean held-out scenes anchor eval: Singapore Jul-5 (cross-date), Port Said (cross-region),
Alonnisos (cross-type small vessels), **Kornati (cross-region small vessels, added 2026-07-26)**.
TRAIN carries ~1,300+ real reviewed boat boxes across big-ship + small-vessel domains and 11 regions.

---

## Open problems (priority order)

1. ~~On-water false alarms are the dominant precision problem~~ **LARGELY ADDRESSED 2026-08-02.** The W1 Adriatic harvest cut the open-water FP probe **~45%** (1.43→0.79 dets/chip @0.20) and lifted precision on every holdout at matched recall. Land was only ~16% of FPs, so this came from *positives in cluttered in-class water*, not from masking or from negatives. **Consequence: W4 (O3 same-region hard negatives) is now lower priority** — this cycle delivered most of what it targeted. Residual on-water alarms remain the largest FP class; O7 AIS fusion (built) is the downstream filter.
2. **Recall is GSD-capped.** Sentinel-2's 10 m renders sub-15 m vessels as 1–5 px specks — a physical floor. **Structural fix = Sentinel-1 SAR (Phase 2) + the AIS-mismatch reframe (Phase 3)**, both behind the built `sat_fetch.py` provider seam / fusion layer.
3. **Big-ship recall gap (0.69)** — a training-coverage gap (2 scenes), addressed by the staged Finland set + more dense-anchorage scenes; not the GSD limit.
4. **Cross-region small-vessel recall is GSD-capped, not data-capped** (revised 2026-08-02). Kornati now sits at R 0.590 / P 0.558 @0.20. The 2026-07-26 read — "more same-class regions are the lever" — was **tested and refuted**: W1 added 1093 in-class Adriatic/karst boxes across 3 scenes and 3 training schedules, and the recall *ceiling* did not move (conf-0.05 recall 0.904→0.891; aggregate 4-holdout F1 ±0.002). It bought precision instead. **Do not spend another optical review cycle here expecting recall** — see the ROADMAP "Do not re-attempt" table. The remaining Kornati levers are structural: SAR (W2) and AIS fusion (W3).
9. **`eval_holdout.py` gates ONE holdout by default** (found 2026-08-02; **partially mitigated 2026-08-03**). `--holdout` defaults to `fallbacktest`, so a bare invocation — including the one the ROADMAP W1 recipe used to print — silently evaluates Singapore's 26 boxes and reports nothing about Kornati/Alonnisos/Port Said. This nearly caused a promotion decision on 1/4 of the evidence. The W1 recipe is fixed to loop `--holdout`. W5 closed the *worst* variant — a filter matching **zero** chips is now a hard error listing the scenes that are present, so a wrong `--holdout` can no longer report `P=R=0.000` over nothing — but a filter that matches the *wrong real scene* is still silent. Still worth making the default explicit or accepting several `--holdout` values.
5. ~~CPU-only training~~ **RESOLVED 2026-07-26:** venv torch is now 2.12.1+cu130 on the RTX 4070 (`train.py --device` auto-detects CUDA); a 30-epoch 1024px P2 run takes ~25 min.
6. **TLS interception** — every fetch needs the CA-bundle + `--insecure-tls` recipe (below), and the auto-mode classifier blocks `--insecure-tls` until authorized in-session.
7. ~~Minor/cosmetic: stale `BoatDetection` pointers~~ **RESOLVED 2026-07-29:** `export_labels.py`'s header now names `build_dataset.py` → `train.py`, and `detect_boats.py`'s weights-not-found error points at `data/training/weights/` instead of the legacy sibling repo.
8. ~~**The SAR model has no eval harness**~~ **RESOLVED 2026-08-03 (ROADMAP W5)** — and resolving it
   **downgraded the SAR model**. `eval_holdout.py`/`conf_sweep.py` now take `--train-dir`
   (`data/training_sar`), with `PROMOTED_BY_TREE` binding each tree to its checkpoint so a SAR gate
   can never silently borrow the optical `best.pt`. The harness reproduces the recorded Gibraltar gate
   exactly (P 0.392 / R 0.478 / mAP50 0.335) — **but that pair sits at conf ≈0.10**, where the
   in-sample Fujairah probe fires **24.9 dets/chip**, and the 90% recall CI is **[0.27, 0.67]** on 23
   boxes. So `best_sar.pt` has **no swept deploy conf and no usable operating point yet**; every
   optical model is quoted at one, this never was. **New open item → see #10.**
10. **`best_sar.pt` has no usable operating point** (found 2026-08-03 by the new harness). F1 peaks at
   **0.415 @ conf 0.10** (P 0.367 / R 0.478); quieting the false alarms costs nearly all recall
   (R 0.217 @ 0.25), and the recall ceiling is ~0.70 @ conf 0.05. *Caveat: the Fujairah probe is a busy
   anchorage in TRAIN, so it is directional and includes real vessels — not comparable to the optical
   near-empty probes at 0.79 dets/chip.* This is a **model** problem, not a threshold one: the fix is
   W2 (more SAR regions / xView3 transfer), not a conf tweak. Also note **all 23 SAR GT boxes are
   >5 px**, so the `1-2px`/`3-5px` strata are empty — they are an optical 10 m-GSD construct and need
   SAR-specific bin edges to say anything here.

## Recommended next steps

**Everything runnable is run — and as of 2026-08-02, W1 is closed too.** Phase 3 is live-validated
end-to-end, the GPU is unlocked, all AUDIT P1/P2 findings are closed, and the W1 Adriatic cycle ran to
a promotion. **The headline change to the plan: more optical data is no longer the top lever.** W1
proved the Kornati recall ceiling is the 10 m GSD floor, not a data gap, so the queue's centre of
gravity moves to the structural items — **W2 (SAR)** and **W3 (live AIS loop)** — with **W4 downgraded**
(its precision target was largely met as a W1 side effect).

**The full queue with commands, acceptance tests and traps lives in `ROADMAP.md` → THE WORK QUEUE.**
Summary in impact order:

1. ~~**W1 — Attack the Kornati gap**~~ ✅ **DONE 2026-08-02.** Ran in full (3 scenes / 1093 boxes /
   3 schedules); `finetune-adriatic` promoted at conf 0.20 for a ~45% false-alarm reduction. The recall
   premise was **refuted** — Kornati's ceiling is GSD, not data. Full write-up in `ROADMAP.md` → "W1
   outcome". **Do not repeat this cycle expecting recall.**
2. **W2 — SAR needs regions, not one more scene.** `sar-deep2` (+Singapore) regressed the Gibraltar
   gate → next attempt only after 2+ more reviewed deep-water regions, or the xView3 transfer.
   `best_sar.pt` = `sar-deep` unchanged; it scores **0 on unseen regions** (per-scene dB stretch), so
   seed new SAR reviews with `sar_seed.py --water-only --min-depth 12`. **W5 is done, so the gate is
   ready** — and it showed the incumbent is weaker than recorded (no usable operating point, CI
   [0.27, 0.67]; open problems #8/#10). Two consequences: a `sar-deep3` has a low bar to clear, and
   clearing it proves less than it looks. **Prioritise a second reviewed region in TEST**, which fixes
   the model *and* the gate's power at once.
3. **W3 — Operational mode.** The **AIS-only** loop (`aisstream_fetch` → `ais_fishing` →
   `geofence_ais` → `alert_violations`) needs *only* `AISSTREAM_API_KEY` — no imagery, no overpass
   timing — so it is now the fastest path to a live result. The full fusion variant additionally needs
   a fresh overpass inside the AIS capture window.
4. ~~**W5 — `eval_holdout.py --train-dir`**~~ ✅ **DONE 2026-08-03.** Both eval scripts are now
   modality-agnostic (`--train-dir`, `PROMOTED_BY_TREE`), zero-match holdouts and zero-chip FP probes
   are hard errors / warnings instead of silent zeros, and `ultralytics` is lazy-imported so the gate's
   logic carries **50 offline tests** (suite 203→**253**). Optical numbers re-verified unchanged.
   Full write-up: `ROADMAP.md` → "W5 outcome"; the SAR fallout is open problems #8/#10.

**Hygiene round landed 2026-07-29 (before resuming the data loop):** AUDIT #4 (network retry/backoff) is
closed — one shared policy in `scripts/_http.py` wired into every `requests` call site, plus
`GDAL_HTTP_MAX_RETRY`/`RETRY_DELAY` on the `/vsicurl` COG reads, so the fetch-heavy work in steps 1–2
above stops losing a run to a transient 5xx. SAR runs now land in their own `data/raw/sentinel1/` root.
Not covered: websocket reconnect for `aisstream_fetch.py` (a capture-semantics decision, deliberately
left open), and retries do **not** help the Avast schannel wall.

**Three offline product stages landed 2026-07-29** (the sandbox-buildable Tier-1 roadmap items), taking
the suite to **203 offline tests**:
- **`ais_fishing.py`** — infers fishing from AIS *movement* (speed band + sinuosity) and emits the
  **same vessel-records contract** as `gfw_vessels.py` (`source: "ais-heuristic"`), so `--vessels` reads
  it unchanged. The fishing term in `violation_score` now works with **no GFW token and no network**.
- **`geofence_ais.py`** — flags cooperative vessels transmitting from inside the MPA: a third status
  **`reported`** beside `dark`/`matched`, in the same `violation_events` schema. Scored by *behaviour*
  (transit floor 0.10 → fishing weight) rather than presence, so legal transit does not drown the signal.
  Pure-stdlib point-in-polygon; verified against the real Anacapa SMR polygon.
- **`alert_violations.py`** — the promised alerting interface: severity digest (critical/high/medium/low,
  IUU forces critical), MPA-grouped, console + markdown, `--exit-code` cron gate. Reads fusion **and**
  geofence events in one digest.

One provenance bug was found and fixed in passing: `fuse_violations.build_event` hardcoded `"GFW"` as
the evidence tag for *any* joined vessel record, so a locally-derived fishing prior would have claimed
GFW provenance. It now derives the label from the record's `source` (`vessel_evidence_label`).

---

## Operational recipes

**Fetch imagery under Avast TLS interception** (public read-only COGs; `--insecure-tls` is user-authorized each session — see `[[sat-fetch-tls-avast]]`):
```bash
# 1. build combined CA bundle (certifi + Avast root) once, e.g. via PowerShell.
#    ⚠️ AVAST REGENERATES THIS ROOT PERIODICALLY - the bundle goes stale and every HTTPS call fails
#    with "unable to get local issuer certificate". Do NOT hardcode a thumbprint; look it up:
#      $c = Get-ChildItem Cert:\LocalMachine\Root | ? { $_.Subject -match 'Avast' }
#      $pem = "-----BEGIN CERTIFICATE-----`n" +
#             [Convert]::ToBase64String($c.RawData,'InsertLineBreaks') +
#             "`n-----END CERTIFICATE-----`n"
#      Set-Content ca_avast_bundle.pem ((Get-Content (python -c "import certifi;print(certifi.where())") -Raw) + $pem) -Encoding ascii
#    Rotation history: CECD313F...DA80 (thru 2026-07) -> 27469C7BA79BA957EA60AAF214F47B72BE844D75
#    (current 2026-08-02). Live bundle: scratch_logs/ca_avast_bundle_2026-08.pem
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
# back up the outgoing model first — the backups in weights/ are the audit trail
cp data/training/weights/best.pt data/training/weights/best_<outgoing-run-name>.pt
cp data/training/runs/<run>/weights/best.pt data/training/weights/best.pt
```

**The SAR training flow** (undocumented elsewhere — reconstructed from `data/training_sar/`'s manifest
2026-08-01). Both `build_dataset.py` and `train.py` are modality-agnostic via `--data`/`--exports`;
there is no separate SAR script. The SAR set is deliberately a **separate dir tree** so the optical
`build_dataset.py` default can never sweep SAR chips into `data/training/`:
```bash
# exports live in data/processed/sar_training_exports/<tag>/  (tags so far: deep, singapore)
python scripts/build_dataset.py \
    --exports data/processed/sar_training_exports \
    --data data/training_sar \
    --holdout-scene s1deep-gibraltar
# P2 arch, yolov8n backbone transfer — NEVER --pretrained best (optical texture priors are wrong)
python scripts/train.py --weights data/training/yolov8n-p2.yaml --pretrained yolov8n \
    --data data/training_sar/config.yml --imgsz 1024 --batch 4 \
    --mosaic 0.5 --close-mosaic 15 --epochs 30 --name sar-deep3
# NOTE: runs still land under data/training/runs/<name>/ regardless of --data.
# Gate it with the SAME tiny-object harness the optical models get (W5, 2026-08-03):
venv/Scripts/python.exe scripts/conf_sweep.py --train-dir data/training_sar \
    --holdout s1deep-gibraltar --fp-scene s1deep-fujairah --imgsz 1024 \
    --weights data/training/runs/sar-deep3/weights/best.pt      # sweep the INCUMBENT too
venv/Scripts/python.exe scripts/eval_holdout.py --train-dir data/training_sar \
    --holdout s1deep-gibraltar --fp-scenes s1deep-fujairah --imgsz 1024 --conf <swept> \
    --old data/training/weights/best_sar.pt \
    --new data/training/runs/sar-deep3/weights/best.pt
# --old/--new default to best_sar.pt for this tree (eval_holdout.PROMOTED_BY_TREE), so the optical
# best.pt can never leak into a SAR gate. Omit both to just re-measure the incumbent.
cp data/training/runs/sar-deep3/weights/best.pt data/training/weights/best_sar.pt   # if it gates
```
⚠️ **Report a SWEPT conf, never ultralytics' `val` pair.** The incumbent's recorded "P 0.392 / R 0.478"
is the conf-≈0.10 point (24.9 dets/chip on the in-sample Fujairah probe), not a deployable one — see
open problems #8/#10. And the gate is thin: recall CI **[0.27, 0.67]** on 23 boxes, so a small win is
noise. A second reviewed region in TEST buys more gate power than any tuning.
