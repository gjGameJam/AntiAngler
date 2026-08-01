# IMPROVEMENT PLAN — optical (Sentinel-2) & radar (Sentinel-1 SAR) detectors

_Created 2026-07-16, **refreshed 2026-08-01**. Two **separate** models (different sensors, different
failure modes): the optical `best.pt` (`finetune-diverse`, live at imgsz 1024 / conf 0.25) and the SAR
`best_sar.pt` (`sar-deep`, live but thin). This plan targets **accuracy** — precision (kill false
positives) and recall (catch real vessels) — grounded in the 2026-07-15/16 optical-vs-SAR review.
Priorities are ordered by (impact ÷ effort)._

**Where this file sits:** `docs/STATUS.md` = live model numbers · **`docs/ROADMAP.md` → THE WORK QUEUE
= the prioritized plan with commands** · this file = the *why* and the measurement detail behind each
lever. The W-items in ROADMAP map onto the levers here: **W1 → O4**, **W4 → O3**, **W2 → S1/S2/S4**,
**W5 → O6 (SAR)**, **W6 → S6**, **W7 → S7**.

_**Status of the levers (2026-08-01).** Closed: **O1** water mask (shipped), **O2** NDWI (built — a
**calibrated non-win**, off by default), **O5** inference tuning (measured — imgsz 1024 confirmed, 1280
a non-win), **O6** eval harness (exercised; a 2nd small-vessel holdout — Kornati — now exists), **O7**
the AIS-mismatch join (built AND live-validated). SAR **S1–S5 are done** for one region-family:
`best_sar.pt` exists, trained on `data/processed/sar_training_exports/` through the same review loop.
**Still open: O3** (same-region hard negatives), **O4** (more diverse positives — now specifically
Adriatic-class, see the Kornati result below), **S6** (speckle A/B), **S7** (cross-modality), and the
**SAR generalization problem** S1–S4 exposed._

## What the review actually showed (the diagnosis)

- **Optical false positives cluster at coastlines / land / nodata edges** (Cyclades detail tile: a
  pile of boxes on the shore). Root cause: `detect_boats.py` only masks to the **MPA polygon** for
  WDPA runs (the `inside_mpa` tag) — at review time there was **no general water mask** (O1 below
  shipped one), so bright land, surf,
  wakes, foam, sun-glint and thin cloud all get flagged. This is the #1 visible "inaccuracy."
- **Optical misses small vessels** — the 10 m GSD renders a sub-15 m boat as 1-5 px. Real, but a
  *physical* ceiling, not a bug. Recall is a candidate-generation number, not a promise.
- **SAR "inaccuracies" are mostly not-a-model-yet + modality limits:** big metallic hulls light up
  cleanly (Fujairah/Singapore), but small non-metallic boats barely return (Cyclades), and **land/coast
  speckle is a strong clutter/FP source**. There is no `best_sar.pt` yet, so any SAR "detections" today
  are the *optical* model misapplied. (Also: circles over the black no-data wedge in the Fujairah SAR
  panel are just detections plotted where the swath has no data — not errors.)

**Guiding principle (already baked into `detect_boats.py`'s low `--conf` default): recall-favouring candidate generation, then
filter false positives downstream** (water mask now, AIS-mismatch in Phase 3). So: lift recall where
physics allows, and attack precision with *filters* that cost little recall — not by raising conf
(which trades recall away) or piling on hard negatives (which made `finetune-reef` too conservative).

---

## Step 0 — Quantify the FP mix ✅ DONE (2026-07-16)

Ran the live model at the deploy point (imgsz 1024, conf 0.15) on the three holdouts; matched preds to
GT by center distance; classified every false positive land-vs-water via the new WorldCover mask
(`scripts/water_mask.py`) and eyeballed the on-water ones.

| Holdout | chips | TP | FP | FN | P | R | FP-land | FP-water |
|---|---|---|---|---|---|---|---|---|
| fallbacktest (Singapore Jul-5) | 4 | 21 | 10 | 5 | 0.68 | 0.81 | 0 | 10 |
| portsaid (dense anchorage) | 12 | 29 | 1 | 13 | 0.97 | 0.69 | 0 | 1 |
| alonnisos (small vessels/islands) | 64 | 33 | 99 | 16 | **0.25** | 0.67 | 18 | 81 |
| **TOTAL** | | | **110** | | | | **18 (16%)** | **92 (84%)** |

**The result overturned the initial hypothesis.** Land/coast is only **16%** of FPs; **84% are
ON-WATER**, and they are overwhelmingly concentrated on the **small-vessel scene** (Alonnisos, P 0.25).
Port Said (big ships, 100% water) is already P 0.97 — big-vessel precision is a solved problem.

Visual categorization of the on-water FPs (Alonnisos): mostly **empty-water false alarms** (the model
fires on featureless/textured water at conf 0.15), **swell/current lines** (mistaken for wakes),
**shallow-water/reef colour boundaries**, and a real minority of **unlabeled small boats** (so the true
precision is a bit above 0.25 — a labeling-completeness artifact). The land FPs are confident
rock/islet hits (several at conf 0.6-0.7) — the most visually damaging, and exactly what O1 removes.

**→ Revised priority:** O1 (water mask) is a cheap, worth-it partial win (kills the ~16% land FPs,
including the confident rock/islet ones, and is a SAR prerequisite) but is **NOT** the main lever.
**The dominant precision problem is on-water false alarms on small-vessel scenes → O3 (same-region
hard negatives) is the top precision lever**, and much of the residual is a labeling gap (relabel
Alonnisos completely) + the O7 AIS join. Recall (O4) stays the parallel track.

---

## Part 1 — Optical (image) model

| # | Lever | Attacks | Effort | Expected |
|---|---|---|---|---|
| O1 | Water/land mask at inference ✅ **DONE** | precision | low | ships; removes the ~16% land FP class (incl. confident rocks/islets) |
| O2 | NIR / NDWI (mask now, 4-ch model later) | precision (+recall) | med→high | (a) NDWI filter BUILT but a **calibrated non-win** on small-vessel FPs (glint≠blank water); (b) 4-ch model deferred |
| O3 | Same-region hard negatives (glint/coast/cloud) | precision | med | fewer FPs; **watch recall** |
| O4 | More diverse real positives (proven engine) | recall | med (ongoing) | the durable recall lever |
| O5 | Inference/tiling tuning ✅ **MEASURED** | recall | low | imgsz 1024+conf 0.15 confirmed; **1280 a non-win** (see below) |
| O6 | Eval hardening ✅ **harness exercised** (2nd small-vessel holdout still 🏠) | trust | med | fresh deploy-point CIs on both holdouts |
| O7 | Phase 3 AIS-mismatch join ✅ **BUILT** | precision (product) | high | the ultimate FP filter — `fuse_violations.py` (Phase 3) exists; wire runs through it |

### O1 — Water/land mask at inference ✅ **SHIPPED (2026-07-16)**
- **What shipped:** `scripts/water_mask.py` fetches **ESA WorldCover 10 m** (free, same PC STAC, class
  80 = water) over a run's AOI and caches it as `water_mask.tif` (mosaics multi-tile AOIs). `detect_boats.py`
  gained `--water-only`: it tags every detection `on_water` from the cached mask and drops land ones
  (mirrors the `inside_mpa` filter). Offline after the one-time fetch. **Shared with SAR S3.**
- **Demonstrated:** Alonnisos run 76 → 61 detections (dropped 15 on land: rocks/islets/coast), incl. the
  confident conf-0.6-0.7 rock FPs. Step 0 confirms this is ~16% of all FPs — a real but partial win.
- **Deliberately NOT the main lever** (Step 0): most FPs are on-water. Kept opt-in (`--water-only`), so
  the recall-leaning default is unchanged. **Follow-ups:** a small shoreline *dilation* would catch
  sub-pixel islets the 10 m mask misses; and the same mask feeds SAR (S3).

### O2 — NIR band / NDWI
- **Why:** detection is RGB (B04/B03/B02) only. NIR (**B08**) darkens water and makes hulls/wakes pop;
  **NDWI = (Green−NIR)/(Green+NIR)** cleanly separates water from land/foam → it *is* the O1 mask, for free.
- **How (staged):** (a) **mask-only, no retrain:** have `sat_fetch` also read B08, compute NDWI, store a
  per-chip water mask in the manifest → O1 consumes it. (b) **later, 4-channel model:** feed NIR as a 4th
  input channel and retrain (touches provider band packing, `detect_boats` input, and augmentation).
- **Recommend:** do (a) first (feeds O1); defer (b) until O1/O3/O4 plateau.

- **⚠️ (a) BUILT + CALIBRATED — validated NON-WIN for small-vessel precision (2026-07-18).** Shipped
  `scripts/ndwi_mask.py` (re-reads the run's scene B03/B08 through the `pc` provider seam, caches a
  pixel-aligned float `ndwi.tif`) + `detect_boats.py --open-water`/`--ndwi-thresh` (drops a detection only
  if its whole box is water, min NDWI ≥ thresh; also stamps a `min_ndwi` diagnostic on every detection).
  End-to-end verified (synthetic sampler test + real fetch on Santos & Alonnisos under the Avast bundle).
  **But the core premise — "on-water FPs sit on blank water, real boats don't" — is FALSE on the
  small-vessel scene it targets.** Calibrated on the Alonnisos holdout (63 TP / 20 FP, human verdicts):
  the FPs sit on **glint/foam/swell/reef-edge features that reflect NIR *more strongly* than the 1–5 px
  real boats** (FP min-NDWI median **−0.122** vs TP **−0.027**), so the distributions overlap the wrong
  way — no threshold drops FPs without gutting recall (at −0.10: 7/20 FP dropped but **60/63 TP** dropped,
  precision 0.76→0.19; at ≥+0.05: ~0 FP dropped). **Kept opt-in and OFF by default** (correct, tested, and
  `min_ndwi` is a useful diagnostic), but small-vessel precision stays with **O3 (same-region hard
  negatives)** and **O7 (AIS fusion)**, exactly as Step 0 concluded. May still help big-ship/anchorage
  scenes (genuine blank-water FPs), but those are already ~P0.97 so the ceiling is small. Lesson: NDWI is a
  spectral *water* index, and these FPs are real spectral water *features*, not blank water — so a water
  index cannot reject them. The 4-channel model (b) would have to *learn* the object-vs-glint texture
  difference; a single NDWI threshold cannot.

### O3 — Same-region hard negatives
- **Why:** the small-vessel retrain doubled FP/chip (0.70→1.55, STATUS). The fix is reviewed
  empty-water / coast / cloud / glint chips **from the same regions**, exported as hard negatives
  (empty labels) via the loop.
- **Caution:** `finetune-reef` showed too many negatives → over-conservative → recall drop. Cap the
  negative:positive ratio and re-check recall every retrain.
- **How:** fetch empty-water/coastal/thin-cloud chips near deployment regions → `review_server.py` →
  `export_labels.py` (empty labels) → `build_dataset.py` → `train.py`. **Verify:** precision ↑, recall
  held on all holdouts.
- **Safeguard BUILT (2026-07-18): `build_dataset.py --max-neg-ratio R`.** Caps AL hard-negative
  (empty-label) chips in the **TRAIN** split to `R × (AL positives in train)`, seeded-subsampling the
  excess (val/holdout negatives are never capped — they only measure precision). `build_dataset.py` now
  also **always reports the pos/neg balance per split** and records it in `active_learning_manifest.json`,
  so negative flooding is visible. Note (measured 2026-07-18): with the Finland set staged, train neg:pos
  is only **~0.09**, so flooding is *not* a current risk and the cap won't bind at reasonable `R` — it
  bites only when negatives genuinely dominate (e.g. a Finland-free retrain, or after adding many more
  empty-water chips). The discipline that matters is **re-checking holdout recall after every retrain that
  adds negatives**, not a specific ratio.

### O4 — More diverse real positives (the proven engine — highest recall leverage) → **ROADMAP W1**
- **Why:** real diverse data closed every generalization gap it was pointed at — **four times**
  (cross-region via open-ocean, cross-type via small-vessel Med, big-ship via Gulf-of-Finland, broad
  diversity via the 13-set harvest).
- **⚠️ The 2nd small-vessel holdout landed 2026-07-26, and it falsified the optimistic reading.**
  Kornati (Adriatic, 16 chips / **156 human boxes**, never in TRAIN) scored only **R 0.37** on the
  then-live `finetune-bigship` — the Alonnisos 0.67 **did not transfer cross-region**.
  `finetune-diverse` lifts it to **R 0.51 / P 0.57 @0.25** (CI [0.42, 0.60]), still far below
  Alonnisos's 0.80. **Conclusion: recall generalizes within a region-class, not across one.** Small
  vessels in Aegean water are not the same problem as small vessels on a karst coast.
- **Remaining gaps, most valuable first:** **Adriatic/karst-coast** (the measured hole — W1),
  smaller **artisanal/pleasure** boats, and other **sea-states/latitudes** (Atlantic, tropical, high-lat).
- **How:** the in-repo loop, holding out each fresh scene *before* folding it in. **Kornati stays a
  pure holdout** — never fold it in, or the only cross-region small-vessel gate is lost.
- **Watch item:** Kornati precision (FP/chip **3.81** @0.25 vs the old deploy's 3.31). If recall
  climbs while precision craters, that is the **O3** signal, not a reason to reject the model.

### O5 — Inference / tiling tuning ✅ **MEASURED (2026-07-25) — imgsz 1024 + conf 0.15 confirmed; 1280 a non-win**
- **imgsz 1280 (plain, no TTA) does NOT help** — measured on the live `best.pt` (finetune-smallvessel) at
  conf 0.15 on both holdouts (`eval_holdout.py`, center-distance matching):
  - **Alonnisos** (small-vessel, 76-box relabeled): 1024→1280 **recall 0.70→0.49** (WORSE), precision flat
    (0.40→0.41), FP/chip 1.23→0.83 — upscaling past the 1024 train size *loses* the 1–5 px vessels it was
    meant to enlarge (>5px stratum 0.69→0.47). A clear net loss.
  - **Port Said** (big-ship, 42-box): 1024→1280 **recall 0.69→0.69, precision 0.97→0.97, FP/chip 0.08→0.08
    — IDENTICAL**. So the Port Said lift STATUS reports (0.69→0.81) is entirely from **`--augment` (TTA:
    multi-scale + flips)**, *not* from raw imgsz. Raising imgsz alone buys nothing on big ships either.
  - ⇒ **Keep imgsz 1024 as the deploy default; 1536 not worth testing** (1280 already shows zero/negative
    return, and it is 1.6× slower). The big-ship recall lever is TTA (already offered as the opt-in
    `detect_boats --imgsz 1280 --augment` big-ship profile), not tile/inference size.
- **conf operating point confirmed:** `conf_sweep.py` on the relabeled Alonnisos (64 chips / 76 boxes,
  imgsz 1024) puts the **F1 peak at conf 0.15 (F1 0.510)** — adjacent 0.10→0.470, 0.20→0.406. Recall-vs-conf
  is steep (R 0.99@0.05 → 0.70@0.15 → 0.38@0.20), FP-vs-conf shallower, so 0.15 stays the recall-favouring
  candidate-generation operating point. **No deploy-conf change.**
- **merge-dist:** eval matched at center-distance 8 px (80 m); no double-counting seen in the holdout GT. A
  full `detect_boats.py --merge-dist` sweep on a live scene needs 🏠 imagery (do-from-home) — deferred.

### O6 — Eval hardening ✅ **HARNESS EXERCISED (2026-07-25) — fresh deploy-point numbers on the relabeled holdouts**
- **Deploy-point P/R at the operating point (conf 0.15 / imgsz 1024), with bootstrap 90% CIs and FP strata,
  re-measured on the relabeled holdouts** (the harness `eval_holdout.py` already reports all of this):

  | Holdout | chips | boxes | P | R | FP/chip | recall 90% CI |
  |---|---|---|---|---|---|---|
  | Alonnisos (small-vessel) | 64 | 76 | 0.40 | 0.70 | 1.23 | [0.59, 0.79] |
  | Port Said (big-ship) | 12 | 42 | 0.97 | 0.69 | 0.08 | [0.57, 0.79] |

  Both reproduce STATUS's post-relabel baseline (Alonnisos 0.40/0.70, Port Said 0.97/0.69); FP/chip on
  Alonnisos is now **1.23** (vs the 1.55 STATUS recorded pre-relabel — relabeling reclassified real boats
  that were counted as FPs). These are the honest baselines O3 (hard negatives) must beat.
- **Still open (needs 🏠 fetch/GPU/review, not runnable in this sandbox):** a **2nd small-vessel holdout**
  to confirm the 0.70 Alonnisos magnitude (still one 64-chip scene), broader **region diversity**, and a
  finer **FP-by-Step-0-cause** strata (glint/swell/reef vs land) — the last needs the WorldCover/NDWI masks
  over a fresh scene. The bootstrap-CI + size-strata reporting the harness already emits is sufficient for
  attributable deltas on the existing holdouts.

### O7 — Phase 3 AIS-mismatch (strategic, the real precision ceiling) ✅ **BUILT (2026-07-23)**
- **dark-vessel = detection − AIS** filters out FPs that sit on a matching AIS track and keeps real dark
  vessels. This *inverts* the metric from recall (GSD-capped) to mismatch precision (improvable). It's the
  payoff that makes a modest detector "good enough" — see `docs/ROADMAP.md` Phase 3. Complementary to
  O1-O4, not a substitute.
- **Status: ✅ BUILT AND LIVE-VALIDATED (2026-07-26).** `fuse_violations.py` (matcher) + the AIS
  ingesters (`aisstream_fetch.py`, `marinecadastre_fetch.py`) + the GFW join (`gfw_vessels.py`), plus
  the 2026-07-29 additions (`ais_fishing.py`, `geofence_ais.py`, `alert_violations.py`). Real
  violations produced over Anacapa Island SMR. **The FP-filtering endgame is available today** over any
  run + AIS file — and `ais_fishing.py` means the fishing term works with **no GFW token and no
  network**. Remaining: the near-real-time operational mode (ROADMAP **W3**).

**Recommended optical sequence:** Step 0 → **O1 (+O2a NDWI mask)** → **O3** → **O4 ongoing** → O6 → O5,
with O7 as the strategic overlay. O1, O2, O5, O6 and O7 are all now closed (see the status block at the
top), so **the remaining optical accuracy work is exactly the two data-driven levers**: **O4** (diverse
positives — specifically Adriatic-class → ROADMAP **W1**) and **O3** (same-region hard negatives →
**W4**). Both are 🏠 do-from-home (fetch + GPU + human review). **Order matters: O4 before O3** — new
positives change the FP mix, so negatives mined first would target a distribution that no longer exists.

> ⚠️ **Reading the conf numbers below.** O5/O6 were measured **2026-07-25 on `finetune-smallvessel`**,
> whose operating point was **conf 0.15**. Two promotions have happened since, and the deploy conf has
> moved **+0.05 each time** (0.15 → 0.20 → **0.25**, the live value). The *conclusions* still hold
> (imgsz 1024 is right, 1280 is a non-win, the F1 peak is found with `conf_sweep.py`); the *absolute
> conf and P/R figures* in O5/O6 are historical. Live numbers are in `docs/STATUS.md`.

---

## Part 2 — Radar (SAR) model — `best_sar.pt` exists; the open problem is **generalization**

**Scope decision from the review: target LARGER / metallic vessels (the dark-ship / AIS-off mission).**
Do not expect SAR to match optical on small artisanal boats — C-band barely returns them. Budget for
speckle + coastal clutter as the dominant FP sources.

> **Status 2026-08-01.** `best_sar.pt` = **`sar-deep`** (promoted 2026-07-26): P2 arch, `yolov8n`
> backbone transfer, 30 epochs in 76 s on the RTX 4070, trained on 149 chips / 206 vessel boxes / 112
> hard-negatives from **one region-family** (Fujairah / Gibraltar / Hormuz — warm, deep, Gulf-like).
> Gibraltar test split: **P 0.392 / R 0.478 / mAP50 0.335**.
>
> **The measured problem: it does not generalize across regions.** On a fresh Singapore Strait scene it
> scored **0 hits at conf 0.15** — because `providers/s1.py` applies a **per-scene dB percentile
> stretch**, every new region is a different pixel distribution, and a 206-box model is immediately
> out-of-domain. The obvious fix failed too: retraining with the Singapore scene folded in
> (`sar-deep2`, +267 boxes) **regressed the Gibraltar gate** (P 0.392→0.285, R 0.478→0.435) and was not
> promoted. Singapore data stays staged in `data/training_sar/` for the next attempt.
>
> **⚠️ And the gate itself is weak: `eval_holdout.py` cannot evaluate the SAR set at all** (it
> hardcodes `data/training/`), so that reject decision rested on ultralytics' own metrics over 23 test
> boxes — no center-distance P/R, no FP/chip, no CI. **Fix that first (ROADMAP W5).**

| # | Lever | Attacks | Effort | Status |
|---|---|---|---|---|
| S1 | Training data: xView3-SAR (+HRSID/SAR-Ship) | build | med | ⬜ **the big open lever** (registration-gated) — own labelled set exists instead |
| S2 | Preprocessing alignment to the provider chips | build (critical) | med | ✅ satisfied for own data (chips *are* provider output); **critical if S1 lands** |
| S3 | Land/water mask (shared with O1) | precision | low-med | ✅ `sar_seed.py --water-only` + `--min-depth` (GEBCO) |
| S4 | Train separate P2 SAR model | build | med | ✅ `sar-deep` promoted; **retrain blocked on region diversity** |
| S5 | Real S1 holdout via HITL review | trust | med | ✅ Gibraltar held out — but see the harness gap above |
| S6 | Speckle filter (Lee/Refined-Lee) A/B | precision/recall | low | ⬜ open (**W6**) — may erase 1-5 px targets; test, don't assume |
| S7 | Cross-modality S1×S2 validation | trust | low | ⬜ open (**W7**) — Fujairah/Singapore/Cyclades pairs already on disk |

### S1 — Training data
- **xView3-SAR** (arXiv 2206.00897): Sentinel-1 dual-pol VV/VH, maritime, dark-vessel labels with vessel
  length + a fishing/non-fishing/vessel class + confidence. Filter to vessel classes; optionally by length
  to match the "larger vessel" scope. HRSID / SAR-Ship as supplements. Free (registration).

### S2 — Preprocessing alignment **(the make-or-break step)**
- The SAR detector must be trained on the **exact representation the provider emits at inference**:
  pseudo-RGB `[VV, VH, VV/VH]` after the **dB-percentile stretch** (`providers/s1.py::scale_to_uint8`).
  Run xView3's VV/VH through that **same function** to build the training chips — do not train on raw
  amplitude or a different stretch, or the model sees a different distribution than it's deployed on.

### S3 — Land/water mask
- Reuse the O1 shared water-mask util. Mask SAR to water before/after inference to kill the bright
  land/coast returns seen in the Cyclades panel. Highest-leverage SAR precision move.

### S4 — Train the separate model
- P2 head (stride-4, for small targets), imgsz 1024, `train.py` (supports `.yaml` arch + warm-start).
  Warm-start from a **SAR-pretrained** backbone or train from scratch — **never** from the optical
  `best.pt` (wrong texture priors). Output `data/training/weights/best_sar.pt`; `detect_boats.py --weights`
  already accepts it (no code change).

### S5 — Real S1 holdout
- Fetch a real S1 scene (Fujairah/Singapore/Cyclades already on disk) → `review_server.py` to hand-label →
  a real SAR eval set. Use xView3 val for the transfer number and the real scene for the honest
  deployment number.

### S6 — Speckle filtering A/B
- Try Lee/Refined-Lee before tiling; measure recall on 1-5 px targets both ways — speckle filters can
  smooth away the very targets we want. Keep only if it helps.

### S7 — Cross-modality validation
- Run `best_sar.pt` on the S1 chips and compare against the optical detections over the same AOI (the
  comparison harness from this session). Confirm SAR flags **large/dark vessels the optical missed under
  cloud/night** — the whole point. Feeds Phase 3 fusion.

**Recommended SAR sequence:** S1 + S2 (foundational) → S3 → S4 → S5 → S6 → S7.

---

## Cross-cutting

- **Shared water-mask utility** serves O1 **and** S3 — build once (`scripts/` helper), consume from both
  `detect_boats.py` paths.
- ~~**Fix the SAR run-dir misnomer**~~ ✅ **DONE 2026-07-29:** `Provider.output_subdir` now picks the run
  root (`sentinel2` for `pc`, `sentinel1` for `s1`), so modalities never share a directory. Purely
  additive — pre-2026-07-29 SAR runs still live under `data/raw/sentinel2/` and still work, since every
  downstream stage takes an explicit `--run`.
- **Keep the two training trees separate.** `data/processed/training_exports/` is **optical-only**
  (`build_dataset.py` sweeps *every* tag there into `data/training/`); SAR exports go to
  `data/processed/sar_training_exports/` → `data/training_sar/`. Dropping a SAR export in the optical
  dir would silently poison the optical training set.
- **The active-learning HITL loop is the engine for both models** — every accuracy lever above flows
  through fetch → detect → review → export → build → train → promote, with a held-out scene each round.
- **Phase 3 AIS fusion (O7) is the precision endgame for both** — it filters each model's false positives
  against AIS, which is what ultimately makes 10 m / C-band detection operationally "good enough."
