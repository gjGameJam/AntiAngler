# IMPROVEMENT PLAN — optical (Sentinel-2) & radar (Sentinel-1 SAR) detectors

_Created 2026-07-16. Two **separate** models (different sensors, different failure modes):
the optical `best.pt` (live) and the SAR `best_sar.pt` (not built yet). This plan targets
**accuracy** — precision (kill false positives) and recall (catch real vessels) — grounded in the
2026-07-15/16 optical-vs-SAR review. Read `docs/STATUS.md` for live model numbers, `docs/ROADMAP.md`
for the phase structure this feeds. Priorities are ordered by (impact ÷ effort)._

## What the review actually showed (the diagnosis)

- **Optical false positives cluster at coastlines / land / nodata edges** (Cyclades detail tile: a
  pile of boxes on the shore). Root cause: `detect_boats.py` only masks to the **MPA polygon** for
  WDPA runs (`inside_mpa`, :200-215) — there is **no general water mask**, so bright land, surf,
  wakes, foam, sun-glint and thin cloud all get flagged. This is the #1 visible "inaccuracy."
- **Optical misses small vessels** — the 10 m GSD renders a sub-15 m boat as 1-5 px. Real, but a
  *physical* ceiling, not a bug. Recall is a candidate-generation number, not a promise.
- **SAR "inaccuracies" are mostly not-a-model-yet + modality limits:** big metallic hulls light up
  cleanly (Fujairah/Singapore), but small non-metallic boats barely return (Cyclades), and **land/coast
  speckle is a strong clutter/FP source**. There is no `best_sar.pt` yet, so any SAR "detections" today
  are the *optical* model misapplied. (Also: circles over the black no-data wedge in the Fujairah SAR
  panel are just detections plotted where the swath has no data — not errors.)

**Guiding principle (already in `detect_boats.py:59`): recall-favouring candidate generation, then
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
| O5 | Inference/tiling tuning | recall | low | small, diminishing |
| O6 | Eval hardening (2nd small-vessel holdout, FP strata) | trust | med | trustworthy deltas |
| O7 | Phase 3 AIS-mismatch join | precision (product) | high | the ultimate FP filter |

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

### O4 — More diverse real positives (the proven engine — highest recall leverage)
- **Why:** real diverse data closed both generalization gaps (cross-region, cross-type) — twice. Remaining
  gaps: **smaller artisanal/pleasure boats**, **other regions/sea-states** (Atlantic, tropical, high-lat),
  and a **2nd small-vessel holdout** to confirm the 0.67 magnitude.
- **How:** the in-repo loop, holding out each fresh scene *before* folding it in; keep one rotating holdout.

### O5 — Inference / tiling tuning
- imgsz 1024 is the dominant lever, already spent. Test 1280/1536 (compute vs marginal recall). Tune conf
  per region if Step 0 shows region-dependent FP rates. Verify merge-dist (30 m) isn't double-counting.

### O6 — Eval hardening
- Add a 2nd small-vessel holdout; add region diversity; report P/R **at the operating point** per holdout
  with bootstrap CIs (harness exists); stratify FP by Step-0 cause so future deltas are attributable.

### O7 — Phase 3 AIS-mismatch (strategic, the real precision ceiling)
- **dark-vessel = detection − AIS** filters out FPs that sit on a matching AIS track and keeps real dark
  vessels. This *inverts* the metric from recall (GSD-capped) to mismatch precision (improvable). It's the
  payoff that makes a modest detector "good enough" — see `docs/ROADMAP.md` Phase 3. Complementary to
  O1-O4, not a substitute.

**Recommended optical sequence:** Step 0 → **O1 (+O2a NDWI mask)** → **O3** → **O4 ongoing** → O6 → O5, with
O7 as the strategic overlay.

---

## Part 2 — Radar (SAR) model — build `best_sar.pt`

**Scope decision from the review: target LARGER / metallic vessels (the dark-ship / AIS-off mission).**
Do not expect SAR to match optical on small artisanal boats — C-band barely returns them. Budget for
speckle + coastal clutter as the dominant FP sources.

| # | Lever | Attacks | Effort | Notes |
|---|---|---|---|---|
| S1 | Training data: xView3-SAR (+HRSID/SAR-Ship) | build | med | closest analog: S1 dual-pol, dark-vessel labels + length |
| S2 | Preprocessing alignment to the provider chips | build (critical) | med | train == inference representation or it fails |
| S3 | Land/water mask (shared with O1) | precision | low-med | coastal clutter is the top SAR FP |
| S4 | Train separate P2 SAR model | build | med | NOT warm-started from optical `best.pt` |
| S5 | Real S1 holdout via HITL review | trust | med | honest eval on our own chips |
| S6 | Speckle filter (Lee/Refined-Lee) A/B | precision/recall | low | may erase 1-5 px targets — test, don't assume |
| S7 | Cross-modality S1×S2 validation | trust | low | already have Fujairah/Singapore/Cyclades pairs |

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
- **Fix the SAR run-dir misnomer:** SAR fetches currently write under `data/raw/sentinel2/` (the default
  `--output-dir`). Add `data/raw/sentinel1/` so modalities don't mix (cosmetic, do alongside SAR work).
- **The active-learning HITL loop is the engine for both models** — every accuracy lever above flows
  through fetch → detect → review → export → build → train → promote, with a held-out scene each round.
- **Phase 3 AIS fusion (O7) is the precision endgame for both** — it filters each model's false positives
  against AIS, which is what ultimately makes 10 m / C-band detection operationally "good enough."
