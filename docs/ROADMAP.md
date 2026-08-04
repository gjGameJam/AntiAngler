# ROADMAP — improving the detector & building the MPA-violation product

_Snapshot: **2026-08-04** (verified against code and against real eval runs, not memory). The forward
plan. Written to be read **cold**, with no prior conversation context._

> **Last session (2026-08-03/04) closed W2, W3-A, W6, W7 and W8, and removed W9.** The headline is
> **`sar-deep3` promoted** — the first SAR result that is statistically decisive rather than
> noise-band, because the gate finally has two regions. Start at **"THE WORK QUEUE"** below; the
> per-item "outcome" boxes carry what was measured and why.

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
| **Optical detector** | `data/training/weights/best.pt` = **`finetune-adriatic`** (promoted 2026-08-02). YOLOv8n-**P2** on Sentinel-2 10 m RGB. **Deploy: `imgsz 1024`, `conf 0.20`** — baked into `detect_boats.py` defaults (verified in code 2026-08-03). ⚠️ conf moved **DOWN** 0.25→0.20 at this promotion; always re-sweep, never inherit. |
| **SAR detector** | `data/training/weights/best_sar.pt` = **`sar-deep3`** (promoted **2026-08-04**, W2). **Deploy: `imgsz 1024`, `conf 0.10`** — the first swept SAR operating point. Trained on 5 region families (Fujairah/Hormuz/Singapore/Kaohsiung/Suez) with **Gibraltar + Santos held out**. Gibraltar P 0.545 / R 0.522, Santos P 0.316 / R 0.424 — the outgoing `sar-deep` scored **0/99 on Santos**. Backup: `best_sar_deep.pt`. Still noisy on fresh regions (1.86 FP/chip on Santos). |
| **Phase 3 (AIS fusion)** | **All stages built, offline-tested, AND live-validated** (2026-07-26). Definition of done **met**. |
| **Offline test suite** | **263 tests**, pure stdlib, no network / keys / ML deps: `python -m unittest discover -s scripts/tests` |
| **Eval / gating harness** | `eval_holdout.py` + `conf_sweep.py`, **modality-agnostic since W5** (`--train-dir`, default optical). Gates both trees; `PROMOTED_BY_TREE` stops an optical checkpoint leaking into a SAR gate. |
| **Code health** | All `docs/AUDIT.md` P1/P2 findings closed. The P3 items are conscious house-style deferrals, not debt. |

**Four holdout scenes anchor every optical gate.** These are the numbers a candidate model must beat —
the **live `finetune-adriatic` at `imgsz 1024 / conf 0.20`** (its 2026-08-02 promotion eval; Kornati
re-verified 2026-08-03 by the W5 harness):

| Holdout | Type | Boxes | P | R | Note |
|---|---|---|---|---|---|
| **Kornati** (Adriatic) | cross-region small-vessel | 156 | 0.558 | **0.590** | CI [0.50, 0.68]. Weakest link, but **GSD-capped, not data-capped** — W1 refuted the data premise |
| Alonnisos (Aegean) | cross-type small-vessel | 76 | — | 0.855 | relabeled 2026-07-16 → honest baseline |
| Port Said | big-ship dense anchorage | 42 | — | 0.786 | `--imgsz 1280 --augment` profile reaches 0.81 |
| Singapore Jul-5 | cross-date | 26 | — | 0.731 | the original gate; **regressed** 0.769→0.731 at the W1 promotion |

Supporting figures at the same point: open-water FP probe **0.79 dets/chip**, base mAP50 **0.640**,
base recall 0.590. ⚠️ **Gate on all four** — `eval_holdout.py --holdout` takes ONE scene and defaults to
`fallbacktest` (Singapore), so a bare invocation silently reports 1/4 of the evidence (STATUS #9).

**The SAR gate is now TWO regions** (`--train-dir data/training_sar`, 84 chips / **122 boxes**:
`s1deep-gibraltar` 23 + `s1warm-santos` 99). `best_sar.pt` = `sar-deep3` @ imgsz 1024, **conf 0.10**:

| holdout | P | R | F1 | FP/chip | mAP50 | recall 90% CI |
|---|---|---|---|---|---|---|
| Gibraltar | 0.545 | 0.522 | **0.533** | 0.29 | 0.415 | [0.40, 0.65] |
| Santos | 0.316 | 0.424 | 0.362 | 1.86 | 0.261 | [0.33, 0.51] |

Both scenes peak at conf 0.10. Combined test-split mAP50 **0.277**, base recall **0.418**; open-water
Fujairah probe (in-TRAIN, directional) **19.1 dets/chip**. ⚠️ **Loop `--holdout` over BOTH scenes** —
it takes one at a time, and Gibraltar alone hid the incumbent's total blindness on Santos.

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
sandbox, no credentials.

**Status in one line: W1–W9 are all closed or removed. Three things remain: W10, W4, and W3-B.**
All AUDIT P1/P2 findings are closed, Phase 3's definition of done is met, both detectors are promoted
at swept operating points, and the SAR gate now spans two regions.

**Start with W10** — it is the only item that is both cheap (~10 min of GPU) and unambiguously
worthwhile: the W6 despeckle measured a 63% false-alarm cut but could not be *confirmed* on the old
one-region gate, and that gate is now 5× thicker. Then **W4** (optical hard negatives) or **W3-B**
(full fusion), both of which need more of your time than machine time.

**What was learned last session that changes the plan:**
- **The gate, not the model, was the binding constraint on SAR** — and fixing it immediately turned an
  unprovable result into a decisive one. Prefer widening a gate over tuning a model when the CI is
  wide enough to hide the effect you are chasing.
- **Two roadmap items dissolved on contact with measurement** (W6's 1–5 px premise, W7 entirely).
  Check whether the question is answerable with the data on hand *before* building the experiment.
- **Optical is at the 10 m GSD floor** (W1, 2026-08-02) and further optical harvesting is a
  precision/coverage tool, not a recall lever.

| # | Item | Attacks | Effort | Blocker |
|---|---|---|---|---|
| ~~**W1**~~ | ~~Adriatic optical scenes → retrain → re-gate~~ | recall (Kornati gap) | — | ✅ **RUN 2026-08-02 — see "W1 outcome" below. Bought precision, NOT recall.** |
| ~~**W2**~~ | ~~SAR region diversity **or** xView3 transfer~~ | SAR generalization | — | ✅ **DONE 2026-08-04 — `sar-deep3` PROMOTED at conf 0.10. Gibraltar P 0.367→0.545, Santos 0.000→0.424 (CIs disjoint). See "W2 outcome".** |
| **W3-B** | Full fusion vs a fresh overpass (path A ✅ done) | product proof | S–M | 🏠 **Path A RAN LIVE 2026-08-03/04** — 17 reserves geofenced, 1 event, GFW-corroborated (see "W3 outcome"). **Path B needs an acquisition inside the AIS capture window** |
| **W4** | O3 same-region hard negatives | optical precision | M | 🏠 review — **downgraded**: W1 already delivered a ~45% open-water FP cut |
| **W10** ⬅ **start here** | Re-run the W6 despeckle A/B on the 2-region gate | SAR precision | **S** | 🏠 GPU only (~10 min). No review needed — see "W10" below |
| ~~**W5**~~ | ~~`eval_holdout.py --train-dir` (SAR gating harness)~~ | trust / unblocks W2 | — | ✅ **DONE 2026-08-03 — see "W5 outcome" below. W2's blocker is cleared, and the SAR gate turned out weaker than recorded.** |
| ~~**W6**~~ | ~~S6 speckle-filter A/B~~ | SAR precision/recall | — | ✅ **RAN 2026-08-03/04 — see "W6 outcome". Peak-preserving despeckle: same recall, −63% false alarms, recall ceiling 0.70→0.87. Plain Lee is a trap (ceiling 0.70→0.39). Not promoted — fold into W2.** |
| ~~**W7**~~ | ~~S7 cross-modality S1×S2 check~~ | trust | — | ✅ **RAN 2026-08-03 — see "W7 outcome" below. Not answerable as posed: S1/S2 orbits guarantee a ≥4.4 h gap. Georeferencing agreement WAS confirmed.** |
| ~~**W8**~~ | ~~Websocket reconnect for `aisstream_fetch.py`~~ | W3 robustness | — | ✅ **DONE 2026-08-03 — wall-clock duration, gaps recorded, `(mmsi,timestamp)` dedup. Also fixed a truncated capture silently claiming the full window.** |

---

## W10 — Re-run the despeckle A/B on the 2-region gate 🏠 **← start here**

**Why.** W6 measured that a **peak-preserving Lee despeckle cuts open-water false alarms 63% at
matched recall** — but on the *old* one-region gate (23 boxes / 15 distinct vessels, recall CI ±0.2)
that could not be confirmed, and IoU-mAP50 moved the opposite way. The gate is now **122 boxes across
two regions**, which is exactly the thickness that was missing. This is the cheapest open item in the
file: no fetching, no review, ~10 minutes of GPU. It either confirms a real precision win or kills the
idea for good.

**It matters most for the residual problem `sar-deep3` still has:** 1.86 FP/chip on Santos at the
deploy point. Despeckling targets precisely that.

```bash
# 1. Build the despeckled tree (labels are geometric -> copied unchanged)
python scripts/sar_despeckle.py --src data/training_sar --dst data/training_sar_peak   # --mode peak default
# 2. Train with the SAME recipe sar-deep3 used, so the only variable is the filter
python scripts/train.py --weights data/training/yolov8n-p2.yaml --pretrained yolov8n \
    --data data/training_sar_peak/config.yml --imgsz 1024 --batch 4 \
    --mosaic 0.5 --close-mosaic 15 --epochs 30 --name sar-peak
# 3. Sweep BOTH holdouts (--holdout takes ONE scene — loop it, or you gate on Gibraltar alone and
#    repeat the exact mistake that hid sar-deep's total blindness on Santos)
for H in s1deep-gibraltar s1warm-santos; do
  python scripts/conf_sweep.py --train-dir data/training_sar_peak --holdout "$H" \
      --fp-scene s1deep-fujairah --imgsz 1024 --weights data/training/runs/sar-peak/weights/best.pt
done
# 4. Gate at the swept conf, both holdouts. Pass --old/--new EXPLICITLY: a despeckled tree is not in
#    PROMOTED_BY_TREE, and resolve_weights only accepts an explicit path for an unregistered tree.
for H in s1deep-gibraltar s1warm-santos; do
  python scripts/eval_holdout.py --train-dir data/training_sar_peak --holdout "$H" \
      --fp-scenes s1deep-fujairah --imgsz 1024 --conf <swept> \
      --old data/training/weights/best_sar.pt --new data/training/runs/sar-peak/weights/best.pt
done
```

**Acceptance:** FP/chip drops on **both** holdouts at matched-or-better recall, and the Santos recall
CI does not fall below `sar-deep3`'s [0.33, 0.51]. Compare against the incumbent numbers in "Where we
are" above.

**Traps:**
- **The comparison is not apples-to-apples unless each model is gated on its own preprocessing.**
  `sar-deep3` must be measured on `data/training_sar`, the candidate on `data/training_sar_peak`. The
  filter is part of the model.
- **Promoting means a pipeline change, not a weights swap.** Inference must despeckle exactly as
  training did. Nothing enforces this yet — `detect_boats.py` has no despeckle flag, so adopting it
  means either writing despeckled chips in `providers/s1.py` or adding that flag. Decide before
  promoting, not after.
- `--mode lee` is a **measured trap** (recall ceiling 0.696→0.391). Use the default `peak`.
- The despeckled trees are **gitignored** — reproducible from `sar_despeckle.py`, not committed.

---

## W1 — Close the Kornati gap (Adriatic optical data) 🏠 *(done — kept for the record)*

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
# 6. Gate on all four holdouts, then pick the operating point.
#    ⚠️ eval_holdout.py defaults to the 'fallbacktest' holdout ONLY. A bare invocation silently
#    gates on Singapore's 26 boxes and says nothing about the other three. You MUST loop --holdout.
for H in kornati-adriatic alonnisos portsaid fallbacktest; do
  python scripts/eval_holdout.py --old data/training/weights/best.pt --old-imgsz 1024 \
      --new data/training/runs/finetune-adriatic/weights/best.pt --new-imgsz 1024 --holdout "$H"
  python scripts/conf_sweep.py --weights data/training/runs/finetune-adriatic/weights/best.pt \
      --imgsz 1024 --holdout "$H"
done
#    Sweep BOTH models, not just the new one: a retrain can move the operating point in either
#    direction, and comparing a new model at the old model's conf measures the threshold, not the
#    model. Re-run eval_holdout --conf <new deploy conf> to get bootstrap CIs at the point you
#    actually intend to ship.
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
- **Deploy conf is NOT monotonic.** It moved +0.05 at each of two promotions (0.15→0.20→0.25), then
  **back down to 0.20** for `finetune-adriatic`. Let `conf_sweep.py` decide, in both directions.
- Re-apply the scene-aware holdout after `build_dataset.py` if you bypass `--holdout-scene`
  (snippet in `docs/STATUS.md` → Operational recipes).

### W1 outcome (2026-08-02) — ran in full; the premise was half right

**Done:** 3 scenes fetched (`sibenik-primosten` + `hvar-pakleni`, HR; `gocek-fethiye`, TR — all 16
chips, 100% AOI cover / 0% AOI cloud), reviewed to **1093 boat boxes** (692 confirmed + 401 drawn,
only 19 false positives → the live model ran at **P 0.973 / R 0.63** on these scenes). Export tag
`2026-08-02`; TRAIN grew 2512→2725. Three schedules trained.

**Result: the Kornati recall ceiling did NOT move. The gain was precision.**

| Model | Schedule | Best avg F1 (4 holdouts) | Base mAP50 |
|---|---|---|---|
| `finetune-diverse` | — | 0.707 @0.30 | 0.614 |
| **`finetune-adriatic`** ✅ promoted | 30 ep | 0.705 @0.25 | **0.640** |
| `finetune-adriatic-60ep-b` | 60 ep | 0.706 @0.20 | 0.608 |

Aggregate holdout F1 is **tied within noise across all three**. Kornati's recall *ceiling* (conf 0.05)
is 0.904 → 0.891 — unchanged. What did improve is the **precision–recall tradeoff**: at matched
precision ≈0.47 the new models find 0.756–0.788 recall where the live model found 0.673, and
**open-water false alarms fell ~45%** (1.43 → 0.79 dets/chip @0.20) — a direct hit on STATUS open
problem #1. Promoted `finetune-adriatic` at **conf 0.20** (the only candidate with *zero* forgetting;
the 60-epoch run gained Kornati recall but lost base recall 0.585→0.556 and regressed Port Said).

**Acceptance test: NOT met.** Kornati R 0.590 @0.20, CI [0.50, 0.68], **overlaps** the live model's
[0.42, 0.60]; Singapore regressed 0.769→0.731. Promoted anyway as a deliberate call for the FP win.

**What this means for W4 and beyond.** More same-class data is now a **measured** dead end for Kornati
*recall* — three schedules on 1093 fresh in-class boxes moved aggregate F1 by ±0.002. Kornati R ~0.5–0.6
at usable precision looks like the **10 m GSD floor** for that scene's small craft, not a data gap. Do
not spend another review cycle on Adriatic-class optical expecting recall. The remaining levers for
Kornati are structural, not optical: **SAR (W2)** and **AIS fusion (W3)**. Note the precision win means
**W4 (hard negatives) is now lower priority** — this cycle already delivered most of what W4 targeted.

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

**Blocker W5 is CLEARED (2026-08-03).** Gate a candidate with the same harness the optical models get:
```bash
python scripts/eval_holdout.py --train-dir data/training_sar --holdout s1deep-gibraltar \
    --imgsz 1024 --conf <swept> --fp-scenes s1deep-fujairah \
    --old data/training/weights/best_sar.pt --new data/training/runs/sar-deep3/weights/best.pt
python scripts/conf_sweep.py --train-dir data/training_sar --holdout s1deep-gibraltar \
    --fp-scene s1deep-fujairah --imgsz 1024   # sweep BOTH models; SAR has never had a swept deploy conf
```
**But W5 also showed the current gate is weaker than its recorded numbers** (see "W5 outcome"):
`sar-deep`'s P 0.392 / R 0.478 sits at **conf ≈0.10**, where the in-sample Fujairah probe fires
**24.9 dets/chip**, and the recall CI is **[0.27, 0.67]** on 23 boxes. So: quote a swept operating
point for any `sar-deep3`, and treat "did not regress the Gibraltar gate" as necessary but not
sufficient — with a CI that wide, a thin win is indistinguishable from noise. A second reviewed
region in TEST would do more for the gate's power than any tuning.

**Never warm-start SAR from the optical `best.pt`** — wrong texture priors. Use `--pretrained yolov8n`
or a SAR-pretrained backbone.

### W2 outcome ✅ **DONE 2026-08-04 — `sar-deep3` PROMOTED. The first unambiguous SAR win.**

**Ran:** the 3 new regions were reviewed (134 chips, **160 vessel boxes + 92 hard-negative chips**),
exported, and rebuilt with **Santos held out as a SECOND TEST region** alongside Gibraltar — the
change W6/W7 argued for. Gate evidence went **23 boxes → 122** (Gibraltar 23 + Santos 99).

Seed precision varied enormously by region, which is worth knowing before the next review session:
**santos ~80%** (82 correct / 103 seeded), **kaohsiung ~28%**, **suezgulf ~10%**. `sar_seed.py` is
tuned for Fujairah-like deep anchorages; in cluttered coastal water it mostly produces work.

**The incumbent was not weak out-of-region — it was blind.** `best_sar.pt` (sar-deep) scored
**0 of 99 on Santos at EVERY threshold, including conf 0.05**: zero detections, zero false positives.
The per-scene dB stretch in `providers/s1.py` makes each new region a different pixel distribution,
and the thin model simply had nothing to say. This is the same failure as the Singapore probe, now
measured at the floor.

**Gate at the joint F1 peak (conf 0.10 — both scenes peak there):**

| | Gibraltar old → new | Santos old → new |
|---|---|---|
| P | 0.367 → **0.545** | 0.000 → **0.316** |
| R | 0.478 → **0.522** | 0.000 → **0.424** |
| FP/chip | 0.54 → **0.29** | 0.00 → 1.86 |
| mAP50 | 0.335 → **0.415** | 0.044 → **0.261** |
| recall 90% CI | [0.27, 0.67] → [0.40, 0.65] | **[0.00, 0.00] → [0.33, 0.51]** |

Combined test split: **mAP50 0.072 → 0.277**, base recall **0.098 → 0.418**. Open-water probe
**31.60 → 19.12** dets/chip (−40%).

**Acceptance: MET on every clause** (Gibraltar P ≥ 0.39 ✓ 0.545, R ≥ 0.48 ✓ 0.522, mAP50 ≥ 0.335 ✓
0.415; the new region scores non-zero ✓ emphatically). Unlike every previous SAR decision this one is
**statistically decisive** rather than noise-band: the Santos recall CIs are *completely disjoint*.
Gibraltar's still overlap, so the Gibraltar recall gain alone would not have been provable — which is
exactly why the second TEST region mattered.

**Promoted** `data/training/runs/sar-deep3/weights/best.pt` → `best_sar.pt` at **deploy conf 0.10**.
Outgoing model backed up as **`best_sar_deep.pt`**; revert by copying it back.

**What this does and does not fix.** `best_sar.pt` now has a **swept deploy conf and a usable
operating point** for the first time (STATUS #10 was "no usable operating point") — F1 0.533 Gibraltar
/ 0.362 Santos at conf 0.10, with precision up 49% and false alarms nearly halved on Gibraltar. It is
**not** clean: Santos still runs **1.86 FP/chip** at the deploy point, so a fresh region remains noisy
and review is still the quality gate. Next levers, now that the gate can actually resolve them: re-run
the **W6 peak-preserving despeckle** A/B on this thicker gate, and add a third reviewed region.

---

### W2 prep ✅ **NO FETCHING NEEDED — 4 regions were already staged and review-ready (audited 2026-08-04)**

Option A's fetch/seed/companion steps were **already run on 2026-07-17** and never handed off. Six
`s1warm-*` SAR runs sit on disk, each with `water_mask.tif`, a populated `companion/` optical pane
(1:1 with chips), and `sar_seed.py` detections. **None is in `data/training_sar/`.** The only missing
step is human review.

| run | chips | seed dets | chips w/ dets | max/chip | region family | verdict |
|---|---|---|---|---|---|---|
| `s1warm-santos` | 49 | 103 | 23 | 13 | Brazil / S. Atlantic | ⬅ **review first** — most distinct from anything in TRAIN |
| `s1warm-kaohsiung` | 49 | 103 | 11 | 45 | Taiwan | ⬅ **second** — new family |
| `s1warm-suezgulf` | 36 | 87 | 15 | 19 | Red Sea / Suez | ⬅ **third** — new family |
| `s1warm-jebelali` | 56 | 163 | 18 | 50 | Gulf | ⏸ deprioritise — same family as Fujairah/Hormuz |
| `s1warm-rastanura` | 40 | **0** | 0 | — | Gulf | ⛔ seed found nothing |
| `s1warm-durban` | 28 | **1** | 1 | — | S. Africa | ⛔ seed found nothing |

```bash
python scripts/review_server.py --run data/raw/sentinel2/2026-07-17T19-56-10Z_s1warm-santos/
python scripts/export_labels.py --run data/raw/sentinel2/2026-07-17T19-56-10Z_s1warm-santos/ \
    --out data/processed/sar_training_exports --tag santos     # NEVER training_exports/ (optical!)
```
Note `santos` has 6 chips marked reviewed with **zero** box verdicts — a false start; re-review it.

**These were NOT depth-gated** (`--min-depth` unused, no `depth_m` on the detections) — and that is
**correct here**. The deep-water guidance exists to avoid bathymetric clutter in open water, but
Santos/Kaohsiung/Suez are **port approaches**, where the anchored ships we want ARE in shallow water.
A depth gate would drop the targets. Do not "fix" this.

**Strategic change from W6 + W7 — put a reviewed region in TEST, not TRAIN.** Both items concluded
independently that the *gate*, not the model, is now the limiting factor: it is 23 boxes / **15
distinct vessels** with a recall CI of **[0.27, 0.67]**, wide enough that the `sar-deep2` reject was
noise and that W6's 63% false-alarm reduction could not be confirmed. Holding **Santos** out as a
second TEST region alongside Gibraltar roughly doubles the gate's evidence and makes every subsequent
SAR decision cheaper to trust:
```bash
python scripts/build_dataset.py --exports data/processed/sar_training_exports \
    --data data/training_sar --holdout-scene s1deep-gibraltar s1warm-santos
```
Then fold in the **W6 peak-preserving despeckle** on the same pass (`scripts/sar_despeckle.py`), so
the region gain and the filter are measured together on a gate thick enough to resolve them.

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

### W3 outcome — path A ✅ **RAN LIVE 2026-08-03/04. Path B still open.**

**Ran:** a 60-minute live AIS capture (23:13→00:13 UTC) over a Santa Barbara Channel bbox
(`-120.3 33.35 -118.8 34.5`) chosen to enclose **17 IUCN Ia reserves**, all `NO_TAKE: All` — then
`ais_fishing.py` → `geofence_ais.py` (looped over all 17 polygons) → `gfw_vessels.py` →
`alert_violations.py`. **Acceptance MET.**

| | |
|---|---|
| Capture | **199 positions / 22 vessels**, 0 skipped, 0 errors, no disconnects in 60 min |
| Fishing inference | 18 of 22 vessels scored; 4 below the evidence floor → `null`, not `0.0` |
| Geofence | 17 reserves checked → **1 vessel inside** (Scorpion SMR, Santa Cruz Island) |
| Identity | GFW: `ISLAND ADVENTURE`, flag USA, gear **PASSENGER**, `fishing_hours 0`, not IUU-listed |
| Score | **0.10** — the exact transit floor |
| Default digest | **0 alerts** (correctly suppressed); `--min-severity low` shows it |

**The event is independently corroborated three ways**, which is what makes it a real acceptance
test rather than a smoke test: (1) its own kinematics — 22.8 kn decelerating to 15.0 kn on an
easterly track, i.e. a fast transit, not working gear; (2) geography — it is inbound to **Scorpion
Anchorage**, the main visitor landing for Channel Islands National Park; (3) **GFW independently
classifies the gear as PASSENGER with zero fishing hours**, agreeing with the local `ais_fishing.py`
heuristic (p≈0.01) which had no access to that registry. Two independent methods, same verdict.

**The scoring chain behaved exactly as designed.** With only the AIS heuristic the event scored
**0.11**; re-run with the GFW record (`fishing_hours 0`) it fell to **0.10**, the pure transit floor,
and `evidence` became `["AIS", "GFW"]`. A legal transit through a no-take reserve is **recorded but
not paged** — the emit-and-rank house pattern working end to end on live data.

**Operational notes for the next run:**
- **An empty default digest is the correct steady state**, not a failure. `--min-severity medium`
  deliberately sits above the 0.10 transit floor, so a well-behaved MPA produces silence. Use
  `--min-severity low` to see the full record.
- **The AISStream free feed is sparse here** — ~3 vessels/min over a 140×128 km box that includes the
  LA approach lanes. Coverage is genuine (pings spread across the full box width; a container ship in
  the westbound lane), the water is just quiet. Budget an hour or more for a usable sample, and note
  `ais_fishing.py` needs **≥3 pings over ≥10 min** per vessel before it will score one at all.
- **TLS asymmetry:** the AISStream **websocket was unaffected** by the Avast interception, but the
  **GFW HTTPS call failed** with `CERTIFICATE_VERIFY_FAILED` until `REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE`
  were pointed at `scratch_logs/ca_avast_bundle_2026-08.pem`. `gfw_vessels.py` degraded gracefully
  (null identity record, no crash) rather than failing the run — but the identity join is then
  silently empty. **Export the bundle before any GFW stage.**

**Still open — path B (full fusion).** This run had no imagery: fusion needs a fresh overpass whose
`scene.datetime` falls *inside* the AIS capture window, and none was available. W8 (now done) is the
prerequisite that makes a multi-hour rolling buffer survivable, so path B is unblocked in principle;
it just needs a capture timed against a real S2/S1 acquisition.

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

## W5 — Give `eval_holdout.py` a `--train-dir` ✅ **DONE 2026-08-03**

**Why (original).** `eval_holdout.py` hardcoded `TRAIN = REPO/"data"/"training"`, so it could only ever
evaluate the **optical** set. The SAR model lives at `data/training_sar/`, which means **the SAR gate
had no tiny-object harness** — no center-distance P/R, no FP/chip, no bootstrap CI, no size strata.
The `sar-deep2` promote/reject decision was made on thinner evidence than any optical decision has been.

**Shipped.** `--train-dir` (default `data/training`) on **both** `eval_holdout.py` and `conf_sweep.py`,
plus the guards that make a cross-modality gate honest:

- **`PROMOTED_BY_TREE`** maps a dataset tree to the checkpoint that gates it
  (`data/training`→`best.pt`, `data/training_sar`→`best_sar.pt`). Both checkpoints live in the *same*
  `data/training/weights/`, so nothing on disk links a tree to its model — without the table a bare
  SAR gate would have defaulted to the **optical** `best.pt` and printed authoritative-looking garbage.
  An unregistered tree errors with an inventory of available weights; **add a row for a new modality.**
- **A zero-match `--holdout` is now a hard error** listing the scenes that *are* in the split, raised
  before any weight loads. It used to sail through and report `P=R=0.000` over zero chips.
- **A zero-chip FP probe is skipped with a warning** instead of reporting `0` detections — on the SAR
  tree the optical probe defaults (`gbr-cairns-reefs`, `testrun`) match nothing, and `0` reads as
  "no false positives". `conf_sweep`'s open-water column prints `n/a` for the same reason.
- **An explicit-but-missing `--old`/`--new`/`--weights` errors.** It used to be dropped silently, so a
  typo'd candidate path made the run compare the old model against *itself* and look like "no regression".
- **`ultralytics` is now lazy-imported** (`load_yolo()`), so both scripts are importable stdlib-only and
  the gate's logic is covered by **50 new offline tests** (`scripts/tests/test_eval_holdout.py`; suite
  203→**253**) — matching/strata/bootstrap/subset code that decides promotions and had none.

**Acceptance: MET.** `eval_holdout.py --train-dir data/training_sar --holdout s1deep-gibraltar
--imgsz 1024` reproduces the recorded Gibraltar gate **exactly** — P 0.392 / R 0.478 / mAP50 0.335 —
and the built subset's mAP50 equals the tree's own test split, confirming it selects precisely the
Gibraltar chips. Optical back-compat re-verified on real numbers: Kornati **P 0.558 / R 0.590 @0.20**,
CI [0.50, 0.68], base mAP50 0.640, open-water probe 0.79 dets/chip — all matching `docs/STATUS.md`.

### W5 outcome — the harness immediately devalued the SAR gate

Running it revealed that `best_sar.pt` is **materially weaker than "P 0.392 / R 0.478" suggested**,
because that pair is not a deployable operating point. `conf_sweep.py` on the same tree:

| conf | P | R | F1 | FP/chip | Fujairah dets/chip |
|---|---|---|---|---|---|
| 0.05 | 0.286 | 0.696 | 0.405 | 1.14 | 35.4 |
| **0.10** | 0.367 | **0.478** | **0.415** ⬅ peak | 0.54 | 24.9 |
| 0.20 | 0.462 | 0.261 | 0.333 | 0.20 | 16.2 |
| 0.25 | 0.714 | 0.217 | 0.333 | 0.06 | 13.3 |

Three things follow:
1. **The recorded gate sits at ≈conf 0.10** — the sweep recovers R 0.478 (11/23) exactly there. Every
   optical model is quoted at a swept deploy conf (0.20–0.25); the SAR model never was.
2. **There is no usable operating point yet.** At the F1 peak the in-sample Fujairah probe fires
   **24.9 dets/chip**; pushing conf up to quiet it costs almost all recall (0.478→0.217 by conf 0.25).
   *Caveat: Fujairah is a busy anchorage in TRAIN, so that probe is directional and includes real
   vessels — it is NOT comparable to the optical near-empty probes (0.79 dets/chip).*
3. **Recall ceiling ~0.70** (conf 0.05), and the 90% bootstrap CI at conf 0.10 is **[0.27, 0.67]** — so
   wide on 23 boxes that the `sar-deep2` reject was, as suspected, within noise.

**Also learned:** all 23 SAR ground-truth boxes are **>5 px**, so the `1-2px`/`3-5px` strata are empty
and carry no signal for SAR. If the strata should discriminate on this modality they need SAR-specific
bin edges — the tiny-object problem that motivated them is an *optical* 10 m-GSD artifact.

**What this means for W2:** the gate is no longer eyeball-grade, but it is still **thin** (35 chips /
23 boxes, CI ±0.2). Treat clearing it as necessary, not sufficient, and report the swept operating
point — never ultralytics' `val` pair — when proposing `sar-deep3`.

---

## W6 — SAR speckle A/B (S6) ✅ **RAN 2026-08-03/04 — peak-preserving despeckle is a qualified WIN**

**Original intent.** Try Lee / Refined-Lee before tiling and measure recall on 1–5 px targets **both
ways**. Speckle filters can smooth away the very targets we want — test, do not assume.

**Two premises of that framing turned out to be wrong, and one right.**

**1. The stated measurement is impossible on this holdout.** SAR ground-truth box sizes:

| split | boxes | min | median | <11 px |
|---|---|---|---|---|
| train | 303 | 4.0 px | 19.0 px | 47 (16%) |
| val | 147 | 4.0 px | 16.0 px | 36 (24%) |
| **test (Gibraltar)** | **23** | **12.0 px** | **17.9 px** | **0** |

There are **no 1–5 px targets in TEST** — the smallest is 12 px. "Recall on 1–5 px targets" cannot be
measured here at all. (Consistent with W5: the `1-2px`/`3-5px` strata are an *optical* 10 m-GSD
construct.) The answerable question became overall P/R at a swept operating point.

**2. Domain matters, and the obvious filter is the wrong one.** Speckle is *multiplicative* in linear
power, but `providers/s1.py` emits **dB-stretched 8-bit** chips, where it is approximately *additive*.
So the correct choice is Lee's original **local-statistics (additive)** form; the multiplicative
variant would be mis-specified. Noise variance is estimated as the **median of local variances**
(homogeneous water dominates a maritime scene) — self-calibrating, which matters because s1.py
stretches every scene differently. Same robust-statistics posture as `sar_seed.py`'s `median + k·MAD`.

**3. Speckle filters really can erase targets — but which filter decides everything.**

**Ran:** filtered copies of the whole SAR tree (labels are geometric, so they transfer unchanged and
the A/B compares pixels only), then trained each with the *identical* `sar-deep` recipe and gated each
on **its own** preprocessing (train and inference must match).

Pixel-level, over the 23 GT boxes:

| tree | target peak | bg speckle sd | contrast (peak−bg)/sd |
|---|---|---|---|
| original | 253.0 | 29.63 | 6.86 |
| Lee w7 | 241.9 (−4%) | 23.65 (−20%) | **9.71 (+41%)** |
| peak-preserving Lee w7 | 242.6 | 23.90 (−19%) | 9.42 (+37%) |

Detection, swept per model on the Gibraltar holdout:

| model | F1 peak | P / R @peak | recall ceiling (conf 0.05) | FP probe @peak | recall 90% CI | mAP50 |
|---|---|---|---|---|---|---|
| `sar-deep` (baseline) | 0.415 @0.10 | 0.367 / 0.478 | 0.696 (16/23) | 797 | [0.27, 0.67] | 0.335 |
| `sar-lee` (plain Lee) | 0.381 @0.10 | 0.421 / 0.348 | **0.391 (9/23)** | — | [0.21, 0.50] | 0.187 |
| **`sar-leeref`** (peak-preserving) | **0.431 @0.25** | 0.393 / **0.478** | **0.870 (20/23)** | **292** | [0.31, 0.67] | 0.181 |

**The headline: at matched recall (both 11/23 TP, R 0.478) the peak-preserving filter cuts open-water
false alarms 797 → 292 raw detections, a 63% reduction**, with precision slightly up (0.393 vs 0.367)
and the recall *ceiling* far higher (0.870 vs 0.696) — i.e. more headroom to trade precision for
recall, which is exactly what a model with "no usable operating point" (STATUS #10) needs.

**The methodological lesson worth keeping: pixel contrast did NOT predict detector performance — it
inverted it.** Plain Lee had the *best* contrast (+41%) and the *worst* detection (recall ceiling
0.696 → 0.391, losing 7 of 16 findable vessels). Smoothing a compact target's peak destroys it even
while the peak-to-background *ratio* improves. **Do not use SNR/contrast metrics as a proxy for
detection quality on a learned detector — measure detection.**

**Honest caveats — this is not yet promotable:**
- **Nothing here is statistically proven.** The recall CIs overlap heavily ([0.27, 0.67] baseline vs
  [0.31, 0.67] refined) on 23 boxes / **15 distinct vessels** (W7). The FP-probe reduction (797→292)
  is the largest and most consistent effect, but that probe is **in-TRAIN Fujairah** and therefore
  directional — it includes real vessels.
- **IoU-mAP50 got worse** (0.335 → 0.181) even as the instance-level metrics improved. The harness
  itself flags IoU-mAP as unstable at this scale, and the center-distance metrics are the trustworthy
  ones — but the disagreement should be resolved on a thicker gate before promoting.
- **The "refined" filter here is NOT textbook Refined-Lee** (Lee 1981's 8 edge-aligned sub-windows).
  It is a simpler hybrid: Lee output everywhere except where `x > mean + 2σ`, where the original pixel
  is kept — i.e. smooth the background, never touch a compact target's peak. Real Refined-Lee should
  be tried if this direction is pursued.
- **Adopting it is a pipeline change, not a weights swap.** Inference must despeckle exactly as
  training did, so it belongs in `providers/s1.py` (chips written despeckled — review then sees what
  the model sees) or as an explicit `detect_boats.py` preprocessing flag. That decision is open.

**Recommendation:** fold the peak-preserving despeckle into the **W2** attempt rather than promoting
it standalone — W2 adds regions, which is what actually fixes the model *and* thickens the gate
enough to tell whether this 63% FP cut is real.

**The filter is committed as `scripts/sar_despeckle.py`** (verified to reproduce the A/B trees
byte-for-byte, so the script *is* the code behind the numbers above). To redo or extend the A/B:
```bash
python scripts/sar_despeckle.py --src data/training_sar --dst data/training_sar_peak   # --mode peak
python scripts/train.py --weights data/training/yolov8n-p2.yaml --pretrained yolov8n \
    --data data/training_sar_peak/config.yml --imgsz 1024 --batch 4 \
    --mosaic 0.5 --close-mosaic 15 --epochs 30 --name sar-peak
# gate on ITS OWN preprocessing — train and inference must match — and pass --weights explicitly,
# since a despeckled tree is not in PROMOTED_BY_TREE (resolve_weights accepts an explicit path)
python scripts/conf_sweep.py --train-dir data/training_sar_peak --holdout s1deep-gibraltar \
    --fp-scene s1deep-fujairah --imgsz 1024 --weights data/training/runs/sar-peak/weights/best.pt
```
The derived trees are **gitignored** (reproducible; not worth ~334 committed PNGs) while
`data/training_sar/` itself stays tracked as source data.

---

## W7 — cross-modality S1×S2 ✅ **RAN 2026-08-03 — the question is not answerable as posed**

**Original intent.** Run the SAR detector on S1 chips, compare against optical detections over the
same AOI, and confirm SAR flags large/dark vessels optical missed — "the entire justification for the
modality".

**What was run.** The confound of `best_sar.pt`'s bad operating point was sidestepped by using the
**human-reviewed** SAR boxes as truth instead of the model. Gibraltar was the strongest available pair
(reviewed SAR holdout + a paired S2 scene the *same day*); the optical detector was run on
`s2deep-gibraltar` (61 detections @ conf 0.20) and Suez Gulf was run in the reverse direction
(optical-reviewed truth vs SAR candidates).

### W7 outcome — S1 and S2 can never image the same water within ~4.4 h

The 10 paired runs on disk have S1↔S2 acquisition gaps of **7.4 h to 859 h**; only two pairs are even
same-day, and *both* are 7.4 h. That is not luck — it is orbital. Local **solar** overpass time across
all 20 scenes:

| Provider | Local solar overpass (n=10 each) |
|---|---|
| Sentinel-2 | **09:40 – 10:38** (tight, ~10:30 descending node) |
| Sentinel-1 | **05:15 – 05:51** (descending) **or** **17:54 – 18:04** (ascending) |

Both are sun-synchronous with different local times of ascending node. Measuring the separation
per-AOI gives two clean clusters and no values in between: **4.42 / 4.47 / 4.70 / 4.70 h** where S1
used its *descending* pass, and **7.35 / 7.35 / 7.38 / 7.50 / 7.58 / 7.63 h** where it used the
*ascending* pass. So the **best achievable S1↔S2 separation over one point is ~4.4 h** and the common
case is ~7.4 h. **No choice of date or AOI improves this.** A 12 kn vessel covers ~98 km in 4.4 h, so
a per-vessel coincidence test is only
defensible for *stationary* vessels — and "SAR saw it, optical did not" cannot be distinguished from
"the vessel was not there yet".

**The measured numbers, for the record** (Gibraltar, 7.4 h gap, after deduplicating **both** modalities
at 30 m — the same rule `detect_boats.py` applies cross-chip):

| | count |
|---|---|
| SAR reviewed boxes | 23 → **15 distinct vessels** (8 were chip-overlap duplicates) |
| Optical detections @ conf 0.20 | 61 |
| Coincident within 200 m | **4 / 15 (27%)** |
| SAR-only | 11 |
| Optical-only | 57 |

Suez Gulf (the reverse direction, 7.4 h) is the control that proves the confound: **0 of 7**
optically-confirmed vessels had a SAR candidate within 200 m, median nearest neighbour **2,813 m** —
it is a transit corridor, so essentially everything moved between overpasses.

**What is genuinely established:** the 4 coincidences agree to **23–175 m across two sensors, two
CRSs and two providers**, which is the first quantitative confirmation that the S1 and S2 chip
georeferencing paths are mutually consistent. That was never measured before and it validates the
`Provider` seam's geolocation.

**What is NOT established, and cannot be with free Sentinel data:** that SAR finds vessels optical
missed. Re-scoping options, in order of cost:
1. **Restrict to demonstrably stationary vessels** — an anchorage AOI (Fujairah, Jebel Ali) rather
   than a strait. Gibraltar/Suez were both transit corridors, which is the worst possible choice.
2. **Use AIS as the arbiter instead of the other sensor** — for a vessel with AIS at both overpass
   times, the position at each is known, so "did each modality see it" becomes answerable. This
   folds W7 into the Phase-3 machinery rather than treating it as an imagery problem.
3. Accept an **aggregate/statistical** comparison (counts and size distributions over the same water)
   and drop per-vessel matching entirely.

**Side finding relevant to W2/W5:** the SAR TEST holdout's "23 boxes" are **15 distinct vessels**.
Per-chip labels are correct (a vessel in two overlapping chips genuinely appears in both), but the
*effective independent sample* behind the Gibraltar gate is 15, not 23 — so the recall CI **[0.27,
0.67]** is, if anything, optimistic about its own width. A second reviewed region matters even more
than "Prioritise a second reviewed region in TEST" already implied.

---

## W8 — Websocket reconnect for `aisstream_fetch.py` ✅ **DONE 2026-08-03**

**Why it was left open.** Every other network call site got the shared `scripts/_http.py` policy in
the AUDIT #4 round, but a websocket "retry" means **reconnect-and-resume mid-capture**, which changes
*capture semantics* rather than transport settings. It was left alone rather than half-done.

**The three semantics decisions (settled 2026-08-03):**

| Question | Decision | Why |
|---|---|---|
| `--duration` = wall-clock or connected-time? | **Wall-clock** — deadline fixed at entry; an outage eats into the window | Fusion is only valid when `scene.datetime` falls *inside* the capture window, so a capture must cover a known stretch of time, not accumulate a quota. Also keeps the end time predictable for cron. |
| Dedup on reconnect? | **Yes, on `(mmsi, timestamp)`** | A reconnect can in principle re-deliver a ping. Distinct from `--dedup latest` (one ping per vessel), which is untouched and still applied afterwards. |
| Record connection gaps? | **Yes — full connection log** | Same honesty property `ais_fishing.py` already applies to AIS gaps: unobserved time is recorded, never assumed. |

**A pre-existing bug was found and fixed in passing.** `main()` wrote `"duration_s": args.duration`
— the *requested* window — unconditionally, while `collect()` would `break` out of its loop when the
server closed the socket. So a 60-minute capture truncated at minute 3 wrote a file claiming
`duration_s: 3600`, and **nothing downstream could tell it had been truncated**. Ctrl-C had the same
problem. The output now carries what actually happened:

```json
"duration_s": 3600,        // requested (unchanged, back-compatible)
"elapsed_s": 3600.0,       // wall-clock actually spent
"connected_s": 2870.4,     // of which genuinely connected
"reconnects": 2,
"gaps": [{"from": "...T23:31:04Z", "to": "...T23:36:12Z", "seconds": 308.0}],
"truncated": false,
"duplicates_dropped": 0
```

**Shipped:** auto-reconnect until the wall-clock deadline with capped exponential backoff
(`reconnect_delay()` — 0s/2s/4s/8s… capped at 30s, mirroring `_http.py`'s shape and pure/testable),
`(mmsi, timestamp)` duplicate suppression, per-session connected-time accounting, and gap intervals
stamped in UTC. An AISStream **`error` frame still fails fast and is never retried** — a bad key or
bbox would otherwise just hammer the endpoint. New flags: `--no-reconnect` (old behaviour),
`--reconnect-backoff`, `--max-backoff`, each with an `AIS_*` env fallback in house style.

**Acceptance: MET.** 10 new offline tests (suite **253 → 263**), all using an injected `sleep` so
they never actually wait: reconnect-after-drop records a gap, a replayed ping is dropped as a
duplicate, a failed handshake is retried, an `error` frame is *not* retried, `--no-reconnect` stops
at the first drop, and the backoff schedule is pinned. Run time 0.016 s.

**A flapping-endpoint bug was caught in self-review before this ever ran live.** The first cut reset
the backoff counter on *every received frame* ("a frame proves the connection is usable"). That is
wrong for the exact failure it needs to handle: an endpoint that serves one frame and drops would
reset the counter every cycle and be reconnected at **0 s delay forever** — a hammering loop. The
reset is now gated on a session staying up for `stability_s` (default 30 s), so a flapping endpoint
escalates 0s→2s→4s→… while a genuinely healthy session still resets to 0. Both directions are pinned
by tests (`test_flapping_endpoint_escalates_backoff`, `test_stable_session_resets_backoff`).

---

# DO NOT RE-ATTEMPT — measured dead ends

_Each of these was built or measured, cost real time, and did **not** work. They are recorded so a
future session does not rediscover them. Do not undo these conclusions without new evidence._

| Thing | Verdict | Why |
|---|---|---|
| **imgsz 1280 (plain, no TTA)** | ❌ non-win | Port Said identical; Alonnisos recall **0.70→0.49**. Upscaling past the 1024 *train* size loses the 1–5 px vessels it was meant to enlarge. 1536 not worth testing. The big-ship lift comes from **`--augment` TTA**, not raw imgsz. |
| **NDWI / NIR open-water filter for precision** | ❌ calibrated non-win | `ndwi_mask.py` + `--open-water` are built, correct, and **off by default**. The premise ("FPs sit on blank water") is FALSE: glint/foam/swell reflect NIR *more* than 1–5 px boats (FP min-NDWI median −0.122 vs TP −0.027). At −0.10 it drops 7/20 FP but **60/63 TP**. A water index cannot reject real spectral water *features*. |
| **Piling on hard negatives** (`finetune-reef`) | ⚠️ counter-lesson | Recall *and* base mAP both dropped — the detector went over-conservative. Hence `--max-neg-ratio`. Add **positives** to fix recall; use negatives surgically. |
| **Plain Lee despeckling for SAR** | ❌ measured non-win (2026-08-04) | Best *pixel contrast* of anything tried (+41%) and the **worst detection**: recall ceiling 0.696→**0.391**, losing 7 of 16 findable vessels. Smoothing a compact target's peak destroys it even as peak-to-background ratio improves. The **peak-preserving** variant (Lee everywhere except `x > mean+2σ`) is the one that works — see "W6 outcome". |
| **Pixel SNR / contrast as a proxy for detector quality** | ❌ actively misleading (2026-08-04) | W6 measured +41% contrast for the filter that *hurt* detection most. Contrast and detection ranked in **opposite order** across three variants. Measure detection. |
| **Textbook CFAR / top-hat for SAR seeding** | ❌ fails by construction | `providers/s1.py`'s dB-percentile stretch is tuned for *display*: water sits mid-bright (median ~136/255) and vessels saturate at 255. CFAR assumes dark water. `sar_seed.py` instead thresholds the dual-pol geometric mean `sqrt(VV·VH)` at an adaptive `median + k·MAD` — robust and stretch-invariant. |
| **Warm-starting SAR from the optical `best.pt`** | ❌ wrong priors | Different sensor, different texture statistics. Use `--pretrained yolov8n` or a SAR-pretrained backbone. |
| **One more SAR scene to fix generalization** (`sar-deep2`) | ❌ regressed | +Singapore alone dropped the Gibraltar gate P 0.392→0.285. SAR needs *regions* (W2-A) or *transfer* (W2-B). |
| **`sentinel-1-grd` as the SAR collection** | ❌ incompatible | PC's GRD assets are in radar geometry: `crs=None`, identity transform, geolocation only via 189 GCPs → `transform_bounds` crashes. Use **`sentinel-1-rtc`** (real UTM grid, clean 10 m affine). GRD needs a GCP/warp reader first. |
| **Raising `--conf` to fix precision** | ❌ wrong tool | Trades away the recall the 10 m GSD already caps. The architecture is deliberately *recall-favouring candidate generation, then filter downstream* (water mask → AIS fusion). |
| **Paid/commercial imagery (Planet, VHR)** | ⛔ out of scope | Hard $0-budget constraint. Refs were removed deliberately; do not reintroduce. |
| **batch 16 @ imgsz 1024 with the P2 head** | 💥 crashes | Silent torch C++ abort mid-epoch. Use **`--batch 4`** (plateaus ~7 GB). |
| **Per-vessel S1×S2 coincidence matching for MOVING vessels** | ❌ structurally impossible (2026-08-03) | Sentinel-2 crosses at ~10:30 local solar; Sentinel-1 at ~05:45 or ~18:00. The **minimum** S1↔S2 gap over one point is ~4.4 h (measured across 10 paired AOIs: 4.42–4.70 h when S1 is descending, 7.35–7.63 h ascending, nothing between), during which a 12 kn vessel moves ~98 km. Gibraltar 4/15 coincident, Suez Gulf 0/7. No date or AOI choice fixes it — re-scope to stationary vessels, or arbitrate with AIS. See "W7 outcome". |
| **More Adriatic-class optical data to lift Kornati *recall*** | ❌ measured dead end (2026-08-02) | W1 ran in full: 3 in-class scenes, **1093 fresh boxes**, 3 schedules (30/60/60-corrected). Aggregate 4-holdout F1 moved **±0.002**; Kornati's recall ceiling 0.904→0.891 (unchanged). It *did* buy precision (~45% fewer open-water FPs). Kornati R ~0.5–0.6 at usable precision is the **10 m GSD floor**, not a data gap → go structural (SAR/AIS), not more optical. |
| **Longer training (60 ep) to squeeze more from a warm start** | ❌ non-win + forgetting | Satellite val mAP50 rose 0.785→0.798 but holdout F1 did **not** follow (0.705→0.706), and base-photo recall *fell* 0.585→0.556. Val mAP50 is a poor proxy for the holdout gates. Also: `close_mosaic` counts back from the END of training, so holding it at 15 while doubling epochs moved the mosaic-free phase to epoch 46 and patience-20 early-stopped at 21 with "best = epoch 1". **Scale `--close-mosaic` proportionally with `--epochs`.** |

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
2026-07-29 with three more offline stages. `docs/PHASE3_PLAN.md` holds the checklist — **every box in
the definition of done is ticked**. Per-script contracts + first-run setup are in `CLAUDE.md`. The only
remaining Phase-3 work is **W3** (the near-real-time operational mode) and **W8** (websocket reconnect,
which W3 depends on).

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
- ✅ **DONE (2026-08-03)** — **the SAR gating harness (W5)**: `eval_holdout.py` + `conf_sweep.py` take
  `--train-dir`, so the same tiny-object gate (center-distance P/R, FP/chip, size strata, bootstrap CI)
  now covers `data/training_sar`. `PROMOTED_BY_TREE` binds each tree to its checkpoint so an optical
  model cannot silently gate SAR; zero-match holdouts and zero-chip FP probes became hard errors /
  warnings instead of silent zeros; a missing explicit `--old`/`--new` no longer degrades to comparing
  a model against itself. `ultralytics` moved into `load_yolo()` so both are importable stdlib-only →
  **50 offline tests** (suite 203→253) over the logic that decides promotions. Optical numbers
  re-verified unchanged; the SAR findings are in "W5 outcome" above.
- **Still open, now tracked in THE WORK QUEUE above:** cross-modality S1×S2 (**W7** / `IMPROVEMENT_PLAN.md`
  S7), the detector-accuracy levers **O3** → **W4** and **O4** → **W1**, and websocket reconnect (**W8**).
  (The Skylight dark-call cross-check — `PHASE3_PLAN.md` 3B, formerly W9 — was **dropped from the queue
  2026-08-03**: it was gated on external API registration, is not a pipeline dependency, and the idea
  stays recorded in `PHASE3_PLAN.md` if it is ever wanted back.)
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
| **Gate a SAR candidate** | same two, plus `--train-dir data/training_sar --holdout s1deep-gibraltar --fp-scenes s1deep-fujairah` (`--fp-scene`, singular, for `conf_sweep`). Weights default to `best_sar.pt` via `PROMOTED_BY_TREE` — the optical default can no longer leak into a SAR gate |
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
| Test the P3 stages offline | `python -m unittest discover -s scripts/tests -p 'test_*.py'` (263 tests; synthetic, no network / key / ML deps — incl. 50 for the eval/gate harness and 10 for websocket reconnect) |
| Violation schema | `docs/PIPELINE.md` → `violation_events.json` (pinned) |

## Key sources
Small-object: SAHI 2202.06934, copy-paste 1902.07296, NWD 2110.13389, P2/SOD-YOLOv8 2408.04786.
Imagery: Kanjir survey (PMC5877374), S2-YOLO RSE 2025. Dark vessels: ESA/GFW (75% untracked),
Skylight (skylight.global/platform), xView3-SAR 2206.00897.
