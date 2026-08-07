# ROADMAP — improving the detector & building the MPA-violation product

_Snapshot: **2026-08-06** (verified against code and against real eval runs, not memory). Written to be
read **cold**, with no prior conversation context._

> **W1–W12 are all closed.** Nothing is queued. Both detectors are promoted at swept operating points,
> Phase 3 has run end-to-end on real data, and all AUDIT P1/P2 findings are closed. What remains in
> this file is (a) the current state, (b) a one-line result per closed item, (c) the long-form outcomes
> for the few whose *lesson* is non-obvious, and (d) the **DO NOT RE-ATTEMPT** table — the most
> valuable section here, because every row cost real time to learn.

**Read in this order:**
1. **`docs/STATUS.md`** — live model, per-run eval numbers, model progression, open problems.
2. **this file** — what was tried, what it measured, and what not to retry.
3. **`docs/PIPELINE.md`** — the build spec and the inter-stage data contracts.
4. **`docs/AUDIT.md`** — code-health findings (all P1/P2 closed).
5. **`CLAUDE.md`** — per-script contracts, data layout, house rules, first-run setup.

---

## Where we are

| | Current state |
|---|---|
| **Optical detector** | `data/training/weights/best.pt` = **`finetune-w4ionian`** (promoted 2026-08-06, W4). YOLOv8n-**P2** on Sentinel-2 10 m RGB. **Deploy: `imgsz 1024`, `conf 0.20`** — baked into `detect_boats.py` defaults. The conf did **not** move at this promotion, but it was still re-swept on all four holdouts for both models; "unchanged" is a measurement, not an inheritance. Backup: `best_finetune-adriatic.pt`. |
| **SAR detector** | `data/training/weights/best_sar.pt` = **`sar-peak2`** (promoted 2026-08-06, W12). **Deploy: `imgsz 1024`, `conf 0.20`.** 🚨 **The SAR pipeline is DESPECKLED end-to-end** — every S1 fetch must pass `sat_fetch.py --provider s1 --despeckle peak`. Backup: `best_sar_deep4.pt`. |
| **Phase 3 (AIS fusion)** | All stages built, offline-tested, live-validated (2026-07-26), and **run end-to-end on real data** (2026-08-06, W3-B) producing all three statuses in one scene. |
| **Offline test suite** | **280 tests**, pure stdlib, no network / keys / ML deps: `python -m unittest discover -s scripts/tests` |
| **Eval / gating harness** | `eval_holdout.py` + `conf_sweep.py`, modality-agnostic via `--train-dir`. `PROMOTED_BY_TREE` stops an optical checkpoint leaking into a SAR gate. |
| **Code health** | All `docs/AUDIT.md` P1/P2 findings closed. The P3 items are conscious house-style deferrals, not debt. |

⚠️ **Two standing facts that will bite a future session** (not queued work — just true):
- **Every SAR run on disk predates W12 and holds RAW chips**, so it is out of domain for the promoted
  `sar-peak2`. Re-fetch with `--despeckle peak`, or re-filter with `sar_despeckle.py --run <dir>`.
  `manifest["chip_postprocess"]` says which a run is; a missing key means raw.
- **AISStream delivers zero frames** on a key that authenticates (reproduced 2026-08-06: bad key
  dropped in 1.1 s, real key held 46.7 s with 0 frames, while a third-party push socket delivered
  fine). Account/key side, not the pipeline. Historical AIS via `marinecadastre_fetch.py` is
  unaffected and is what W3-B actually used.

### The optical gate — four holdout scenes, 300 boxes

These are the numbers a candidate must beat: the live **`finetune-w4ionian` at `imgsz 1024 / conf 0.20`**.

| Holdout | Type | Boxes | P | R | recall 90% CI |
|---|---|---|---|---|---|
| **Kornati** (Adriatic) | cross-region small-vessel | 156 | 0.550 | **0.635** | [0.56, 0.71] |
| Alonnisos (Aegean) | cross-type small-vessel | 76 | 0.500 | 0.829 | [0.75, 0.94] |
| Port Said | big-ship dense anchorage | 42 | 0.943 | 0.786 | [0.67, 0.91] |
| Singapore Jul-5 (`fallbacktest`) | cross-date | 26 | 0.875 | 0.808 | [0.71, 0.89] |

Pooled **216 TP / 149 FP**. Supporting figures at the same point: base mAP50 **0.603**, base recall
0.594, open-water `testrun` probe 29 raw dets. ⚠️ **Gate on all four** — `eval_holdout.py --holdout`
takes ONE scene and defaults to `fallbacktest`, so a bare invocation silently reports 1/4 of the
evidence. Loop it.

### The SAR gate — two regions, 122 boxes, on the DESPECKLED tree

`--train-dir data/training_sar_peak` (84 chips: `s1deep-gibraltar` 23 boxes + `s1warm-santos` 99).
That tree is **not** in `PROMOTED_BY_TREE`, so pass `--old/--new` explicitly.

| holdout | P | R | F1 | FP/chip | recall 90% CI |
|---|---|---|---|---|---|
| s1deep-gibraltar | 0.576 | **0.826** | 0.679 | 0.40 | **[0.72, 0.94]** |
| s1warm-santos | 0.618 | 0.475 | 0.537 | 0.59 | [0.34, 0.58] |

Pooled **66 TP / 43 FP** (vs `sar-deep4`'s 54 / 101). Santos recall is the weak number.
⚠️ **Do not judge SAR precision on the in-TRAIN Fujairah probe** — TRAIN now holds three Gulf families,
so a better-fitted model fires *more* there. Judge on holdout FP/chip.

---

# THE WORK QUEUE

**Empty.** W1–W12 are all closed — see the results table below. No work is currently queued.

---

# RESULTS — every work item, one line each

| # | Item | Date | Verdict | Headline |
|---|---|---|---|---|
| **W1** | Adriatic optical scenes → retrain | 08-02 | ✅ promoted `finetune-adriatic` | Bought **precision, not recall**: open-water FPs −45%. Recall ceiling unmoved (0.904→0.891). |
| **W2** | SAR region diversity | 08-04 | ✅ promoted `sar-deep3` | First *statistically decisive* SAR result: Gibraltar P 0.367→0.545, Santos 0.000→**0.424**, CIs disjoint. The gate, not the model, had been the constraint. |
| **W3-A** | AIS-only live loop | 08-03/04 | ✅ ran live | 17 reserves geofenced, 1 event, GFW-corroborated. |
| **W3-B** | Full fusion vs a real overpass | 08-06 | ✅ ran (historical) | All three statuses in one scene — see long-form below. |
| **W4** | Same-region hard negatives | 08-06 | ✅ promoted `finetune-w4ionian` | Premise **inverted** — see long-form below. |
| **W5** | `eval_holdout.py --train-dir` | 08-03 | ✅ built | Gave SAR the same tiny-object harness as optical, and immediately showed the SAR gate was weaker than recorded (no swept conf, recall CI ±0.2). |
| **W6** | SAR speckle A/B | 08-03/04 | ⚠️ qualified win | Peak-preserving Lee wins; **plain Lee is a trap**. Verdict on *what* it wins was wrong — see long-form. |
| **W7** | Cross-modality S1×S2 | 08-03 | ❌ not answerable | Structurally impossible for moving vessels — see long-form. |
| **W8** | Websocket reconnect (`aisstream_fetch`) | 08-03 | ✅ built | Capped backoff to a **wall-clock** deadline, plus honest capture accounting (`connected_s`, `gaps`, `reconnects`) so a holey capture is legible instead of silently claiming the full window. |
| ~~W9~~ | — | 08-03 | removed | Dropped from the queue as not a pipeline dependency. |
| **W10** | Re-run the despeckle A/B on the 2-region gate | 08-04 | ❌ refuted W6 | Not promoted. The precision claim did not replicate — see long-form. |
| **W11** | SAR retrain on 3 new regions (2 arms) | 08-05 | ✅ promoted `sar-deep4` | Arm B (`sar-peak2`) won decisively but needed an inference change → became W12. Also caught an **ultralytics version confound** in W10's arm. |
| **W12** | Despeckle at inference → promote `sar-peak2` | 08-06 | ✅ promoted `sar-peak2` | Mismatched preprocessing costs **two-thirds of recall, silently** — see long-form. |

---

# THE FIVE THAT TAUGHT SOMETHING NON-OBVIOUS

Everything else is adequately summarised by the table above. These five are kept in full because the
*lesson* is not recoverable from the headline.

## W1 + W4 — optical is at the 10 m GSD floor (two independent confirmations)

**W1 (2026-08-02):** 3 Adriatic-class scenes, **1093 fresh boxes**, three training schedules
(30 ep / 60 ep / 60 ep with proportional `close-mosaic`). Aggregate 4-holdout F1 moved **±0.002**.
Kornati's recall *ceiling* (conf 0.05) went 0.904→0.891 — unchanged. It **did** buy precision:
open-water false alarms −45%, base mAP50 0.614→0.640. Promoted for that.

**W4 (2026-08-06):** 4 more in-class scenes, **490 boxes**. Ceiling 0.891→**0.897** — flat again.
Promoted for a real but different gain: pooled **216 TP / 149 FP vs 209 / 149** at conf 0.20, i.e.
seven more vessels at an *identical* false-alarm count.

🔑 **Two-for-two on ~1580 boxes: same-class optical positives buy CALIBRATION, never PERCEPTION.**
The model scores faintly-seen vessels higher (so deploy-conf recall rises) but does not learn to see
vessels it was blind to. Kornati R ~0.5–0.6 at usable precision is the **10 m GSD floor, not a data
gap**. Do not plan a third optical harvest expecting a ceiling lift — go structural (SAR / AIS).

**W4's second lesson (the screening heuristic was wrong).** W4 was specified as a *hard-negative*
mine. Five AOIs were ranked by **detections per chip** and the four densest reviewed. The harvest came
back at detector precision **0.962** — 11 false positives in 293 boxes — with the human drawing **208
missed boats**. It yielded 490 positives and 33 negatives: a *recall* harvest. Detection density
selects for **boat-rich** water, not **error-rich** water. Deep open sea scored **0 detections over 25
chips** and was dropped as useless (and as a `finetune-reef` over-conservatism risk).

**W4's third finding — a train/val leak, live since 2026-07-26.** `build_dataset.py` purged
previously-layered files by the `__` marker in `<run>__<chip>`, but `ingest_s2ships_finland.py` names
its chips `finlandS2_…` with no `__`. They survived every purge while the seeded split kept
reassigning them: **510 of 1034 Finland chips were in BOTH train and val.** The holdout TEST split was
never affected (it is scene-filtered `__`-named chips), so **every gate number and promotion decision
stands**; what was contaminated is the val set YOLO uses for best-epoch selection. Fixed by purging on
**provenance** (stem present in an export tag) rather than filename pattern; regression-tested in
`scripts/tests/test_build_dataset.py`, verified to fail against the pre-fix code.
⚠️ **Val / `results.csv` numbers recorded before 2026-08-06 are not comparable to later ones.**

## W6 → W10 → W11 → W12 — the despeckle saga, and why it flip-flopped

The verdict on SAR speckle filtering reversed **three times**. That pattern is the lesson.

| | gate | training chips | verdict |
|---|---|---|---|
| **W6** (08-03/04) | 1 region, 23 boxes | 48 | "−63% false alarms" — a *precision* lever |
| **W10** (08-04) | 2 regions, 122 boxes | 90 | **refuted**: FP probe +6%; a *recall* lever, not precision. Not promoted. |
| **W11** (08-05) | 2 regions, 122 boxes | 122 | **wins on BOTH axes**: pooled 66 TP / 43 FP vs 54 / 101, Gibraltar recall CI disjoint |
| **W12** (08-06) | same | same | shipped — inference-side filter added, `sar-peak2` promoted |

🔑 **The filter's effect is not a fixed property — it scales with training-set size.** That is why
one-shot A/Bs kept mis-reading it. Two further traps came out of the sequence:
- **W10's arm was confounded by an ultralytics version difference** (8.3.221 vs 8.4.90) because it was
  trained under the system Python while its baseline used the venv. **Always train *and* gate with
  `venv/Scripts/python.exe`.**
- **`sar_despeckle.py` does not overwrite an existing `--dst`** (only `mkdir(exist_ok=True)`), and a
  derived tree goes stale the moment `build_dataset.py` runs. `rm -rf` before rebuilding — reusing the
  stale tree in W11 would have left 84 orphan chips, 42 of which had changed split.

**W12 — what shipping it actually required, and the number that justifies the care.** A new optional
`Provider.postprocess_chip(chip, opts)` hook (no-op in the base class) called from `sat_fetch`'s tiling
loop **after tiling**, implemented in `providers/s1.py` by delegating to
`sar_despeckle.despeckle_chip_chw` — so training and inference execute *the same function* rather than
two implementations that must be kept in agreement. Driven by `--despeckle {off,peak,lee}`, default
**off**, and recorded in `manifest["chip_postprocess"]` so a run is self-describing.

Placement is load-bearing: the training chips were filtered **per chip, after tiling, with nodata
already zeroed**, so a mosaic-level filter would disagree within ~window/2 px of every tile boundary.

🔑 **Feeding a raw-trained model despeckled chips (or vice versa) costs about two-thirds of recall and
raises no error.** Measured: `sar-deep4` on despeckled chips scores Gibraltar R **0.217** instead of
0.609, Santos 0.283 instead of 0.505. That is why the flag defaults to off, why the manifest records
it, and why the live path was verified by fetching the same scene twice — 20/20 chips came back
byte-identical to `despeckle(raw)`.

⚠️ `sar_seed.py`'s `median + k·MAD` threshold is still calibrated on RAW chips; re-tune `k` before
pointing it at filtered runs.

## W7 — cross-modality S1×S2 is structurally impossible for moving vessels

Sentinel-2 crosses at ~10:30 local solar time; Sentinel-1 at ~05:45 or ~18:00. Measured across 10
paired AOIs, the **minimum** S1↔S2 gap over one point is **~4.4 h** (4.42–4.70 h when S1 is
descending, 7.35–7.63 h ascending — nothing in between). A 12 kn vessel moves ~98 km in that time.
Gibraltar: 4 coincident of 15; Suez Gulf: 0 of 7.

🔑 **No choice of date or AOI fixes this** — it is orbital geometry, not sampling. Re-scope to
stationary vessels, or arbitrate with AIS (which is what the product does anyway). The transferable
habit: check whether the question is answerable with the data that *can* exist before building the
experiment.

## W3-B — the fusion product, proven end-to-end on real data

Ran on **2024-08-23, Santa Barbara Channel / Anacapa** (S2 at 18:39:19Z, 0% AOI cloud, 30 chips) with
MarineCadastre historical AIS filtered to ±1 h (2763 positions / 74 vessels). All three statuses
appeared in one scene:

- **matched (1)** — **ORION**, MMSI 368109370. AIS ping 18:38:52Z vs acquisition 18:39:19Z: a
  **27-second** offset. At 5.4 kn her feasible-movement envelope was 575 m; the detection sat 450 m
  away, so the gate matched her. The envelope design validating itself on real data.
- **dark (7)** — one inside Scorpion (Santa Cruz Island) SMR.
- **reported (2)** — ISLAND EXPLORER (1.21 h inside Scorpion) and WHALE RIDER, by AIS geofencing alone.

⚠️ **Two honest limits on reading that result.** The 7 dark detections are **unreviewed** at conf 0.20
over rocky, kelpy water — some are certainly false positives. More importantly, **"dark" does not mean
"violation" here**: US AIS carriage is mandatory only for certain vessel classes, and the Channel
Islands are full of small dive and recreational craft that legally carry none. The dark signal is only
strong where any vessel is anomalous. Relatedly, `ais_fishing.py` scored ISLAND EXPLORER 0.36 because
slow meandering reads as fishing-like — which is also exactly what a dive or whale-watching boat does.

**Why historical rather than live:** AISStream authenticates but delivers zero frames (see "Where we
are"). The historical path is also *better* for this proof — you pick the date after the fact, so the
scene and the AIS are guaranteed to be in the same window with no overpass timing to hit.

---

# DO NOT RE-ATTEMPT — measured dead ends

_Each of these was built or measured, cost real time, and did **not** work. They are recorded so a
future session does not rediscover them. Do not undo these conclusions without new evidence._

| Thing | Verdict | Why |
|---|---|---|
| **A third same-class optical harvest to lift the recall *ceiling*** | ❌ two-for-two dead end (2026-08-06) | W1 (1093 boxes) and W4 (490 boxes) both left Kornati's conf-0.05 ceiling flat (0.904→0.891→0.897). Same-class positives buy **calibration**, never **perception**. Kornati R ~0.5–0.6 at usable precision is the **10 m GSD floor**, not a data gap → go structural (SAR/AIS). |
| **Ranking candidate AOIs by *detections per chip* to mine hard NEGATIVES** | ❌ measures the wrong thing (2026-08-06) | W4: the harvest came back at precision **0.962** with **208 missed boats** — 490 positives, 33 negatives. Detection density selects for **boat-rich**, not **error-rich**, water; deep open sea gave 0 dets / 25 chips. To mine negatives, rank by **reviewed FP density** from a prior round, or target known confuser water (glint / swell / whitecap / reef edge). |
| **Masking land to fix optical precision** | ❌ not the lever (2026-07-16, Step 0) | Classified all 110 FPs across three holdouts: **only 16% are land**; **84% are ON-WATER**, overwhelmingly on the small-vessel scene (Alonnisos P 0.25 vs Port Said P 0.97). `water_mask.py` shipped anyway — it is cheap and kills the confident rock/islet hits — but big-vessel precision was already solved and the real problem is on-water alarms. |
| **NDWI / NIR open-water filter for precision** | ❌ calibrated non-win (2026-07-18) | `ndwi_mask.py` + `--open-water` are built, correct, and **off by default**. The premise ("FPs sit on blank water") is FALSE: glint/foam/swell reflect NIR *more* than 1–5 px boats (FP min-NDWI median −0.122 vs TP −0.027). At −0.10 it drops 7/20 FP but **60/63 TP** (P 0.76→0.19). A water *index* cannot reject real spectral water *features*. |
| **imgsz 1280 (plain, no TTA)** | ❌ non-win | Port Said identical; Alonnisos recall **0.70→0.49**. Upscaling past the 1024 *train* size loses the 1–5 px vessels it was meant to enlarge. 1536 not worth testing. The big-ship lift comes from **`--augment` TTA**, not raw imgsz. |
| **Piling on hard negatives** (`finetune-reef`) | ⚠️ counter-lesson | Recall *and* base mAP both dropped — the detector went over-conservative. Hence `--max-neg-ratio`. Add **positives** to fix recall; use negatives surgically. |
| **Plain Lee despeckling for SAR** | ❌ measured non-win (2026-08-04) | Best *pixel contrast* of anything tried (+41%) and the **worst detection**: recall ceiling 0.696→**0.391**. Smoothing a compact target's peak destroys it even as peak-to-background ratio improves. Use the **peak-preserving** variant (`--mode peak`). |
| **Trusting a one-region A/B to identify a *mechanism*** | ❌ actively misleading (2026-08-04, W10) | On 23 boxes the despeckle A/B did not merely overstate an effect — it named the **wrong one** (precision instead of recall) and got the sign of the FP change backwards. A thin gate can mislead about *what* an intervention does, not just *how much*. Widen the gate before drawing a causal conclusion. |
| **Pixel SNR / contrast as a proxy for detector quality** | ❌ actively misleading (2026-08-04) | W6 measured +41% contrast for the filter that *hurt* detection most. Contrast and detection ranked in **opposite order** across three variants. Measure detection. |
| **Mixing interpreters across an A/B** | ❌ silent confound (found 2026-08-05, W11) | W10's despeckle arm was trained under the system Python (torch 2.5.1 / ultralytics 8.3.221) while its baseline used the venv (2.12.1 / 8.4.90). Nobody noticed for a full cycle. **Train and gate with `venv/Scripts/python.exe`.** |
| **Textbook CFAR / top-hat for SAR seeding** | ❌ fails by construction | `providers/s1.py`'s dB-percentile stretch is tuned for *display*: water sits mid-bright (median ~136/255) and vessels saturate at 255. CFAR assumes dark water. `sar_seed.py` instead thresholds the dual-pol geometric mean `sqrt(VV·VH)` at an adaptive `median + k·MAD` — robust and stretch-invariant. |
| **Writing a region off on a `sar_seed.py` zero** | ❌ evidence about the seeder (2026-08-05) | Two runs recorded as "seed found nothing" (Ras Tanura 0 candidates, Durban 1) held **65 real vessels** between them; re-running `detect_boats.py --weights best_sar.pt` over the *identical* chips found 15 and 12 candidates, and review confirmed 58 and 7. Seed new regions with the **model**. |
| **Warm-starting SAR from the optical `best.pt`** | ❌ wrong priors | Different sensor, different texture statistics. Use `--pretrained yolov8n`. |
| **One more SAR scene to fix generalization** (`sar-deep2`) | ❌ regressed | +Singapore alone dropped the Gibraltar gate P 0.392→0.285. SAR needs *regions*, not one more scene. |
| **`sentinel-1-grd` as the SAR collection** | ❌ incompatible | PC's GRD assets are in radar geometry: `crs=None`, identity transform, geolocation only via 189 GCPs → `transform_bounds` crashes. Use **`sentinel-1-rtc`**. |
| **Per-vessel S1×S2 coincidence matching for MOVING vessels** | ❌ structurally impossible (2026-08-03) | Minimum S1↔S2 gap over one point is ~4.4 h (measured, 10 AOIs), during which a 12 kn vessel moves ~98 km. No date or AOI choice fixes it. See the W7 section above. |
| **Raising `--conf` to fix precision** | ❌ wrong tool | Trades away the recall the 10 m GSD already caps. The architecture is deliberately *recall-favouring candidate generation, then filter downstream* (water mask → AIS fusion). |
| **Longer training (60 ep) to squeeze more from a warm start** | ❌ non-win + forgetting | Satellite val mAP50 rose 0.785→0.798 but holdout F1 did **not** follow (0.705→0.706), and base-photo recall *fell* 0.585→0.556. Val mAP50 is a poor proxy for the holdout gates. Also: `close_mosaic` counts back from the END of training — **scale it proportionally with `--epochs`** or the mosaic-free phase never arrives. |
| **batch 16 @ imgsz 1024 with the P2 head** | 💥 crashes | Silent torch C++ abort mid-epoch. Use **`--batch 4`** (plateaus ~7 GB). |
| **Paid/commercial imagery (Planet, VHR)** | ⛔ out of scope | Hard $0-budget constraint. Refs were removed deliberately; do not reintroduce. |

---

# REFERENCE

## How the phases were built

- **Phase 0 — detector foundations.** Base ~621 overhead boat photos → active-learning loop on real
  Sentinel-2 chips. The dominant early lever was **inference resolution**: upscaling the *existing*
  model to imgsz 1024 lifted holdout recall 0.31→0.73 on its own, and the retrain then bought ~45%
  fewer FPs at matched recall. The **P2 stride-4 head** stacked on top of that. Then real, diverse
  training data proved highest-leverage — twice (cross-region via open-ocean, cross-type via
  small-vessel Med) — until W1/W4 found the GSD floor.
- **Phase 1 — provider seam.** `sat_fetch.py` became a thin orchestrator; source-specific steps
  (`open_catalog` / `search_candidates` / `assess_aoi` / `read_aoi` / `scale_to_uint8`, plus the
  optional `postprocess_chip` added in W12) live on a `Provider` under `scripts/providers/`. Adding a
  source = subclass + `register()`. `output_subdir` keeps each modality's runs in its own root so a
  SAR chip can never reach the optical training set.
- **Phase 2 — Sentinel-1 SAR.** Free on the same STAC, no credentials. Needs its **own** model
  (`best_sar.pt`), its own training tree (`data/training_sar*`), and its own gate. Progression:
  `sar-deep` → `sar-deep2` (rejected) → `sar-deep3` (W2, first decisive) → `sar-deep4` (W11) →
  `sar-peak2` (W12, current).
- **Phase 3 — AIS dark-vessel fusion (the product).** All stages built and offline-tested pure-stdlib;
  live-validated 2026-07-26; run end-to-end on real data 2026-08-06 (W3-B). Its definition of done is
  met. The three statuses — `dark` / `matched` / `reported` — are the product's actual output.

## Quick reference — the gating workflow

Everything else (per-script contracts, the full command list, first-run setup) is in **`CLAUDE.md`**;
the copy-paste recipes with their environment quirks are in **`docs/STATUS.md` → Operational recipes**.

| Task | Command |
|---|---|
| Pick the operating point **first** | `conf_sweep.py --weights <run>/weights/best.pt --imgsz 1024` — sweep **both** models; reading a new model at the old model's conf measures the threshold, not the model |
| Gate a candidate | `eval_holdout.py --old <old>.pt --new <new>.pt --old-imgsz 1024 --new-imgsz 1024 --holdout <scene> --conf <swept>` — **loop `--holdout` over all four** optical scenes (or both SAR scenes) |
| Gate a SAR candidate | add `--train-dir data/training_sar_peak --fp-scenes s1deep-fujairah`, and pass `--old/--new` explicitly (that tree is not in `PROMOTED_BY_TREE`) |
| Promote | back up the outgoing model first (`weights/best_*.pt` is the audit trail), then copy the run's `best.pt` over `data/training/weights/best.pt` |
| Train optical | `train.py --weights data/training/yolov8n-p2.yaml --pretrained best --imgsz 1024 --batch 4 --mosaic 0.5 --close-mosaic 15 --epochs 30 --name <n>` |
| Train SAR | same but `--pretrained yolov8n --data data/training_sar_peak/config.yml` (**never** `--pretrained best`) |
| Add an imagery source | subclass `Provider` (`providers/base.py`), `register()` it, set `output_subdir` |

## Key sources
Small-object: SAHI 2202.06934, copy-paste 1902.07296, NWD 2110.13389, P2/SOD-YOLOv8 2408.04786.
Imagery: Kanjir survey (PMC5877374), S2-YOLO RSE 2025. Dark vessels: ESA/GFW (75% untracked),
Skylight (skylight.global/platform), xView3-SAR 2206.00897.
