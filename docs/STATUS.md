# STATUS — boat detector / active-learning loop

_Snapshot: 2026-07-12. Point-in-time; verify against code before trusting specifics._

This tracks the live state of the satellite ship-detection model and its active-learning
loop. For the build spec + data contracts see `PIPELINE.md`; for known antipatterns see
`AUDIT.md`; for the daily GFW/WDPA streams see `../CLAUDE.md`.

---

## TL;DR

- **Live model:** `data/training/weights/best.pt` = the **`finetune-openocean`** run (promoted 2026-07-12), **deployed at `imgsz 1024`, `conf 0.15`** — the YOLOv8n-**P2** arch (stride-4 head, strides [4,8,16,32]) retrained with real human-reviewed **open-ocean** training data (Fujairah + Gibraltar), warm-started from the prior P2 backbone. `detect_boats.py` defaults bake this in (in-repo `best.pt`, conf 0.15). Backups: `best_finetune-p2-1024.pt` (prior live, P2 Singapore-only), `best_finetune-aug-1024.pt`, `best_finetune-aug-640.pt`, `best_multiscene.pt`, `best_baseline_pre-multiscene.pt`.
- **The whole in-repo active-learning loop works** end-to-end: `sat_fetch → detect_boats → review_server → export_labels → build_dataset → train → promote → rescan`.
- **Best result (imgsz 1024, plain-n):** on the clean held-out scene, deployment before/after — old (OLD@640, conf 0.25) → aug-1024 (NEW@1024, conf 0.20): instance **recall 0.31 → 0.73**, **precision 0.53 → 0.76**, **FP/chip 1.75 → 1.50**. The recall jump is the first statistically clean one (bootstrap CIs fully separate).
- **P2 head (Phase 0 remainder, promoted 2026-07-12) — the last free detector lever, and it STACKED.** A stride-4 P2 detection head trained fresh at 1024 (warm-started backbone) beats the prior plain-n `aug-1024` model on the same holdout at the shared deploy point conf 0.20: **recall 0.73 → 0.77** (20 vs 19 TP), **precision 0.76 → 0.87**, **FP/chip 1.50 → 0.75** (half the false positives). At its F1-peak conf 0.15 it reaches **recall 0.85 / precision 0.85 / FP-chip 1.0** — beating live @0.20 on every axis. No base-domain regression (base mAP50 0.634 → 0.658, base R 0.618 → 0.649) and open-water FP probe `testrun` **37 → 7**. Chose conf 0.20 for a conservative FP posture. **Caveat unchanged:** same thin 26-box single-location (Singapore) holdout — recall CIs still overlap ([0.44,0.68] vs [0.43,0.88]), so magnitude is uncertain; a fresh cross-location scene is still needed to confirm. Cost: P2 is ~1.5× slower (12.2 vs 8.1 GFLOPs).
- **Open-ocean retrain (promoted 2026-07-12, LIVE) — the data-collection loop, decisively validated.** Cross-validating the P2 model on FRESH open-ocean scenes exposed a catastrophic generalization gap: on a held-out dense-anchorage scene (Port Said, 42 boxes) it scored **recall 0.00** (0/42 @ conf 0.25) — the 0.77 Singapore number was wildly optimistic. Fix: fetched + human-reviewed real open-ocean scenes (Fujairah **131** boxes + Gibraltar **29**) through the in-repo loop, folded them into TRAIN (Port Said + Singapore held out via the new `build_dataset.py --holdout-scene`), retrained the P2 arch (warm-started, batch 4, 30 ep / 7 h). On the Port Said holdout (never trained on): **recall 0.00 → 0.67** @ conf 0.25 (**0.71 @ conf 0.15**, the swept F1 peak) at **precision 0.94–1.00, FP/chip 0–0.17**; recall CIs cleanly separated ([0,0] vs [0.52,0.79]). **No forgetting:** Singapore recall flat within noise (0.65→0.62) with precision UP (0.85→0.94), FP/chip DOWN (0.75→0.25); base mAP50 0.59→0.68. Two open-ocean scenes taught the model to detect a third in a new region. **Caveats:** Port Said is dense-anchored (same vessel *type* as Fujairah-in-train) → proves cross-*region*, not yet cross-*type* (small fishing boats untested); still one 42-box scene. **Next: scale the loop — more diverse open-ocean scenes, esp. small fishing vessels.**
- **The key finding — inference resolution is the dominant lever, not the retrain.** Just running the *existing* model at `imgsz 1024` (no training) lifted holdout recall **0.31 → 0.73** (1–5 px vessels → 2–10 px stop being sub-cell). The 5-h retrain at 1024 then bought a *cleaner frontier*: at matched recall (0.73) it has **~45% fewer false positives** than the raw-upscaled old model (FP/chip 1.5 vs 2.75). A conf sweep (`scripts/conf_sweep.py`) put the operating point at 0.20 (F1 near-peak; recall-vs-conf is steep, FP-vs-conf shallow).
- **Earlier win (augmentation, superseded as live):** `scripts/augment_positives.py` water-gated copy-paste fixed the 11-vs-58 boat/empty imbalance → 113 boat chips / 722 boxes; that model is now the `_640` backup.
- **Next levers:** (1) **scale the open-ocean data loop** — more diverse scenes, especially *small fishing vessels* (cross-type generalization is still unproven); (2) PlanetScope 3 m (Phase 1) for the physical GSD ceiling; (3) the AIS dark-vessel reframe (Phase 3, the operational payoff). The P2 head + real open-ocean data closed the biggest detector gaps; remaining gains are more/diverse data + better imagery. See `docs/ROADMAP.md`.

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
| **`finetune-openocean` (07-12)** | +Fujairah(131)+Gibraltar(29) **real open-ocean** | SG R 0.62 (held); **Port Said open-ocean R 0.00→0.67**⁴ | ✅ **LIVE** |

¹ `finetune-aug`/`finetune-aug-1024` numbers are from the rebuilt instance-level eval (mAP50 / instance-recall); earlier rows show the old IoU-mAP50 / mAP-recall. `finetune-aug` (imgsz 640, conf 0.25) beat the multiscene model on recall (0.19→0.31), precision (0.39→0.53), FP/chip (2.0→1.75). Trained from the *original baseline* on the rebalanced set, `--mosaic 0.5 --close-mosaic 15`.

² `finetune-aug-1024` = the same recipe retrained at **imgsz 1024**, evaluated + **deployed at imgsz 1024, conf 0.20**. Recall 0.73 (19/26) at P 0.76, FP/chip 1.5. The imgsz-1024 *inference* is the dominant lever (upscaling the old model alone gave recall 0.73, CIs fully separate from 640's 0.31); the retrain bought ~45% fewer FPs at matched recall. Operating point chosen via `scripts/conf_sweep.py` (F1 peaks ~0.15–0.20). **Requires imgsz-1024 inference to realize** — `detect_boats.py` defaults updated accordingly.

³ `finetune-p2-1024` = a **YOLOv8n-P2** arch (`data/training/yolov8n-p2.yaml`, stride-4 head added) trained **fresh** at imgsz 1024, 30 epochs, warm-started from the prior `best.pt` backbone via `train.py --pretrained best` (219/437 layers transferred; the new P2 branch + Detect heads init random). Command: `train.py --weights data/training/yolov8n-p2.yaml --pretrained best --imgsz 1024 --batch 4 --mosaic 0.5 --close-mosaic 15 --epochs 30`. **Note `--batch 4`** (not 16): batch 16 @1024 with the P2 head exhausted memory (~18 GB worker RSS → ~1 GB commit headroom → silent torch C++ abort mid-epoch-1, no traceback); batch 4 plateaus at ~7 GB. Ran 8.58 h on CPU. Holdout @conf 0.20: recall 0.77 (20/26) / P 0.87 / FP-chip 0.75; @conf 0.15 (F1 peak): 0.85 / 0.85 / 1.0. See `train.py --pretrained` (added for this: warm-start a fresh `.yaml` arch).

⁴ `finetune-openocean` = same P2 arch retrained after folding in **real human-reviewed open-ocean scenes** (Fujairah 131 + Gibraltar 29 boxes) fetched/reviewed through the in-repo loop. Port Said (dense-anchorage Med, 42 boxes) held out via `build_dataset.py --holdout-scene fallbacktest portsaid` (new flag; reproducible scene-aware split, no leakage). Instance recall on Port Said: **0.00 → 0.67** @ conf 0.25, **0.71 @ conf 0.15** (F1 peak per `conf_sweep.py --holdout portsaid`), precision 0.94–1.00, FP/chip 0–0.17. Singapore held (R 0.65→0.62, P 0.85→0.94, FP/chip 0.75→0.25); base mAP50 0.59→0.68. Command: `train.py --weights data/training/yolov8n-p2.yaml --pretrained best --imgsz 1024 --batch 4 --mosaic 0.5 --close-mosaic 15 --epochs 30`.

`finetune-reef` suppressed reef false positives but **lowered recall and base-domain mAP** — piling on hard negatives made the detector too conservative. That lesson drove Phase 0: fix the imbalance by adding *positives*, not more negatives. Left unpromoted in `runs/finetune-reef/`.

## Current training data (4 reviewed scenes, 3 export tags)

`data/processed/training_exports/{2026-07-08, 2026-07-10, 2026-07-11}` layered by `build_dataset.py`.

| Scene (run dir) | Location | Chips | Boat boxes | Role |
|---|---|---|---|---|
| `…11-36-29Z_bbox` | Singapore Jun-13 | 4 | ~153 | TRAIN (positives) |
| `…11-44-23Z_fallbacktest` | Singapore Jul-5 | 4 | 26 | **TEST holdout** (recall) |
| `…01-08-52Z_testrun` | False Bay ZA | 30 | 2 (+28 hard-neg) | TRAIN (open-water negatives) |
| `…04-32-01Z_gbr-cairns-reefs` | GBR Cairns | 35 | 6 (+30 hard-neg) | TRAIN (reef negatives) |

Balance problem in numbers: **~11 boat-bearing chips vs ~58 hard-negative chips.**

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

**Phase 0 is complete, and the cross-location "good enough" gate has now been run** — it exposed the
Singapore-only model's ~0 open-ocean recall and the fix (real open-ocean data → `finetune-openocean`,
LIVE at conf 0.15) closed it (Port Said 0.00→0.71). **The active-learning data loop is now the proven
engine.** `build_dataset.py --holdout-scene` is implemented (reproducible scene-aware split). **The full
forward plan — Phases 1–3 (PlanetScope 3 m, Sentinel-1 SAR, AIS dark-vessel fusion) — is in
`docs/ROADMAP.md`.** Short version, in impact order:
1. **Scale the open-ocean data loop — highest leverage, now proven.** Fetch/review more DIVERSE
   open-ocean scenes, especially **small fishing vessels** (Port Said/Fujairah are big anchored ships;
   cross-*type* generalization is unproven). Each fresh scene: hold it out, eval the live model on it
   first (the honest number), then fold in and retrain. Keep at least one fresh scene as a rotating
   holdout.
2. **Phase 1 (PlanetScope 3 m)** for the physical GSD ceiling, or **Phase 3 (AIS dark-vessel reframe)** —
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
