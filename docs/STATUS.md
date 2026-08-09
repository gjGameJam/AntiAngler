# STATUS — boat detector / active-learning loop

_Snapshot: **2026-08-06**. This is the **live state**: which model is promoted, what the gates say,
what is broken. For what was tried and what not to retry, see `ROADMAP.md`; for the build spec and
data contracts, `PIPELINE.md`; for code-health findings, `AUDIT.md`; for per-script contracts and the
full command list, `../CLAUDE.md`. Point-in-time — verify against code before trusting specifics._

> **2026-08-06 — SPLIT LEAK FOUND AND FIXED (affected every model since 2026-07-26).**
> `build_dataset.py` purged previously-layered files by the `__` marker in `<run>__<chip>`. The
> Finland ingester names its chips `finlandS2_…` — **no `__`** — so they survived every purge while
> the seeded split kept reassigning them: **510 of 1034 Finland chips ended up in BOTH `train` and
> `val`**. **The holdout TEST split was never affected** (it is scene-filtered `__`-named chips), so
> **every `eval_holdout.py` number and every promotion decision remains valid**; what was contaminated
> is the val set YOLO uses for **best-epoch selection**. Fixed by purging on **provenance** (stem
> present in an export tag); regression-tested in `scripts/tests/test_build_dataset.py`, verified to
> fail against the pre-fix code. ⚠️ **Val / `results.csv` numbers recorded before 2026-08-06 are not
> comparable to later ones.**

---

## TL;DR — current state

- **Live optical model:** `data/training/weights/best.pt` = **`finetune-w4ionian`** (promoted
  **2026-08-06**, ROADMAP W4), **`imgsz 1024`, `conf 0.20`**. The deploy conf did **not** move at this
  promotion — the first time — but it was still re-swept on all four holdouts for both models;
  "unchanged" is a measurement, not an inheritance. Promoted as a **recall win at held false-alarm
  load**: pooled **216 TP / 149 FP vs the incumbent's 209 / 149** — seven more vessels for the same
  number of false alarms. Backup: `best_finetune-adriatic.pt`.
  ⚠️ **Watch items, all of which moved the wrong way:** base mAP50 **0.640→0.603** (the "no forgetting"
  number prior promotions cited; base *recall* is flat 0.590→0.594 and every center-distance metric
  improved, but it is recorded, not buried); **Alonnisos recall down at both confs** (0.855→0.829
  @0.20); the **3–5 px stratum 0.846→0.538** on a 13-box sample. A precision-leaning alternative was
  available and not taken: at conf 0.25 the same model gives 198 TP / **79** FP (−47% false alarms)
  for 11 fewer vessels.
- **Live SAR model:** `data/training/weights/best_sar.pt` = **`sar-peak2`** (promoted **2026-08-06**,
  ROADMAP W12), **`imgsz 1024`, `conf 0.20`**. 🚨 **The SAR pipeline is DESPECKLED end-to-end** — every
  S1 fetch must pass `sat_fetch.py --provider s1 --despeckle peak`. Gibraltar **P 0.576 / R 0.826**
  (recall CI **[0.72, 0.94]**), Santos **P 0.618 / R 0.475**; pooled **66 TP / 43 FP** vs `sar-deep4`'s
  54 / 101. Backup: `best_sar_deep4.pt`.
- **Phase 3 (AIS fusion):** built, offline-tested, live-validated (2026-07-26) and **run end-to-end on
  real data** (2026-08-06) — all three statuses in one scene, now **human-reviewed**: **precision
  0.750** (6 TP / 2 FP), **recall 1.000** (6/6) at conf 0.20. The ORION AIS match is confirmed real.
  🚨 **But the one in-MPA dark candidate was a false positive on a nearshore rock** — see "Phase 3"
  below before citing this run. **The product has no validated in-MPA dark vessel yet.**
- **Nothing is queued.** W1–W12 are all closed. See `ROADMAP.md` → RESULTS. The one **open question**
  (not a work item, nobody has scoped it): find a scene that holds a *real* in-MPA dark vessel — that
  is the last unproven link in the product story.

### The two things most likely to bite next session

1. **Every SAR run on disk predates W12 and holds RAW chips** → out of domain for `sar-peak2`.
   A preprocessing mismatch costs **about two-thirds of recall and raises no error** (measured:
   `sar-deep4` on despeckled chips scores Gibraltar R 0.217 instead of 0.609). Re-fetch with
   `--despeckle peak`, or re-filter with `sar_despeckle.py --run <dir>`.
   `manifest["chip_postprocess"]` says which a run is; a missing key means raw.
2. **AISStream authenticates but delivers zero frames** (reproduced 2026-08-06: invalid key dropped in
   1.1 s, real key held open 46.7 s with 0 frames, while a third-party push socket delivered 3 frames
   in 0.3 s — so the local TLS path is fine). Account/key side. Historical AIS via
   `marinecadastre_fetch.py` is unaffected and is what W3-B actually used.

---

## The gates

**Optical — four holdout scenes, 300 boxes.** Live `finetune-w4ionian` @ `imgsz 1024 / conf 0.20`:

| Holdout | Type | Boxes | P | R | recall 90% CI |
|---|---|---|---|---|---|
| **Kornati** (Adriatic) | cross-region small-vessel | 156 | 0.550 | **0.635** | [0.56, 0.71] |
| Alonnisos (Aegean) | cross-type small-vessel | 76 | 0.500 | 0.829 | [0.75, 0.94] |
| Port Said | big-ship dense anchorage | 42 | 0.943 | 0.786 | [0.67, 0.91] |
| Singapore Jul-5 (`fallbacktest`) | cross-date | 26 | 0.875 | 0.808 | [0.71, 0.89] |

Pooled **216 TP / 149 FP**; base mAP50 0.603, base recall 0.594, open-water `testrun` probe 29.

**SAR — two regions, 122 boxes, on the DESPECKLED tree** (`--train-dir data/training_sar_peak`;
that tree is **not** in `PROMOTED_BY_TREE`, so pass `--old/--new` explicitly):

| holdout | Boxes | P | R | F1 | FP/chip | recall 90% CI |
|---|---|---|---|---|---|---|
| s1deep-gibraltar | 23 | 0.576 | **0.826** | 0.679 | 0.40 | **[0.72, 0.94]** |
| s1warm-santos | 99 | 0.618 | 0.475 | 0.537 | 0.59 | [0.34, 0.58] |

⚠️ **Gate on ALL holdouts.** `eval_holdout.py --holdout` takes ONE scene and defaults to
`fallbacktest`, so a bare invocation silently reports 1/4 of the optical evidence. **Loop it.** Its
`--conf` also defaults to 0.25, which is nobody's deploy point — **always pass `--conf` explicitly**,
taking the value from `conf_sweep.py` on that tree.
⚠️ **Do not judge SAR precision on the in-TRAIN Fujairah probe** — TRAIN holds three Gulf families, so
a better-fitted model fires *more* there. Judge on holdout FP/chip.
⚠️ **The `1-2px` / `3-5px` recall strata are empty on SAR and always will be.** Measured: SAR TEST
boxes are min **12 px**, median **17.9 px**, none below 11 px. Those bin edges are an *optical* 10 m-GSD
construct; reading "0/0" there as a failure is a misread. SAR needs its own bin edges to say anything.

---

## What works

1. **End-to-end loop, all in-repo** (exact commands in `../CLAUDE.md`): `sat_fetch.py` →
   `detect_boats.py` → `review_server.py` → `export_labels.py` → `build_dataset.py` → `train.py` →
   promote → rescan.
2. **GPU fine-tuning** via the AntiAngler venv, ultralytics 8.4.90 / torch 2.12.1+cu130 on the
   RTX 4070. A 30-epoch 1024px P2 run takes **~25 min**. **Keep the machine awake** — sleep inflates
   wall-clock massively. ⚠️ **Use `venv/Scripts/python.exe` for training AND gating**; the system
   Python carries torch 2.5.1 / ultralytics 8.3.221 and mixing the two silently confounded W10's A/B.
3. **Trustworthy tiny-object eval** — `eval_holdout.py`: instance-level P/R by center-distance
   matching (IoU-mAP is unstable at 1–5 px), FP/chip, recall stratified by GT pixel size, bootstrap
   recall CI. Modality-agnostic via `--train-dir`. `conf_sweep.py` runs one low-floor pass then
   re-thresholds to give an instance-level PR curve — that is how the deploy conf is chosen.
4. **Review UI** — `review_server.py` shows every chip **twice**: the labelling pane with boxes, and a
   clean un-boxed copy beside it (a 2 px box outline can be wider than a 1–5 px vessel). SAR runs get
   a third pane, the co-registered optical companion, so land is unambiguous under speckle.
5. **Fetching under Avast TLS interception** — solved; recipe below.

---

## Model progression

_Each row is a promoted or rejected retrain. Only the last two carry full detail; the earlier rows are
kept for the shape of the curve, not for reproduction._

| Run (`data/training/runs/<name>`) | Data added | Result | Promoted? |
|---|---|---|---|
| _pre-AL baseline_ | base ~621 overhead photos only | SG 0.035 mAP50 / 0.077 R | was baseline |
| `finetune-multiscene` (07-10) | +SG Jun-13 + False Bay | 0.167 / 0.269 | superseded |
| `finetune-reef` (07-11) | +GBR Cairns reefs (35) | 0.129 / 0.215 — **regressed** | ❌ the negatives counter-lesson |
| `finetune-aug` (07-11) | +copy-paste synthesis | 0.444 / 0.31 | superseded |
| `finetune-aug-1024` (07-11) | same data at **imgsz 1024** | 0.741 / 0.73 | superseded — **resolution was the dominant early lever** |
| `finetune-p2-1024` (07-12) | **P2 head** (stride-4) @ 1024 | 0.773 / 0.77 | superseded — the arch in use today |
| `finetune-openocean` (07-12) | +Fujairah/Gibraltar real open-ocean | Port Said R **0.00→0.67** | superseded — proves cross-*region* |
| `finetune-smallvessel` (07-14) | +Cyclades/Egadi small-vessel Med | Alonnisos R **0.31→0.67** | superseded — proves cross-*type* |
| `finetune-bigship` (07-26) | +Gulf-of-Finland S2 (1,034 chips / 7,984 boxes) | Alonnisos R 0.18→**0.67** | superseded |
| `finetune-diverse` (07-26) | +11 reviewed scenes (13-set harvest) | recall up on **all 4** holdouts | superseded |
| `finetune-adriatic` (08-02) | +3 Adriatic scenes (**1093 boxes**) | **precision win, not recall**: open-water FP −45%, base mAP50 0.614→0.640 | superseded (→ `_adriatic` backup) |
| **`finetune-w4ionian` (08-06)** | +4 Ionian/Adriatic/Aegean scenes (**490 boxes** / 100 chips) | **recall win at held FP load**: pooled **216 TP / 149 FP vs 209 / 149**; Kornati R 0.590→**0.635**, fallbacktest 0.731→**0.808**, Port Said FP/chip 0.33→**0.17**. ⚠️ base mAP50 0.640→**0.603**, Alonnisos R 0.855→0.829 | ✅ **LIVE** (conf **0.20**) |

Two rejected schedules are worth remembering: `finetune-adriatic-60ep` early-stopped at 21 with
"best = epoch 1" because `--close-mosaic 15` was not scaled with `--epochs 60` (it counts back from the
END of training), and `-60ep-b` got the best Kornati recall but lost base recall 0.585→0.556 —
forgetting. **Scale `--close-mosaic` with `--epochs`.**

**Command for the live model:**
```bash
venv/Scripts/python.exe scripts/train.py --weights data/training/yolov8n-p2.yaml --pretrained best \
    --imgsz 1024 --batch 4 --mosaic 0.5 --close-mosaic 15 --epochs 30 --name finetune-w4ionian
```
⚠️ **`--batch 4`** — batch 16 @1024 with the P2 head is a silent torch C++ abort mid-epoch.

---

## Training data

**Optical `data/training/`** — **train 1818 / val 520 / test 159** (621 base boat photos + 1876
active-learning chips across 12 export tags), assembled by:
```bash
python scripts/build_dataset.py \
    --holdout-scene fallbacktest portsaid alonnisos kornati-adriatic --max-neg-ratio 1.0 --seed 42
```
The four holdout scenes are forced whole into TEST and never appear in TRAIN. `--max-neg-ratio 1.0` is
the `finetune-reef` safeguard (too many hard negatives → over-conservative → recall drops).

**SAR `data/training_sar/`** — **train 122 / val 76 / test 84**, no base-photo layer, purely reviewed
SAR chips from **8 export tags** (`deep`, `singapore`, `santos`, `kaohsiung`, `suezgulf`, `jebelali`,
`durban`, `rastanura`). TEST = Gibraltar 35 chips (23 boxes) + Santos 49 chips (99 boxes).
**`data/training_sar_peak/`** is the despeckled twin that `sar-peak2` trains and gates on — same
splits, byte-identical labels.

| Scene (run) | Region class | Role |
|---|---|---|
| `…s1deep-fujairah` | deep anchorage, Gulf | TRAIN+VAL (also the FP probe — in-sample, directional) |
| `…s1deep-hormuz` | lane chokepoint, Gulf | TRAIN+VAL |
| `…s1deep-singapore` | strait, SE Asia | TRAIN+VAL |
| `…s1warm-kaohsiung` | port approach, Taiwan | TRAIN+VAL |
| `…s1warm-suezgulf` | Red Sea / Suez | TRAIN+VAL |
| `…s1warm-jebelali` | port, Gulf | TRAIN+VAL |
| `…s1warm-durban` | port approach, S. Africa | TRAIN+VAL — **non-Gulf, the diversity buy** |
| `…s1warm-rastanura` | oil terminal, Gulf | TRAIN+VAL |
| `…s1warm-santos` | port approach, Brazil | **TEST holdout #2** (99 boxes) |
| `…s1deep-gibraltar` | strait, Atlantic/Med | **TEST holdout** (23 boxes = 15 distinct vessels) |

⚠️ **Seed a NEW SAR region with the MODEL, not `sar_seed.py`.** Two regions were written off on a
classical-seed zero (Ras Tanura 0 candidates, Durban 1); re-running `detect_boats.py --weights
best_sar.pt` over the *identical* chips gave 15 and 12 candidates, and review confirmed **58 and 7**
real vessels. A seed reporting "nothing" is evidence about the seeder, not about the water.

---

## Phase 3 — the AIS fusion product

Built, offline-tested (pure stdlib), live-validated 2026-07-26, and **run end-to-end on real data
2026-08-06** over the Santa Barbara Channel (S2 2024-08-23 18:39:19Z, 0% AOI cloud, 30 chips; NOAA
MarineCadastre AIS filtered to ±1 h → 2763 positions / 74 vessels). All three statuses in one scene:

- **matched (1)** — ORION, MMSI 368109370: AIS ping at 18:38:52Z vs acquisition 18:39:19Z, a
  **27-second** offset. At 5.4 kn her feasible-movement envelope was 575 m; the detection sat 450 m
  away. The envelope design validating itself on real data. ✅ **Detection confirmed real by review**
  (2026-08-06), so the match is validated end to end and not a coincidental hit on a rock.
- **dark (7 claimed → 5 real)** — see the review result below.
- **reported (2)** — ISLAND EXPLORER (1.21 h inside Scorpion) and WHALE RIDER, by AIS geofencing alone.

### The review result (2026-08-06) — first honest end-to-end precision

All **30 chips** were human-reviewed in `review_server.py`. At `conf 0.20`, `finetune-w4ionian`:

| | count |
|---|---|
| detections emitted | 8 |
| confirmed real (TP) | **6** |
| false positives | **2** |
| **precision** | **0.750** |
| distinct vessels visible to the reviewer | 6 |
| **recall** | **1.000** (6/6) |

Three things that only reviewing could have told us:

1. 🚨 **The in-MPA dark candidate was a rock.** The single dark detection inside an MPA — `det_00003`,
   inside **Scorpion (Santa Cruz Island) SMR**, the one this run was cited for — is a **false
   positive** on a rock ~118 m off the Santa Cruz shore. Point-in-polygon against
   `wdpa_marine_ia_ib.gpkg` puts **none of the 5 confirmed dark vessels inside any Ia/Ib MPA**. The
   scene proves the *plumbing* end to end; it does **not** contain a validated in-MPA dark vessel.
   Do not cite it as one.
2. **The water mask is a partial fix — it does NOT rescue the in-MPA candidate.** Tested directly
   (2026-08-07): `water_mask.py` + `detect_boats.py --water-only` on this run dropped **1 of the 2**
   false positives, precision **0.750 → 0.857** at unchanged recall. The FP mix is **1 land / 1
   on-water**, not 2/2 land — the 2026-07-16 measurement (84% on-water / 16% land) is *not*
   overturned by an island scene:
   - `det_00007` (Anacapa spine) is **8 m** from mask land → dropped. ✅
   - `det_00003` (the **in-MPA** one) is **118 m offshore** — an awash rock / surf-and-kelp patch on
     open water beside a headland. WorldCover correctly calls it water, so **no land mask will ever
     catch it**. The FP that mattered is on-water.

   A **shoreline buffer** would clear both: all 6 TPs sit >460 m from land while both FPs are within
   118 m, so dropping detections within ~150 m of mask land gives 1.000/1.000 here. ⚠️ **Do not adopt
   that on this evidence** — n=8 on one scene, and a 150 m no-boat ring would gut recall on exactly
   the anchorage scenes the detector is gated on (Port Said, Singapore, Jebel Ali all hold vessels
   moored well inside 150 m of shore). It is a hypothesis with an obvious failure mode, not a lever.
   Kept as a measurement so nobody re-derives it from scratch.
3. **A "missed" box is not always a missed vessel.** The reviewer drew one box in
   `chip_r00000_c02021` — it is **7.6 m** from `det_00001` in the overlapping neighbour chip, i.e. the
   *same vessel*, correctly merged away by `--merge-dist 30` and correctly re-labelled per the
   complete-labels rule. Counting it as a miss would have understated recall as 0.857. **Always
   distance-check human boxes against the deduped detection list before calling anything a miss.**

**What is on disk in that run dir** (`data/raw/sentinel2/2026-08-06T18-01-57Z_sbchannel-anacapa-20240823/`):
`detections.json` is the **reviewed 8** and is canonical — `reviews.json` and `violation_events.json`
both refer to it, so leave it that way. `detections_wateronly.json` is the 7-detection `--water-only`
re-run kept for the record, and `water_mask.tif` is the cached WorldCover mask (no refetch needed).
`export_labels.py` reads only `reviews.json` + `manifest.json`, so the review is safe either way.
**The 30 reviewed chips have not been exported** — `export_labels.py --run <that dir>` would add 6
SB-Channel positives plus 24 hard negatives, including two genuine coastal-rock negatives of exactly
the class that caused both errors. Nobody has decided whether to fold them in.

⚠️ **"Dark" still does not mean "violation" here.** US AIS carriage is mandatory only for certain
vessel classes, and the Channel Islands are full of small dive and recreational craft that legally
carry none. The dark signal is only strong where any vessel is anomalous. Relatedly, `ais_fishing.py`
scored ISLAND EXPLORER 0.36 because slow meandering reads as fishing-like — which is exactly what a
dive or whale-watching boat does.

---

## Open problems

1. **Recall is GSD-capped.** Sentinel-2's 10 m renders sub-15 m vessels as 1–5 px specks — a physical
   floor, confirmed twice (W1 + W4: ~1580 in-class boxes left Kornati's recall *ceiling* flat). The
   structural fixes are SAR and the AIS-mismatch reframe, both built.
2. **Santos SAR recall 0.475** is the weakest gate number. The SAR gate is still only **two** regions;
   a third reviewed region buys more gate power than any tuning.
3. **`eval_holdout.py` defaults hide evidence** — `--holdout` defaults to `fallbacktest` (1 of 4
   optical scenes) and `--conf` to 0.25 (nobody's deploy point). W5 made a zero-match filter a hard
   error, but a filter matching the *wrong real scene* is still silent. Always loop `--holdout` and
   pass `--conf` explicitly.
4. **TLS interception** — every fetch needs the CA-bundle + `--insecure-tls` recipe below, and
   `--insecure-tls` must be user-authorized each session.
5. ~~The 8 SB Channel detections are unreviewed~~ — **CLOSED 2026-08-06.** All 30 chips reviewed:
   **precision 0.750** (6 TP / 2 FP), **recall 1.000** (6/6 distinct vessels). The ORION AIS match is
   confirmed real. 🚨 **But the one in-MPA dark candidate (Scorpion SMR) was a false positive on a
   rock** — the product loop has no validated in-MPA dark vessel yet. The water mask was then built
   and tested (2026-08-07): it lifts precision to **0.857** but **cannot** drop the in-MPA FP, which
   sits 118 m offshore on water. See "Phase 3" above. **The open successor question is now only one:
   find a scene that holds a real in-MPA dark vessel** — that is what the product still lacks.

---

## Operational recipes

**Fetch imagery under Avast TLS interception** (public read-only COGs; `--insecure-tls` is
user-authorized each session):
```bash
# 1. Build a combined CA bundle (certifi + the Avast root) once.
#    AVAST REGENERATES THIS ROOT PERIODICALLY - the bundle then goes stale and EVERY HTTPS call fails
#    with "unable to get local issuer certificate". Do NOT hardcode a thumbprint; look it up:
#      $c = Get-ChildItem Cert:\LocalMachine\Root | ? { $_.Subject -match 'Avast' }
#      $pem = "-----BEGIN CERTIFICATE-----`n" +
#             [Convert]::ToBase64String($c.RawData,'InsertLineBreaks') + "`n-----END CERTIFICATE-----`n"
#      Set-Content ca_avast_bundle.pem ((Get-Content (python -c "import certifi;print(certifi.where())") -Raw) + $pem) -Encoding ascii
#    Rotation history: CECD313F...DA80 (thru 2026-07) -> 27469C7BA79BA957EA60AAF214F47B72BE844D75
#    (still current 2026-08-06). Live bundle: scratch_logs/ca_avast_bundle_2026-08.pem
export CA="$PWD/scratch_logs/ca_avast_bundle_2026-08.pem"
export REQUESTS_CA_BUNDLE="$CA" SSL_CERT_FILE="$CA" CURL_CA_BUNDLE="$CA" GDAL_HTTP_CAINFO="$CA"
venv/Scripts/python.exe scripts/sat_fetch.py --bbox <minLon minLat maxLon maxLat> \
    --min-coverage 0.6 --label <slug> --insecure-tls
```
**`git push` hits the same wall.** Point git at the same bundle — do **not** disable verification:
```bash
git config --global http.sslCAInfo "C:/gtest/AntiAngler/scratch_logs/ca_avast_bundle_2026-08.pem"
```

**Gate and promote an OPTICAL candidate** (sweep first — the deploy conf has moved both up and down,
and once stayed put; never inherit it):
```bash
# 1. Sweep BOTH models. Reading a new model at the old model's conf measures the threshold, not the model.
venv/Scripts/python.exe scripts/conf_sweep.py --weights data/training/runs/<run>/weights/best.pt --imgsz 1024
# 2. Gate at the swept conf, LOOPING all four holdouts (--holdout takes ONE scene).
for H in kornati-adriatic alonnisos portsaid fallbacktest; do
  venv/Scripts/python.exe scripts/eval_holdout.py \
      --old data/training/weights/best.pt --old-imgsz 1024 \
      --new data/training/runs/<run>/weights/best.pt --new-imgsz 1024 \
      --holdout "$H" --conf <swept>
done
# 3. Promote only if it holds up — back up the outgoing model first (weights/ is the audit trail).
cp data/training/weights/best.pt data/training/weights/best_<outgoing-run>.pt
cp data/training/runs/<run>/weights/best.pt data/training/weights/best.pt
```

**The SAR flow.** `build_dataset.py` and `train.py` are modality-agnostic via `--data`/`--exports` —
there is no separate SAR script. The SAR set is a **separate dir tree** so the optical default can
never sweep SAR chips into `data/training/`:
```bash
# 1. Assemble. BOTH holdout scenes and the neg cap are REQUIRED - omitting them silently rebuilds a
#    one-region gate (the mistake that hid sar-deep's 0/99 on Santos) and an uncapped negative pile.
python scripts/build_dataset.py --exports data/processed/sar_training_exports --data data/training_sar \
    --holdout-scene s1deep-gibraltar s1warm-santos --max-neg-ratio 1.0
# 2. Rebuild the DESPECKLED tree that the live model trains on. rm -rf FIRST: sar_despeckle.py does not
#    overwrite an existing --dst, so a rebuild would leave orphan chips from the old split behind
#    (measured in W11: 84 orphans, 42 of which had changed split).
rm -rf data/training_sar_peak
python scripts/sar_despeckle.py --src data/training_sar --dst data/training_sar_peak   # --mode peak
# 3. Train. NEVER --pretrained best (optical texture priors are wrong). Use the venv interpreter.
venv/Scripts/python.exe scripts/train.py --weights data/training/yolov8n-p2.yaml --pretrained yolov8n \
    --data data/training_sar_peak/config.yml --imgsz 1024 --batch 4 \
    --mosaic 0.5 --close-mosaic 15 --epochs 30 --name sar-peak3
# 4. Sweep + gate, LOOPING both holdouts. The peak tree is not in PROMOTED_BY_TREE -> pass --old/--new.
for H in s1deep-gibraltar s1warm-santos; do
  venv/Scripts/python.exe scripts/conf_sweep.py --train-dir data/training_sar_peak \
      --weights data/training/runs/sar-peak3/weights/best.pt --holdout "$H" \
      --fp-scene s1deep-fujairah --imgsz 1024
  venv/Scripts/python.exe scripts/eval_holdout.py --train-dir data/training_sar_peak \
      --old data/training/weights/best_sar.pt \
      --new data/training/runs/sar-peak3/weights/best.pt \
      --holdout "$H" --fp-scenes s1deep-fujairah --imgsz 1024 --conf <swept>
done
# 5. Promote (back up first). Runs land under data/training/runs/<name>/ regardless of --data.
cp data/training/weights/best_sar.pt data/training/weights/best_sar_<outgoing>.pt
cp data/training/runs/sar-peak3/weights/best.pt data/training/weights/best_sar.pt
```
🚨 **Fetch SAR with `--despeckle peak`** or the promoted model is out of domain:
```bash
venv/Scripts/python.exe scripts/sat_fetch.py --provider s1 --bbox <...> --despeckle peak --insecure-tls
```

---

## Repo state

Work is on branch **`sar-deep3-and-roadmap-closeout`**, not merged to `main`. The 2026-08-06 session
(W4, W12, W3-B, the W3-B chip review, the split-leak fix, and this documentation pass) is
**uncommitted**.
