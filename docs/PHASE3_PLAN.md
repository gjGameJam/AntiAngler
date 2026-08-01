# PHASE 3 COMPLETION PLAN — AIS dark-vessel product

_Snapshot: **2026-08-01**. The plan for finishing Phase 3 (the AIS dark-vessel reframe). Read
`docs/ROADMAP.md` Phase 3 for the strategic framing and `CLAUDE.md` for the per-script contracts._

> ## ✅ PHASE 3 IS COMPLETE — every box in "Definition of done" is ticked
> Built + offline-tested 2026-07-23, **live-validated 2026-07-26** (AISStream, GFW v3, and a first
> real violation over Anacapa Island SMR), extended 2026-07-29 with three more offline stages.
> **This file is now mostly a record.** The only things still open are the *optional* Skylight
> cross-check (3B) and the near-real-time operational mode — both tracked as **W9** and **W3** in
> `docs/ROADMAP.md` → THE WORK QUEUE, which is where forward-looking work now lives.

## The product in one line

**dark-vessel = detection − AIS.** An in-MPA satellite detection with no matching AIS track is a
candidate violation; a matched one is an identified vessel (down-weighted unless GFW shows it fishing).
This inverts the metric from the detector's uncatchable *recall* (10 m GSD floor) to *mismatch
precision*, which the fusion improves.

## Built and offline-tested (updated 2026-07-23)

| Stage | Script | Output | Tests |
|---|---|---|---|
| AIS ingest | `aisstream_fetch.py` | `data/raw/ais/ais_positions_*.json` | `test_aisstream_fetch.py` (20) |
| Fusion (matcher) | `fuse_violations.py` | `violation_events.json` + dark `.geojson` | `test_fuse_violations.py` (24) |
| GFW identity/fishing | `gfw_vessels.py` (+ `--region` MPA filter, 3A) | `data/raw/gfw_vessels/*.json` | `test_gfw_vessels.py` (25) |
| Enrichment join | `fuse_violations.py --vessels` | enriched `violation_events.json` | (in the 24 above) |
| Operator output (2B) | `report_violations.py` | ranked CSV + self-contained HTML | `test_report_violations.py` (10) |
| Batch driver (2A) | `scan_violations.py` | sequences the stages | `test_scan_violations.py` (7) |
| Historical AIS (3C) | `marinecadastre_fetch.py` | archive CSV → AIS contract | `test_marinecadastre_fetch.py` (9) |
| Contract guard (4A) | `scripts/tests/fixtures/` | — | `test_pipeline_e2e.py` (3) |
| **Local fishing signal** (2026-07-29) | `ais_fishing.py` | vessel records, `source: ais-heuristic` | `test_ais_fishing.py` (19) |
| **AIS × MPA geofencing** (2026-07-29) | `geofence_ais.py` | `"reported"` events + `.geojson` | `test_geofence_ais.py` (26) |
| **Alerting interface** (2026-07-29) | `alert_violations.py` | severity digest (console / markdown) | `test_alert_violations.py` (29) |

**203 offline tests** across the whole suite (which also covers `gfw_fetch.py` and the shared
`_http.py` retry policy), no network / key / ML deps. The `violation_events.json` schema is pinned in
`docs/PIPELINE.md` (4B). The full chain:
`sat_fetch → detect_boats → aisstream_fetch → fuse_violations → gfw_vessels → fuse_violations --vessels → report_violations`
(or one `scan_violations.py` invocation).

**The three 2026-07-29 additions changed what the product can do without credentials.** `ais_fishing.py`
supplies the fishing term from AIS *movement* instead of the GFW API, and `geofence_ais.py` needs no
imagery at all — so **an AIS capture alone now produces a scored, alerting product loop** with no GFW
token, no overpass timing, and no GPU. That is the cheapest live demo available (ROADMAP **W3**, path A).

**Three statuses now exist**, and `report_violations.py` / `alert_violations.py` rank all three together
with no special-casing:

| Status | Meaning | Producer | Scored |
|---|---|---|---|
| `dark` | detected, **not** transmitting | `fuse_violations.py` | `presence_confidence` — the product signal |
| `matched` | detected **and** transmitting | `fuse_violations.py` | `presence × w`, `matched_weight` (0.25) → `fishing_weight` |
| `reported` | transmitting from inside the MPA, never detected | `geofence_ais.py` | transit floor (0.10) → `fishing_weight`; IUU escalates |

---

## Workstream 1 — Live validation  🏠 DO FROM HOME  (do first; everything else assumes the calls work)

> **Environment note (2026-07-23): live validation cannot run in the Claude Code web/remote sandbox.**
> The egress policy blocks all three service hosts at CONNECT (`gateway.api.globalfishingwatch.org`,
> `stream.aisstream.io`, `api-doc.globalfishingwatch.org` → 403 / unreachable), and no
> `AISSTREAM_API_KEY` / `GFW_API_TOKEN` is present. So even an unauthenticated path-existence probe
> (expecting 401) is impossible here — the gateway never accepts the tunnel. **Run 1A–1C from a machine
> (or an environment whose network policy allowlists these hosts) with the key + token set.** The
> endpoint paths below follow GFW's documented v3 API and the parsers are drift-tolerant; 1B is the
> checklist to confirm them against a real response.

### 1A. AISStream live run  ✅ DONE (2026-07-26, at home)

> **Result:** first live connect succeeded outright — Singapore Strait bbox, 120 s:
> **41 frames → 41 positions kept (37 vessels), 0 skipped, 0 errors**; records carry real
> names/sog/cog, all inside the bbox, timestamps live. The wire-format handling (lat/lon corner
> order, Go-format timestamps, Sog/Cog sentinels) is confirmed against the real stream, and the
> output file feeds `fuse_violations.py --ais` without error. **Workstream 1 fully closed.**
- **Do:** register a free key at https://aisstream.io → `AISSTREAM_API_KEY` in `.env`;
  `pip install -r requirements-ais.txt`; run `aisstream_fetch.py --bbox <a busy strait> --duration 120`.
- **Verify:** the output has `positions` with N>0 vessels; spot-check a few records' `mmsi/lat/lon/
  timestamp/sog` against the raw frame (add a temporary `print(raw)` if needed); check the `skipped`
  count isn't ~100% (would mean the frame shape differs from `normalize_aisstream_message`'s expectation).
- **Likely fixes:** real streams also send `ShipStaticData`/`StandardClassBPositionReport`; the filter is
  server-side (`FilterMessageTypes`), and `normalize_aisstream_message` already falls back to
  `StandardClassBPositionReport`. If MetaData lat/lon are ever absent, it falls back to the inner report.
- **Acceptance:** a positions file that `fuse_violations.py --ais` reads without error over a real AOI.
- **Effort:** S (mostly account setup).

### 1B. GFW live verification  ✅ DONE (2026-07-26, at home)  *(was the highest drift risk)*

> **Result:** all three endpoints verified against the live v3 API with active ARG trawlers
> (ERIN BRUCE II / MISS TIDE, harvested from a live 4WINGS pull — `gfw_fetch.py` re-confirmed 2xx
> post-hardening). Three corrections landed, each locked by a committed real-response fixture:
> (1) `/v3/events` 422s unless `offset` accompanies `limit`; (2) an MMSI resolves to several
> identity clusters — pick the freshest/busiest (`transmissionDateTo`/counters), gear lives in
> entry-level `combinedSourcesInfo[].geartypes[].name`; (3) insights include is
> `VESSEL-IDENTITY-IUU-VESSEL-LIST` → 201 flat object. The 3A region filter was verified
> ground-truth-correct live (one vessel's fishing inside a shelf bbox, the other's outside,
> matching their 4WINGS effort cells).
- **Do:** `GFW_API_TOKEN` in `.env` (same token `gfw_fetch.py` uses); `gfw_vessels.py --mmsi <known
  fishing MMSI> --dry-run` then without `--dry-run`; add `--insights` on a second run.
- **Verify each endpoint against the live v3 API and fix the isolated parser if the shape differs:**
  - `GET /v3/vessels/search` — confirm the param is `query` (vs `where`/`ssvid`) and that identity lives
    in `entries[].selfReportedInfo[]` / `registryInfo[]`. Parser: `normalize_identity`. Field aliases are
    already tolerated (`vesselId`/`id`, `shipname`/`name`, `geartype`/`geartypes`, `imo`/`imoNumber`).
  - `GET /v3/events` — confirm GET (vs POST) and the params `datasets[0]` / `vessels[0]` / `start-date` /
    `end-date`, and that fishing events have `type=="fishing"` with `start`/`end` (or `durationHours`).
    Parser: `summarize_fishing` + `event_duration_hours` (handles ISO strings and nested `{timestamp}`).
  - `POST /v3/insights/vessels` — confirm the body `includes` values (`FISHING`/`GAP`/`VESSEL-IDENTITY-IUU`)
    and the response nesting. Parser: `normalize_insights` (deliberately fuzzy — searches for `iuu`/`gap`/
    `notake` keys; **this is the least certain and should be eyeballed against a real response**).
- **Acceptance:** a `gfw_vessels_*.json` where at least one real MMSI resolves to a name/flag and a
  sensible `fishing_hours`. Record any endpoint corrections in `CLAUDE.md`'s `gfw_vessels.py` note.
- **Effort:** M. **Risk:** the parsers are tolerant (drift → null fields, not crashes), so this is
  verify-and-adjust, not rebuild.

### 1C. End-to-end real run over one MPA  ✅ DONE (2026-07-26, historical path)

> **Result:** Anacapa Island State Marine Reserve (WDPAID 352751), S2 2024-08-25T18:29Z (0% AOI
> cloud) fused against same-day NOAA MarineCadastre archive AIS (8,383 positions / 145 vessels,
> ±90 min window) via `scan_violations.py`. **2 in-MPA detections → 2 DARK**, one a visually
> confirmed wake vessel (the confirmed dark point). A same-scene bbox run over the adjacent
> Santa Barbara Channel lanes (`fuse --all-detections`) gave **43 detections → 36 matched / 7
> dark**, all 36 GFW-enriched live — incl. the confirmed matched point: BB PELAGIC (MMSI
> 338406714), AIS ping **2 s** from acquisition, **362 m** from the detection, underway 4.5 kn.
> The first fusion caught three real matcher/ingest defects, all fixed + regression-tested the
> same day: MarineCadastre passed AIS sentinel SOG 102.3/COG 360 through as real values; the
> matcher admitted stale pings a fresher ping refuted (a moored tour boat 21 km away "matched");
> and attribution favored huge fallback envelopes over the nearest qualifying vessel. The
> live-capture variant (AISStream rolling buffer over a fresh overpass) still waits on 1A.
- **Do:** pick an IUCN Ia/Ib MPA with traffic; capture AIS live (`aisstream_fetch --wdpa-id <id>
  --duration 600`) **while** fetching a fresh scene over the same AOI, then
  `detect_boats → fuse_violations --ais … → gfw_vessels --from-events … → fuse_violations … --vessels …`.
- **Timing caveat (the real constraint):** imagery is a *past* overpass; AISStream is *live-only*. Fusion
  is only valid when `scene.datetime` falls inside the AIS capture window. In practice run
  `aisstream_fetch` as a rolling buffer and fuse when a fresh overpass lands — or use a historical AIS
  source (3C). Widen `--dt-minutes` only as far as vessel speeds justify.
- **Acceptance:** a `violation_events.json` with a plausible dark/matched split; manually confirm one dark
  point (detection with no nearby AIS) and one matched (detection on an AIS track) against the chip + the
  positions file.
- **Effort:** M.

---

## Workstream 2 — Operational orchestration (turn scripts into a product)

> **Dependency on the 🏠 do-from-home steps (Workstream 1):** the *code* for both items below can be
> **built and unit-tested here** in the sandbox with synthetic fixtures — same as every P3 stage so far.
> - **2B (ranked output)** is fully independent: it reads `violation_events.json` and ranks it, so it
>   builds and tests offline end to end; it only needs *real* events to display eventually.
> - **2A (batch driver)** can be *written* offline, but a real end-to-end *run* invokes the live AIS/GFW
>   ingesters, so its live validation waits on Workstream 1. Neither item depends on the Phase 2 SAR
>   training (that is an alternative detection modality). Net: build both here; defer only 2A's live run.

### 2A. Per-MPA batch driver — `scripts/scan_violations.py`  ✅ DONE (2026-07-23, code+tests; live run still 🏠)
- **Do:** a thin orchestrator that, given a `--wdpa-id` (or a list), runs
  sat_fetch → detect_boats → fuse_violations against a concurrently-captured AIS file, then optionally
  gfw_vessels + re-fuse. Reuses each script's `main(argv)` (all are now import-safe / argv-guarded).
- **Design notes:** honour the timing caveat from 1C — the driver either (a) captures AIS live and fuses a
  fresh overpass, or (b) takes a `--ais` file (historical). Keep it a *driver*: no new fusion logic, just
  sequencing + a per-MPA summary. Complements the planned detection-side `scan_mpas.py`
  (`docs/PIPELINE.md`); consider one `scan_mpas.py` with a `--fuse` flag rather than two drivers.
- **Acceptance:** one command produces `violation_events.json` for a WDPAID end to end.
- **Effort:** M.

### 2B. Operator-facing ranked output  ✅ DONE (2026-07-23) — `scripts/report_violations.py`
- **Do:** `scripts/report_violations.py` that aggregates `violation_events.json` across runs into a ranked
  table (by `violation_score`) — CSV + a self-contained HTML (mirror `review_server.py`'s zero-dependency
  style), one row per event with mpa_name, status, score, vessel identity, evidence, and a link to the chip.
- **Alternative/complement:** overlay dark points in `review_server.py` so a human can adjudicate them on
  the imagery (the review loop already renders chips + detections).
- **Acceptance:** an operator sees the top-N violations ranked, with evidence, without reading JSON.
- **Effort:** M.

---

## Workstream 3 — Signal quality (precision)

### 3A. MPA-specific `fishing_probability`  ✅ DONE (2026-07-23, code+tests; live GFW call still 🏠)
- **Problem:** `summarize_fishing` currently sets `fishing_probability = 1.0` if the vessel fished
  *anywhere* in the `--start`/`--end` window — a vessel-level prior, not "fished in *this* MPA".
- **Do:** pass the MPA region to the Events API so fishing events are spatially filtered to the MPA
  (add an optional `--region`/`--wdpa-id` to `gfw_vessels.py`; GFW events support a region/geometry
  filter). Then `fishing_probability` means "fished inside the MPA in the window". The Insights
  `fishing_in_no_take_mpa` field already carries GFW's own MPA-specific count as a cross-check.
- **Acceptance:** a transiting-but-not-fishing vessel scores at the matched floor; a vessel fishing *in the
  MPA* scores up. Add a test with region-filtered synthetic events.
- **Effort:** S–M.

### 3B. Skylight dark-call cross-check — `scripts/skylight_fetch.py`  ☐
- **Do:** a sibling ingester that pulls Skylight (Ai2) free dark-vessel alerts for an AOI/time
  (https://skylight.global/platform) and normalizes them; then a small comparison that scores our dark
  calls against Skylight's for the same AOI/time (agreement / precision-recall).
- **Why:** free, ready-made ground truth for the dark signal — the fastest external validation of the
  product. Not a pipeline dependency; a validation harness.
- **Unknowns:** Skylight API shape + access (registration). Research first; keep the normalizer tolerant
  like the GFW/AIS ones.
- **Effort:** M (gated on API access).

### 3C. Historical AIS source — `scripts/marinecadastre_fetch.py`  ✅ DONE (2026-07-23, normalizer+tests; download still 🏠)
- **Problem:** AISStream is live-only, so historical scenes can't be fused from it.
- **Do:** a sibling ingester normalizing a **free historical** archive into the AIS contract. Best free
  option: **NOAA MarineCadastre** (US coastal, historical, CSV) → `scripts/marinecadastre_fetch.py`,
  reusing the CSV path `fuse_violations.py --ais` already supports. (Spire is global but paid; GFW has no
  easy raw-position endpoint.) This makes historical US-water MPAs fully fuseable offline.
- **Acceptance:** a historical scene over a US MPA fuses against archive AIS with no live capture.
- **Effort:** M.

---

## Workstream 4 — Robustness & closeout

### 4A. Committed end-to-end fixture test  ✅ DONE (2026-07-23)
- **Do:** a tiny `scripts/tests/fixtures/` (`detections.json` + `ais.json` + `vessels.json`) and a test that
  runs the whole fuse+enrich chain over them — guards the three modules' shared contracts as a unit (today
  each is tested in isolation + round trips in temp dirs).
- **Effort:** S.

### 4B. Pin the real `violation_events` JSON schema in `docs/PIPELINE.md`  ✅ DONE (2026-07-23)
- **Do:** document the exact keys `fuse_violations.py` emits today (incl. `ais_status`, `ais_match`,
  `gfw_vessel_id`, `iuu_listed`, `evidence`, the null-until-GFW fields). The CLAUDE.md prose + the SQL
  sketch predate the implementation; pin the actual shape so downstream (2B, dashboards) can rely on it.
- **Effort:** S.

### 4C. Consolidate env/deps notes  ✅ DONE (2026-07-23) — `CLAUDE.md` 'Phase 3 first-run setup' table
- **Do:** one place listing the P3 env vars (`AISSTREAM_API_KEY`, `GFW_API_TOKEN`) and the
  `requirements-ais.txt` install step, so first-run setup is a single reference.
- **Effort:** S.

---

## Definition of done for Phase 3

- [x] AISStream + GFW calls verified live once; any v3 shape drift fixed in the (isolated) parsers (1A, 1B — **both ✅ 2026-07-26**: three GFW v3 corrections landed with real-response fixtures; AISStream worked first-connect, 41/41 kept over Singapore).
- [x] One real MPA yields a spot-checked `violation_events.json` — a confirmed dark and a confirmed matched (1C — **✅ 2026-07-26, historical path**: Anacapa wake-vessel dark + BB PELAGIC matched at 2 s / 362 m; live-capture variant waits on 1A).
- [x] `fishing_probability` is MPA-specific (3A — `--region-bbox`/`--wdpa-id`; live GFW call still 🏠).
- [x] A batch driver runs the loop for a `--wdpa-id` (2A — `scan_violations.py`; live run still 🏠).
- [x] An operator-facing ranked output exists (2B — `report_violations.py`).
- [x] Historical AIS (3C — `marinecadastre_fetch.py`).
- [x] **An alerting interface** (`alert_violations.py`, 2026-07-29) — the last capability the design
      proposal promised. Severity-bucketed, MPA-grouped, `--exit-code` cron gate.
- [x] **The fishing signal works with no credentials** (`ais_fishing.py`, 2026-07-29).
- [x] **Cooperative in-MPA vessels are flagged too** (`geofence_ais.py`, 2026-07-29) — the third status.

**Optional / not required for done:** Skylight cross-check (3B → ROADMAP **W9**, gated on API access).

## What's left (moved to `docs/ROADMAP.md` → THE WORK QUEUE)

Phase 3's build is finished, so the remaining items are tracked with the rest of the project's work
rather than here:

- **W3 — the near-real-time operational mode.** Everything has run *historically* (MarineCadastre
  archive AIS fused against a past overpass). What has never run is a rolling live AIS buffer fused
  against a fresh overpass. Path A (AIS-only, needs just `AISSTREAM_API_KEY`) is the cheap version.
- **W8 — websocket reconnect for `aisstream_fetch.py`.** Deliberately left open in the AUDIT #4 retry
  round: reconnect-and-resume changes *capture semantics* (duplicate positions, partial windows), which
  is a design decision, not a transport setting. **W3 depends on deciding this** — a rolling buffer is
  exactly where a dropped connection costs you the acquisition.
- **W9 — Skylight cross-check** (3B below), gated on external API access.

Everything above needed only free accounts (AISStream key, GFW token) already in scope; no paid
services and no new heavy dependencies were introduced.
