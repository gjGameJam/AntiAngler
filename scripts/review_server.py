import argparse
import io
import json
import os
import secrets
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Literal, Optional
import uvicorn

# The objective of this script (pipeline stage 3 - "report + review"):
# - Serve a sat_fetch run dir's chips as a local, single-user review UI
# - Overlay the stage-2 detections (detections.json) on each chip
# - Let a human mark each detection correct/incorrect and draw MISSED boats
# - Persist verdicts to reviews.json; scripts/export_labels.py turns reviewed chips
#   into a YOLO training set to fine-tune the detector (active-learning loop)
#
# Security posture (localhost single-user tool, see docs/PIPELINE.md):
# - Binds 127.0.0.1 only (never 0.0.0.0)
# - Host-header allowlist middleware (defends against DNS-rebinding)
# - Random startup token required on every route (in the printed URL + same-origin JS)
# - Chips served by opaque manifest index, never by a client-supplied path
#   (defense-in-depth: resolve + is_relative_to the run dir)
# - Strict per-request-nonce CSP; fully self-contained page, zero external assets
# - Atomic writes for reviews.json (temp-in-same-dir + Path.replace)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")

STATE = {}  # populated in main() before the server starts

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


# --------------------------------------------------------------------------- security

@app.middleware("http")
async def host_allowlist(request: Request, call_next):
    if request.headers.get("host", "") not in STATE["allowed_hosts"]:
        return JSONResponse({"detail": "host not allowed"}, status_code=403)
    return await call_next(request)


def require_token(request: Request):
    tok = request.headers.get("x-auth-token") or request.query_params.get("token")
    if not tok or not secrets.compare_digest(tok, STATE["token"]):
        raise HTTPException(status_code=403, detail="missing or bad token")


# --------------------------------------------------------------------------- data model

class Box(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bbox_pixel: List[float] = Field(min_length=4, max_length=4)
    source: Literal["detection", "human"]
    detection_id: Optional[str] = None
    verdict: Literal["correct", "incorrect", "missed"]


class ReviewIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewed: bool
    boxes: List[Box]


def load_reviews(run_dir):
    """Load a single run's reviews.json (each run keeps its own, so export_labels --run is unchanged)."""
    path = run_dir / "reviews.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"reviewed_by": STATE["reviewer"], "chips": {}}


def save_reviews(run_dir, reviews):
    path = run_dir / "reviews.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reviews, indent=2))
    tmp.replace(path)


def geotiff_to_png(path):
    with rasterio.open(path) as src:
        arr = np.transpose(src.read([1, 2, 3]), (1, 2, 0))  # (H,W,3) uint8; chips are pre-stretched
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- routes

@app.get("/api/state")
def api_state(_=Depends(require_token)):
    # One reviews.json per run; load each once and index chips into its own run's verdicts.
    reviews_by_run = {name: load_reviews(rd) for name, rd in STATE["run_dirs"].items()}
    chips = []
    for chip in STATE["chips"]:
        saved = reviews_by_run[chip["run"]]["chips"].get(chip["filename"])
        chips.append({
            "id": chip["id"],
            "run": chip["run"],
            "filename": chip["filename"],
            "tile_size": chip["tile_size"],
            "has_companion": chip["has_companion"],
            "detections": chip["detections"],
            "review": saved,
            "reviewed": bool(saved and saved.get("reviewed")),
        })
    return {"runs": list(STATE["run_dirs"].keys()), "chips": chips}


@app.get("/chip/{chip_id}.png")
def chip_png(chip_id: int, _=Depends(require_token)):
    rec = STATE["chip_by_id"].get(chip_id)
    if not rec:
        raise HTTPException(status_code=404, detail="unknown chip")
    run_dir = rec["_run_dir"]
    path = (run_dir / rec["filename"]).resolve()
    if not path.is_relative_to(run_dir) or not path.is_file():
        raise HTTPException(status_code=404, detail="chip not found")
    return Response(geotiff_to_png(path), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/companion/{chip_id}.png")
def companion_png(chip_id: int, _=Depends(require_token)):
    """Co-registered optical image for a SAR chip (land reference); 404 if none was generated."""
    rec = STATE["chip_by_id"].get(chip_id)
    if not rec:
        raise HTTPException(status_code=404, detail="unknown chip")
    run_dir = rec["_run_dir"]
    path = (run_dir / "companion" / f"{Path(rec['filename']).stem}.png").resolve()
    if not path.is_relative_to(run_dir) or not path.is_file():
        raise HTTPException(status_code=404, detail="no companion")
    return Response(path.read_bytes(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.post("/api/review/{chip_id}")
def post_review(chip_id: int, body: ReviewIn, _=Depends(require_token)):
    rec = STATE["chip_by_id"].get(chip_id)
    if not rec:
        raise HTTPException(status_code=404, detail="unknown chip")
    run_dir = rec["_run_dir"]
    reviews = load_reviews(run_dir)
    reviews["chips"][rec["filename"]] = {"reviewed": body.reviewed,
                                         "boxes": [b.model_dump() for b in body.boxes]}
    save_reviews(run_dir, reviews)
    return {"ok": True,
            "reviewed_count": sum(1 for c in reviews["chips"].values() if c.get("reviewed"))}


@app.get("/", response_class=HTMLResponse)
def index(request: Request, _=Depends(require_token)):
    nonce = secrets.token_urlsafe(16)
    html = HTML_TEMPLATE.replace("__NONCE__", nonce).replace("__TOKEN__", STATE["token"])
    csp = (f"default-src 'none'; img-src 'self'; style-src 'nonce-{nonce}'; "
           f"script-src 'nonce-{nonce}'; connect-src 'self'; base-uri 'none'; "
           f"frame-ancestors 'none'; form-action 'none'")
    return HTMLResponse(html, headers={"Content-Security-Policy": csp,
                                       "X-Content-Type-Options": "nosniff"})


# --------------------------------------------------------------------------- frontend

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AntiAngler - Detection Review</title>
<style nonce="__NONCE__">
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.4 system-ui, sans-serif; background: #14171c; color: #e6e6e6; }
  header { display: flex; gap: 16px; align-items: center; padding: 8px 14px;
           background: #1c2027; border-bottom: 1px solid #2c313b; position: sticky; top: 0; }
  header b { color: #fff; }
  .grow { flex: 1; }
  .bar { height: 6px; background: #2c313b; border-radius: 3px; width: 220px; overflow: hidden; }
  .bar > i { display: block; height: 100%; background: #3fb950; width: 0%; }
  main { display: flex; gap: 14px; padding: 14px; align-items: flex-start; }
  #wrap { display: flex; gap: 10px; }
  .pane { margin: 0; }
  .pane figcaption { font-size: 12px; color: #93a1b0; line-height: 1.3; margin-bottom: 5px; max-width: 640px; min-height: 1.3em; }
  canvas { max-width: min(88vw, 88vh); border: 1px solid #2c313b; cursor: crosshair; touch-action: none; display: block; }
  #wrap.dual canvas { max-width: min(43vw, 80vh); }
  #cvc { cursor: default; }
  aside { width: 300px; }
  .card { background: #1c2027; border: 1px solid #2c313b; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
  button { font: inherit; padding: 7px 11px; border-radius: 6px; border: 1px solid #3a414d;
           background: #262c36; color: #e6e6e6; cursor: pointer; }
  button:hover { background: #2f3743; }
  button.primary { background: #2ea043; border-color: #2ea043; color: #fff; }
  button:disabled { opacity: .45; cursor: not-allowed; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
  .legend span { display: inline-flex; align-items: center; gap: 5px; margin-right: 12px; }
  .sw { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
  .muted { color: #93a1b0; }
  kbd { background: #2c313b; border-radius: 4px; padding: 1px 6px; font-family: ui-monospace, monospace; }
  ul { padding-left: 18px; margin: 6px 0; }
</style>
</head>
<body>
<header>
  <b>AntiAngler Review</b>
  <span id="runName" class="muted"></span>
  <div class="grow"></div>
  <span id="pos" class="muted"></span>
  <div class="bar"><i id="prog"></i></div>
  <span id="cnt" class="muted"></span>
</header>
<main>
  <div id="wrap">
    <figure class="pane"><figcaption id="lblMain">chip</figcaption><canvas id="cv" width="640" height="640"></canvas></figure>
    <figure class="pane" id="compPane" hidden><figcaption>optical &mdash; land reference (drawn from the paired S2 scene; ships differ by date, land does not)</figcaption><canvas id="cvc" width="640" height="640"></canvas></figure>
  </div>
  <aside>
    <div class="card">
      <div class="row">
        <button id="prev">&larr; Prev</button>
        <button id="next">Next &rarr;</button>
      </div>
      <div class="row">
        <button id="saveDraft">Save draft <kbd>d</kbd></button>
        <button id="saveReviewed" class="primary">Mark reviewed <kbd>s</kbd></button>
      </div>
      <p id="status" class="muted"></p>
    </div>
    <div class="card legend">
      <b>Legend</b><br>
      <div style="margin-top:8px">
        <span><i class="sw" style="background:#e3b341"></i> pending</span>
        <span><i class="sw" style="background:#3fb950"></i> correct</span><br>
        <span><i class="sw" style="background:#f85149"></i> incorrect</span>
        <span><i class="sw" style="background:#58a6ff"></i> missed (drawn)</span>
      </div>
    </div>
    <div class="card">
      <b>How to review</b>
      <ul class="muted">
        <li><b>Click</b> a detection box to cycle pending &rarr; correct &rarr; incorrect.</li>
        <li><b>Drag</b> on empty water to add a <i>missed</i> boat.</li>
        <li><b>Right-click</b> a drawn box to delete it.</li>
        <li>Mark reviewed only once <b>every</b> real boat is boxed and every box judged
            (no pending) - this "complete labels" rule is what makes the export trainable.</li>
      </ul>
    </div>
  </aside>
</main>
<script nonce="__NONCE__">
const TOKEN = "__TOKEN__";
const H = { "X-Auth-Token": TOKEN };
let chips = [], idx = 0, tileSize = 640, boxes = [], img = new Image(), imgC = new Image();
let drag = null;  // {x0,y0,x1,y1} while dragging a new box
let hasComp = false;

const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
const cvc = document.getElementById("cvc"), ctxc = cvc.getContext("2d");
const COLOR = { pending:"#e3b341", correct:"#3fb950", incorrect:"#f85149", missed:"#58a6ff" };

async function api(path, opts) {
  const r = await fetch(path, Object.assign({ headers: H }, opts || {}));
  if (!r.ok) throw new Error(path + " -> " + r.status);
  return r.json();
}

async function boot() {
  const s = await api("/api/state");
  chips = s.chips;
  document.getElementById("runName").textContent =
      s.runs.length === 1 ? s.runs[0] : (s.runs.length + " runs • " + chips.length + " chips");
  loadChip(0);
  refreshProgress();
}

function loadChip(i) {
  idx = Math.max(0, Math.min(chips.length - 1, i));
  const c = chips[idx];
  // Chip size + companion pane are per-chip now (a batch can mix runs / optical + SAR).
  tileSize = c.tile_size || 640;
  cv.width = tileSize; cv.height = tileSize;
  cvc.width = tileSize; cvc.height = tileSize;
  hasComp = !!c.has_companion;
  document.getElementById("compPane").hidden = !hasComp;
  document.getElementById("wrap").classList.toggle("dual", hasComp);
  document.getElementById("lblMain").textContent = hasComp ? "SAR (VV / VH) — label on this side" : "chip";
  // Working set: saved review boxes if present, else detections seeded as pending.
  if (c.review && c.review.boxes) {
    boxes = c.review.boxes.map(b => ({...b}));
  } else {
    boxes = (c.detections || []).map(d => ({
      bbox_pixel: d.bbox_pixel, source: "detection", detection_id: d.detection_id, verdict: "pending"
    }));
  }
  img = new Image();
  img.onload = draw;
  img.src = "/chip/" + c.id + ".png?token=" + encodeURIComponent(TOKEN);
  if (hasComp) {
    imgC = new Image();
    imgC.onload = draw; imgC.onerror = draw;  // missing companion -> boxes on a blank pane
    imgC.src = "/companion/" + c.id + ".png?token=" + encodeURIComponent(TOKEN);
  }
  document.getElementById("pos").textContent = "chip " + (idx + 1) + " / " + chips.length + "  •  " + c.run;
  document.getElementById("status").textContent = c.reviewed ? "already reviewed" : "";
  draw();
}

function drawPane(c2, image) {
  c2.clearRect(0, 0, tileSize, tileSize);
  if (image && image.complete && image.naturalWidth) c2.drawImage(image, 0, 0, tileSize, tileSize);
  c2.lineWidth = 2; c2.font = "12px system-ui";
  for (const b of boxes) {
    const [x1, y1, x2, y2] = b.bbox_pixel;
    c2.strokeStyle = COLOR[b.verdict] || "#e3b341";
    c2.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
}

function draw() {
  drawPane(ctx, img);
  if (drag) {
    ctx.strokeStyle = COLOR.missed; ctx.setLineDash([5, 4]);
    ctx.strokeRect(drag.x0, drag.y0, drag.x1 - drag.x0, drag.y1 - drag.y0);
    ctx.setLineDash([]);
  }
  if (hasComp) drawPane(ctxc, imgC);
}

function toCanvas(e) {
  const r = cv.getBoundingClientRect();
  return [(e.clientX - r.left) * (cv.width / r.width), (e.clientY - r.top) * (cv.height / r.height)];
}
function hit(x, y) {  // smallest box containing the point, so a small box nested in a bigger one stays clickable
  let best = -1, bestArea = Infinity;
  for (let i = 0; i < boxes.length; i++) {
    const [x1, y1, x2, y2] = boxes[i].bbox_pixel;
    if (x >= Math.min(x1,x2) && x <= Math.max(x1,x2) && y >= Math.min(y1,y2) && y <= Math.max(y1,y2)) {
      const area = Math.abs(x2 - x1) * Math.abs(y2 - y1);
      if (area < bestArea) { bestArea = area; best = i; }
    }
  }
  return best;
}

cv.addEventListener("mousedown", e => {
  if (e.button !== 0) return;
  const [x, y] = toCanvas(e);
  drag = { x0: x, y0: y, x1: x, y1: y, moved: false };
});
cv.addEventListener("mousemove", e => {
  if (!drag) return;
  const [x, y] = toCanvas(e);
  drag.x1 = x; drag.y1 = y; drag.moved = true; draw();
});
window.addEventListener("mouseup", e => {
  if (!drag) return;
  const d = drag; drag = null;
  const w = Math.abs(d.x1 - d.x0), h = Math.abs(d.y1 - d.y0);
  if (d.moved && w > 4 && h > 4) {
    boxes.push({ bbox_pixel: [Math.min(d.x0,d.x1), Math.min(d.y0,d.y1), Math.max(d.x0,d.x1), Math.max(d.y0,d.y1)],
                 source: "human", verdict: "missed" });
  } else {  // a click: cycle the verdict of the box under the cursor
    const i = hit(d.x0, d.y0);
    if (i >= 0 && boxes[i].source === "detection") {
      boxes[i].verdict = boxes[i].verdict === "pending" ? "correct"
                       : boxes[i].verdict === "correct" ? "incorrect" : "correct";
    }
  }
  draw();
});
cv.addEventListener("contextmenu", e => {
  e.preventDefault();
  const [x, y] = toCanvas(e);
  const i = hit(x, y);
  if (i >= 0 && boxes[i].source === "human") { boxes.splice(i, 1); draw(); }
});

function pendingCount() { return boxes.filter(b => b.source === "detection" && b.verdict === "pending").length; }

async function save(reviewed) {
  if (reviewed && pendingCount() > 0) {
    document.getElementById("status").textContent = pendingCount() + " detection(s) still pending - judge them first.";
    return;
  }
  const c = chips[idx];
  await api("/api/review/" + c.id, {
    method: "POST", headers: Object.assign({ "Content-Type": "application/json" }, H),
    body: JSON.stringify({ reviewed, boxes })
  });
  c.review = { reviewed, boxes: boxes.map(b => ({...b})) };
  c.reviewed = reviewed;
  const doneGlobal = chips.filter(x => x.reviewed).length;  // count across all runs in the batch
  document.getElementById("status").textContent = (reviewed ? "reviewed ✓" : "draft saved") +
      "  (" + doneGlobal + "/" + chips.length + " reviewed)";
  refreshProgress();
  if (reviewed && idx < chips.length - 1) loadChip(idx + 1);
}

function refreshProgress() {
  const n = chips.filter(c => c.reviewed).length;
  document.getElementById("prog").style.width = (100 * n / chips.length) + "%";
  document.getElementById("cnt").textContent = n + " / " + chips.length + " reviewed";
}

document.getElementById("prev").onclick = () => loadChip(idx - 1);
document.getElementById("next").onclick = () => loadChip(idx + 1);
document.getElementById("saveDraft").onclick = () => save(false);
document.getElementById("saveReviewed").onclick = () => save(true);
window.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") loadChip(idx - 1);
  else if (e.key === "ArrowRight") loadChip(idx + 1);
  else if (e.key === "s") save(true);
  else if (e.key === "d") save(false);
});

boot().catch(err => { document.body.innerHTML = "<p style='padding:20px'>Failed to load: " + err.message + "</p>"; });
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- startup

def env_or(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def _default_runs():
    """REVIEW_RUN env fallback: a single run dir path, or None. CLI --run (one or more) overrides."""
    val = os.getenv("REVIEW_RUN")
    return [Path(val)] if val not in (None, "") else None


def parse_args():
    p = argparse.ArgumentParser(description="Local review UI for stage-2 boat detections.")
    p.add_argument("--run", type=Path, nargs="+", default=_default_runs(),
                   help="One or more sat_fetch run dirs (each with chips/, manifest.json, detections.json). "
                        "Chips from all are reviewed in one batched session; each verdict is written back to "
                        "its own run's reviews.json, so export_labels.py --run <dir> is unchanged.")
    p.add_argument("--host", default=env_or("REVIEW_HOST", "127.0.0.1"),
                   help="Bind address. Keep 127.0.0.1 (loopback) unless you know why not.")
    p.add_argument("--port", type=int, default=int(env_or("REVIEW_PORT", 8000)))
    p.add_argument("--reviewer", default=env_or("REVIEW_REVIEWER", os.getenv("USERNAME", "anon")),
                   help="Name recorded in reviews.json.")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.run:
        raise RuntimeError("Provide one or more run dirs: --run data/raw/sentinel2/<run>/ [<run2>/ ...]")

    # Reviewer name is consulted by load_reviews() when a run has no reviews.json yet.
    STATE["reviewer"] = args.reviewer

    chips = []
    run_dir_map = {}   # name -> resolved Path (insertion order = review order)
    for raw in args.run:
        run_dir = raw.resolve()
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"No manifest.json in {run_dir} - run sat_fetch.py first.")
        manifest = json.loads(manifest_path.read_text())
        name = run_dir.name
        if name in run_dir_map:
            raise RuntimeError(f"Duplicate run dir name {name!r} - pass each run once.")
        run_dir_map[name] = run_dir

        det_by_chip = {}
        det_path = run_dir / "detections.json"
        if det_path.exists():
            for d in json.loads(det_path.read_text()).get("detections", []):
                det_by_chip.setdefault(d["chip"], []).append(
                    {"detection_id": d["detection_id"], "bbox_pixel": d["bbox_pixel"],
                     "confidence": d.get("confidence")})
        else:
            print(f"NOTE: no detections.json in {name} - its chips start with no boxes "
                  "(you can still label from scratch).")

        has_comp = (run_dir / "companion").is_dir()
        tile = int(manifest.get("tile_size", 640))
        for c in manifest["chips"]:
            chips.append({
                "id": len(chips), "run": name, "_run_dir": run_dir,
                "filename": c["filename"], "tile_size": tile, "has_companion": has_comp,
                "detections": det_by_chip.get(c["filename"], []),
            })

    STATE.update(
        chips=chips,
        chip_by_id={c["id"]: c for c in chips},
        run_dirs=run_dir_map,
        token=secrets.token_urlsafe(24),
        allowed_hosts={f"{args.host}:{args.port}", f"127.0.0.1:{args.port}", f"localhost:{args.port}"},
    )

    n_with_det = sum(1 for c in chips if c["detections"])
    url = f"http://127.0.0.1:{args.port}/?token={STATE['token']}"
    print("=" * 72)
    print(f"  AntiAngler review server  -  {len(run_dir_map)} run(s)  -  {len(chips)} chips  "
          f"-  {n_with_det} with detections")
    for name in run_dir_map:
        print(f"    - {name}  ({sum(1 for c in chips if c['run'] == name)} chips)")
    print(f"\n  Open this URL (contains the single-use auth token):\n\n    {url}\n")
    print("  Bound to loopback only. Ctrl+C to stop.")
    print("=" * 72, flush=True)  # flush the whole banner now so the URL shows before uvicorn blocks
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
