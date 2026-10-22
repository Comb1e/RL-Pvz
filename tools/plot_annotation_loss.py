"""View annotation-fit history without copying its data into the HTML.

Use --serve for direct JSONL loading, or choose the source file in a local HTML
tab. Curves retain each screen column's first/minimum/maximum/last samples,
inspired by M4 (Jugel et al., VLDB 2014):
https://www.vldb.org/pvldb/vol7/p797-jugel.pdf
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlsplit

SERIES = (
    ("train loss", "train", "loss", "#65e5b2"),
    ("validation loss", "validation", "loss", "#ffbd66"),
    ("train top loss", "train", "top_loss", "#78baff"),
    ("validation top loss", "validation", "top_loss", "#ff7f9c"),
    ("train bottom loss", "train", "bottom_loss", "#40d9df"),
    ("validation bottom loss", "validation", "bottom_loss", "#f5ed8b"),
    ("L2 penalty", "", "regularization_loss", "#cd9dff"),
)

HTML = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Annotation fit loss</title>
<style>
*{box-sizing:border-box}body{margin:1rem;background:#101418;color:#e8eef2;font:14px system-ui,sans-serif}
h1{font-size:1.5rem}p{max-width:90em;line-height:1.5}a{color:#8acbff}
canvas{display:block;width:100%;height:clamp(300px,68vh,800px);background:#171d23;
border:1px solid #3a4650;cursor:crosshair;touch-action:none}
#legend,.toolbar{display:flex;gap:.8rem;align-items:center;flex-wrap:wrap;margin:.7rem 0}
label{cursor:pointer}.swatch{display:inline-block;width:.8rem;height:.8rem;margin-right:.3rem}
button,select{background:#27343f;color:inherit;border:1px solid #536574;border-radius:4px;
padding:.35rem .65rem;cursor:pointer}button:disabled{opacity:.5;cursor:wait}
#status,#tip{min-height:1.5rem;overflow-wrap:anywhere}
#status{color:#adbdc9}#status.error{color:#ff9e9e}
#tip{font:13px/1.6 ui-monospace,monospace;white-space:pre-wrap}
</style>
<h1>Annotation fit loss</h1>
<p>Connected curves preserve each screen column's first, lowest, highest, and last values.
Zoom to recover individual passes; no smoothing is applied. Data is read from the original
<a id="source">history file</a>, never embedded or copied into this page.</p>
<div class="toolbar">
<button id="reload">Reload history</button><label>Choose history <input id="file" type="file" accept=".jsonl,.json,application/json"></label>
<button id="reset">Show all</button><button id="recent">Last 10%</button>
<label>Y scale <select id="scale"><option value="linear">Linear</option>
<option value="log">Logarithmic (zero-safe)</option></select></label>
</div>
<div id="legend"></div><div id="status" role="status">Loading history…</div>
<canvas id="chart" aria-label="Fit loss: scroll to zoom, drag to pan, double-click to reset"></canvas>
<div id="tip">Scroll to zoom at the pointer. Drag to pan. Double-click to reset.</div>
<script>
'use strict';
const series = __SERIES__;
const sourceURL = new URLSearchParams(location.search).get('history') || __HISTORY_URL__;
const canvas = document.querySelector('#chart'), ctx = canvas.getContext('2d');
const status = document.querySelector('#status'), tip = document.querySelector('#tip');
const source = document.querySelector('#source'), checks = new Map();
const BASE = 16, PAD = {left:92, right:20, top:22, bottom:44};
const state = {data:null, left:0, right:1, drag:null, job:0, abort:null, frame:0};
source.href = sourceURL;
source.textContent = __HISTORY_NAME__;
source.title = sourceURL;
for (let k = 0; k < series.length; k++) {
  const s = series[k], label = document.createElement('label');
  const box = document.createElement('input'), swatch = document.createElement('span');
  box.type = 'checkbox'; box.checked = k < 2; box.onchange = schedule;
  swatch.className = 'swatch'; swatch.style.background = s.color;
  label.append(box, swatch, s.name); document.querySelector('#legend').append(label);
  checks.set(s.key, box);
}
function message(text, error=false) {
  status.textContent = text; status.classList.toggle('error', error);
}
const pause = () => new Promise(resolve => setTimeout(resolve, 0));
function cancelled(job) {
  if (job !== state.job) throw new DOMException('Loading cancelled', 'AbortError');
}

// Keep only numeric columns in memory, rather than the much larger JSON objects.
async function readHistory(stream, total, job) {
  const reader = stream.getReader(), decoder = new TextDecoder();
  let columns = Array.from({length:series.length + 1}, () => new Float64Array(4096));
  let count = 0, lineNumber = 0, bytes = 0, pending = '', lastUpdate = 0;
  function append(line) {
    lineNumber++;
    if (!line.trim()) return;
    let row;
    try { row = JSON.parse(line); }
    catch { throw new Error('Invalid JSON at history line ' + lineNumber); }
    if (!row || !Number.isSafeInteger(row.pass_number))
      throw new Error('Invalid pass number at history line ' + lineNumber);
    if (count && row.pass_number < columns[0][count-1])
      throw new Error('Pass numbers are out of order at history line ' + lineNumber);
    if (count === columns[0].length) {
      columns = columns.map(old => { const next = new Float64Array(old.length * 2); next.set(old); return next; });
    }
    columns[0][count] = row.pass_number;
    series.forEach((s, k) => {
      const v = s.split ? row[s.split]?.[s.field] : row[s.field];
      columns[k+1][count] = typeof v === 'number' && Number.isFinite(v) ? v : NaN;
    });
    count++;
  }
  try {
    while (true) {
      cancelled(job);
      const {value, done} = await reader.read();
      pending += done ? decoder.decode() : decoder.decode(value, {stream:true});
      bytes += value?.byteLength || 0;
      let start = 0, newline;
      while ((newline = pending.indexOf('\n', start)) >= 0) {
        append(pending.slice(start, newline)); start = newline + 1;
      }
      pending = pending.slice(start);
      if (done) break;
      if (performance.now() - lastUpdate > 80) {
        message('Loading ' + count.toLocaleString() + ' passes' + (total ?
          ' · ' + Math.min(100, bytes / total * 100).toFixed(0) + '%' : '') + '…');
        lastUpdate = performance.now(); await pause();
      }
    }
    if (pending.trim()) append(pending);
  } finally {
    await reader.cancel().catch(() => {}); reader.releaseLock();
  }
  if (!count) throw new Error('Annotation history contains no committed passes');
  const data = {passes:columns[0].subarray(0, count), values:[], levels:[], count};
  for (let k = 0; k < series.length; k++) {
    cancelled(job);
    const values = columns[k+1].subarray(0, count);
    message('Indexing curves ' + (k+1) + '/' + series.length + '…');
    data.values.push(values); data.levels.push(buildLevels(values)); await pause();
  }
  return data;
}

// Small multiresolution summaries make redraws depend on screen width, not file size.
// Each node holds raw indices [first, minimum, maximum, last], never averaged values.
function extend(summary, index, values) {
  if (index < 0 || !Number.isFinite(values[index])) return;
  if (summary[0] < 0) { summary.fill(index); return; }
  summary[0] = Math.min(summary[0], index); summary[3] = Math.max(summary[3], index);
  if (values[index] < values[summary[1]]) summary[1] = index;
  if (values[index] > values[summary[2]]) summary[2] = index;
}
function buildLevels(values) {
  const levels = [];
  let length = values.length, previous = null;
  while (length > 1) {
    const nodes = Math.ceil(length / BASE), current = new Int32Array(nodes * 4);
    current.fill(-1);
    for (let n = 0; n < nodes; n++) {
      const summary = current.subarray(n*4, n*4+4), end = Math.min(length, (n+1)*BASE);
      for (let i = n*BASE; i < end; i++) {
        if (previous) for (let j = 0; j < 4; j++) extend(summary, previous[i*4+j], values);
        else extend(summary, i, values);
      }
    }
    levels.push(current); previous = current; length = nodes;
  }
  return levels;
}
function summarize(values, levels, start, end) {
  const summary = [-1,-1,-1,-1];
  let level = -1;
  function add(i) {
    if (level < 0) extend(summary, i, values);
    else for (let j = 0; j < 4; j++) extend(summary, levels[level][i*4+j], values);
  }
  while (start < end) {
    if (end - start < BASE*2) { while (start < end) add(start++); break; }
    while (start % BASE && start < end) add(start++);
    while (end % BASE && start < end) add(--end);
    start /= BASE; end /= BASE; level++;
  }
  return summary;
}
function lowerBound(passes, x) {
  let lo = 0, hi = passes.length;
  while (lo < hi) { const mid = lo + Math.floor((hi-lo)/2); if (passes[mid] < x) lo = mid+1; else hi = mid; }
  return lo;
}
function fullRange() {
  const p = state.data.passes, first = p[0], last = p[p.length-1];
  return first === last ? [first-.5, last+.5] : [first, last];
}
function view(left, right) {
  const [first, last] = fullRange(), span = Math.min(last-first, Math.max(Math.min(1, last-first), right-left));
  state.left = Math.max(first, Math.min(last-span, left)); state.right = state.left + span;
  schedule();
}
function reset() { if (state.data) view(...fullRange()); }
function geometry() {
  const r = canvas.getBoundingClientRect();
  return {width:r.width, height:r.height, w:Math.max(1,r.width-PAD.left-PAD.right),
    h:Math.max(1,r.height-PAD.top-PAD.bottom)};
}
function schedule() { if (!state.frame) state.frame = requestAnimationFrame(() => { state.frame = 0; draw(); }); }
function size() {
  const r = canvas.getBoundingClientRect(), d = devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(r.width*d)); canvas.height = Math.max(1, Math.round(r.height*d));
  ctx.setTransform(d,0,0,d,0,0); schedule();
}
function format(value) {
  if (value === 0) return '0';
  return Math.abs(value) < .0001 || Math.abs(value) >= 100000 ? value.toExponential(3) : Number(value.toPrecision(6)).toString();
}
function draw() {
  const g = geometry(); ctx.clearRect(0,0,g.width,g.height);
  if (!state.data) return;
  const data = state.data, p = data.passes, selected = series.map((s,k) => checks.get(s.key).checked ? k : -1).filter(k => k >= 0);
  // Include neighboring samples so continuous curves reach the viewport edges.
  const start = Math.max(0, lowerBound(p,state.left)-1), end = Math.min(p.length,lowerBound(p,state.right)+1);
  const log = document.querySelector('#scale').value === 'log';
  const transform = log ? v => Math.sign(v)*Math.log1p(Math.abs(v)/1e-6) : v => v;
  const inverse = log ? v => Math.sign(v)*Math.expm1(Math.abs(v))*1e-6 : v => v;
  let lo = Infinity, hi = -Infinity;
  for (const k of selected) {
    const s = summarize(data.values[k],data.levels[k],start,end);
    if (s[0] >= 0) { lo = Math.min(lo,transform(data.values[k][s[1]])); hi = Math.max(hi,transform(data.values[k][s[2]])); }
  }
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  const margin = hi === lo ? Math.max(Math.abs(lo)*.03, log ? .1 : 1e-6) : (hi-lo)*.06;
  const y0 = lo >= 0 ? Math.max(0,lo-margin) : lo-margin, y1 = hi+margin;
  const xAt = index => PAD.left + (p[index]-state.left)/(state.right-state.left)*g.w;
  const yAt = value => PAD.top + (y1-transform(value))/(y1-y0)*g.h;
  ctx.font = '12px system-ui'; ctx.lineWidth = 1; ctx.strokeStyle = '#33404a'; ctx.fillStyle = '#b6c5cf';
  for (let i = 0; i <= 5; i++) {
    const y = PAD.top+g.h*i/5;
    ctx.beginPath(); ctx.moveTo(PAD.left,y); ctx.lineTo(PAD.left+g.w,y); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText(format(inverse(y1-(y1-y0)*i/5)),PAD.left-9,y+4);
    const pass = state.left+(state.right-state.left)*i/5, x = PAD.left+g.w*i/5;
    ctx.textAlign = i === 0 ? 'left' : i === 5 ? 'right' : 'center';
    ctx.fillText(Number(pass.toFixed(1)).toLocaleString(),x,PAD.top+g.h+21);
  }
  ctx.textAlign = 'center'; ctx.fillText('Optimizer pass',PAD.left+g.w/2,g.height-5);
  // Use physical pixel columns on high-DPI screens; every bucket retains extrema.
  const pixels = Math.max(1,Math.ceil(g.w*(devicePixelRatio || 1)));
  const edges = new Int32Array(pixels+1);
  for (let b = 0; b <= pixels; b++) edges[b] = lowerBound(p,state.left+(state.right-state.left)*b/pixels);
  edges[0] = start; edges[pixels] = end;
  ctx.save(); ctx.beginPath(); ctx.rect(PAD.left,PAD.top,g.w,g.h); ctx.clip();
  for (const k of selected) {
    const values = data.values[k], samples = [];
    if (end-start <= pixels*2) {
      for (let i = start; i < end; i++) if (Number.isFinite(values[i])) samples.push(i);
    } else {
      for (let b = 0; b < pixels; b++) {
        const indices = summarize(values,data.levels[k],edges[b],edges[b+1]).filter(i => i >= 0).sort((a,b) => a-b);
        for (const i of indices) if (samples[samples.length-1] !== i) samples.push(i);
      }
    }
    ctx.strokeStyle = series[k].color; ctx.fillStyle = series[k].color; ctx.lineWidth = 1.8;
    ctx.setLineDash(series[k].split === 'validation' ? [6,3] : []);
    // A connected path keeps constant and near-constant loss visible, unlike min/max strokes.
    ctx.beginPath();
    samples.forEach((i,j) => { const x = xAt(i), y = yAt(values[i]); if (j) ctx.lineTo(x,y); else ctx.moveTo(x,y); });
    ctx.stroke();
    // Isolated points and short histories need actual dots, not zero-length strokes.
    if (samples.length <= Math.max(1,g.w/8)) for (const i of samples) {
      ctx.beginPath(); ctx.arc(xAt(i),yAt(values[i]),2.5,0,Math.PI*2); ctx.fill();
    }
  }
  ctx.restore();
  message(data.count.toLocaleString() + ' passes · viewing ' + format(state.left) + '–' + format(state.right) +
    ' · ' + (end-start > pixels*2 ? 'pixel extrema retained' : 'individual samples') +
    (selected.length ? '' : ' · select a curve above'));
}
function passAt(event) {
  const r = canvas.getBoundingClientRect(), g = geometry();
  return state.left + Math.max(0,Math.min(1,(event.clientX-r.left-PAD.left)/g.w))*(state.right-state.left);
}
function pointAt(event) {
  if (!state.data) return;
  const data = state.data, pass = passAt(event);
  let i = Math.min(data.count-1,lowerBound(data.passes,pass));
  if (i && pass-data.passes[i-1] < data.passes[i]-pass) i--;
  const values = series.map((s,k) => checks.get(s.key).checked ?
    s.name + ': ' + (Number.isFinite(data.values[k][i]) ? data.values[k][i].toPrecision(10) : 'n/a') : null).filter(Boolean);
  tip.textContent = 'Pass ' + data.passes[i] + '  ·  ' + values.join('  ·  ');
}
canvas.onpointerdown = event => {
  if (!state.data || event.button !== 0) return;
  state.drag = {x:event.clientX, left:state.left, right:state.right}; canvas.setPointerCapture(event.pointerId);
};
canvas.onpointermove = event => {
  if (state.drag) {
    const d = state.drag, delta = (event.clientX-d.x)/geometry().w*(d.right-d.left);
    view(d.left-delta,d.right-delta);
  }
  pointAt(event);
};
canvas.onpointerup = event => { state.drag = null; if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId); };
canvas.onpointercancel = canvas.onlostpointercapture = () => { state.drag = null; };
canvas.addEventListener('wheel', event => {
  if (!state.data) return;
  event.preventDefault();
  const factor = Math.exp(Math.max(-2,Math.min(2,event.deltaY*.001))), mid = passAt(event);
  view(mid-(mid-state.left)*factor,mid+(state.right-mid)*factor);
}, {passive:false});
canvas.ondblclick = reset;
document.querySelector('#reset').onclick = reset;
document.querySelector('#recent').onclick = () => {
  if (state.data) { const [a,b] = fullRange(); view(b-(b-a)*.1,b); }
};
document.querySelector('#scale').onchange = schedule;
async function load(file=null) {
  const job = ++state.job;
  state.abort?.abort(); state.abort = new AbortController(); state.drag = null;
  document.querySelector('#reload').disabled = true;
  try {
    message('Loading original history…');
    let stream, total;
    if (file) { stream = file.stream(); total = file.size; }
    else {
      if (location.protocol === 'file:') throw new Error(
        'Browsers block direct local-file requests. Choose the original history file above, or run this tool with --serve.');
      const response = await fetch(sourceURL,{cache:'no-store',signal:state.abort.signal});
      if (!response.ok) throw new Error('History request failed: HTTP ' + response.status);
      if (!response.body) throw new Error('Streaming file reads are unavailable; choose the history file above.');
      stream = response.body; total = Number(response.headers.get('Content-Length'));
    }
    const data = await readHistory(stream,total,job); cancelled(job);
    state.data = data; reset();
    tip.textContent = 'Scroll to zoom at the pointer. Drag to pan. Hover for original values.';
  } catch (error) {
    if (job === state.job && error.name !== 'AbortError') message(error.message,true);
  } finally { if (job === state.job) document.querySelector('#reload').disabled = false; }
}
document.querySelector('#reload').onclick = () => load(document.querySelector('#file').files[0] || null);
document.querySelector('#file').onchange = event => { if (event.target.files[0]) load(event.target.files[0]); };
new ResizeObserver(size).observe(canvas);
addEventListener('resize',size);
size(); load();
</script>
</html>
"""


def write_view(history: Path, output: Path) -> None:
    history, output = history.resolve(strict=True), output.resolve()
    if not history.is_file():
        raise ValueError(f"not a history file: {history}")
    if history == output:
        raise ValueError("HTML output must not overwrite the source history")
    try:
        source_url = quote(Path(os.path.relpath(history, output.parent)).as_posix(), safe="/")
    except ValueError:  # Different Windows drives cannot have a relative file URL.
        source_url = history.as_uri()
    metadata = [
        {"name": n, "split": s, "field": f, "key": f"{s}_{f}" if s else f, "color": c}
        for n, s, f, c in SERIES
    ]
    html = HTML
    for key, value in (
        ("__SERIES__", metadata),
        ("__HISTORY_URL__", source_url),
        ("__HISTORY_NAME__", history.name),
    ):
        html = html.replace(key, json.dumps(value).replace("<", "\\u003c"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def serve_view(history: Path, output: Path, port: int) -> None:
    """Serve only the viewer and its original history; never expose other run files."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                file, mime = output, "text/html; charset=utf-8"
            elif path == "/history.jsonl":
                file, mime = history, "application/x-ndjson; charset=utf-8"
            else:
                self.send_error(404)
                return
            try:
                stream = file.open("rb")
            except OSError:
                self.send_error(404, "Source file unavailable")
                return
            with stream:
                remaining = os.fstat(stream.fileno()).st_size
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(remaining))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while remaining:
                        block = stream.read(min(1024 * 1024, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def log_message(self, format: str, *args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"http://127.0.0.1:{server.server_port}/?history=history.jsonl", flush=True)
        print("Reading the original history file. Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, help="original annotation .history.jsonl file")
    parser.add_argument("--output", type=Path, required=True, help="HTML viewer referencing the history")
    parser.add_argument("--serve", action="store_true", help="serve the viewer and original JSONL locally")
    parser.add_argument("--port", type=int, default=0, help="local server port (default: choose a free port)")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        write_view(args.history, args.output)
        if args.serve:
            serve_view(args.history.resolve(), args.output.resolve(), args.port)
        else:
            print(f"Viewer: {args.output.resolve()}")
            print("Use --serve for automatic loading, or choose the source history in the HTML tab.")
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
