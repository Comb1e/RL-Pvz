"""Create a detail-preserving interactive HTML view of annotation-fit history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SERIES = (
    ("train loss", "train", "loss", "#58d68d"),
    ("validation loss", "validation", "loss", "#f5b041"),
    ("train top loss", "train", "top_loss", "#2ecc71"),
    ("validation top loss", "validation", "top_loss", "#e67e22"),
    ("train bottom loss", "train", "bottom_loss", "#1abc9c"),
    ("validation bottom loss", "validation", "bottom_loss", "#d35400"),
    ("L2 penalty", "", "regularization_loss", "#af7ac5"),
)


def _read_history(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or type(row.get("pass_number")) is not int:
            raise ValueError(f"invalid annotation history row {line_number}")
        rows.append(row)
    if not rows:
        raise ValueError("annotation history contains no committed passes")
    return rows


def _payload(rows: list[dict]) -> dict:
    points = []
    for row in rows:
        point = {"pass": row["pass_number"], "regularization_loss": row.get("regularization_loss")}
        for _, split, field, _ in SERIES[:-1]:
            if split:
                value = row.get(split, {}).get(field)
                point[f"{split}_{field}"] = value
        points.append(point)
    return {"series": [{"name": n, "key": (f"{s}_{f}" if s else f), "color": c} for n, s, f, c in SERIES], "points": points}


def write_view(history: Path, output: Path) -> None:
    data = json.dumps(_payload(_read_history(history)), separators=(",", ":")).replace("<", "\\u003c")
    html = f"""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Annotation fit loss</title>
<style>
body{{margin:1rem;background:#101418;color:#e8eef2;font:14px system-ui,sans-serif}}
canvas{{display:block;width:100%;height:min(70vh,620px);background:#171d23;border:1px solid #3a4650;cursor:crosshair}}
#legend{{display:flex;gap:.8rem;flex-wrap:wrap;margin:.7rem 0}}label{{cursor:pointer}}
.swatch{{display:inline-block;width:.8rem;height:.8rem;margin-right:.25rem}}
#tip{{min-height:1.3rem;color:#b8c7d1;font-family:ui-monospace,monospace}}
</style>
<h1>Annotation fit loss</h1>
<p>Every committed point is retained. The canvas shows a per-pixel min/max envelope;
zoom and drag to inspect individual passes.</p>
<div id="legend"></div><canvas id="chart"></canvas><div id="tip"></div>
<script>
const model={data};
const canvas=document.querySelector('#chart'),ctx=canvas.getContext('2d'),tip=document.querySelector('#tip');
const checks=new Map(), state={{left:0,right:model.points.length-1,drag:null}};
for(const s of model.series){{const label=document.createElement('label');const box=document.createElement('input');
box.type='checkbox';box.checked=true;box.onchange=draw;checks.set(s.key,box);const sw=document.createElement('span');
sw.className='swatch';sw.style.background=s.color;label.append(box,sw,s.name);document.querySelector('#legend').append(label)}}
function size(){{const d=devicePixelRatio||1,r=canvas.getBoundingClientRect();canvas.width=Math.max(1,Math.round(r.width*d));canvas.height=Math.max(1,Math.round(r.height*d));ctx.setTransform(d,0,0,d,0,0);draw()}}
function bounds(){{let lo=Infinity,hi=-Infinity;for(const s of model.series)if(checks.get(s.key).checked)for(let i=Math.floor(state.left);i<=Math.ceil(state.right);i++){{const v=model.points[i][s.key];if(Number.isFinite(v)){{lo=Math.min(lo,v);hi=Math.max(hi,v)}}}}if(!Number.isFinite(lo))return [0,1];if(lo===hi){{const p=Math.max(1,Math.abs(lo)*.05);return [lo-p,hi+p]}}const p=(hi-lo)*.06;return [Math.max(0,lo-p),hi+p]}}
function draw(){{const r=canvas.getBoundingClientRect(),w=r.width,h=r.height;ctx.clearRect(0,0,w,h);const [y0,y1]=bounds(),x0=state.left,x1=Math.max(x0+.001,state.right);
ctx.strokeStyle='#33404a';ctx.lineWidth=1;for(let i=0;i<=5;i++){{const y=18+(h-36)*i/5;ctx.beginPath();ctx.moveTo(48,y);ctx.lineTo(w-12,y);ctx.stroke();ctx.fillStyle='#9baab4';ctx.fillText((y1-(y1-y0)*i/5).toPrecision(4),4,y+4)}}
for(const s of model.series){{if(!checks.get(s.key).checked)continue;const buckets=new Map();for(let i=Math.max(0,Math.floor(x0));i<=Math.min(model.points.length-1,Math.ceil(x1));i++){{const v=model.points[i][s.key];if(!Number.isFinite(v))continue;const x=Math.round(52+(model.points[i].pass-model.points[Math.floor(x0)].pass)/(model.points[Math.ceil(x1)].pass-model.points[Math.floor(x0)].pass||1)*(w-64));const b=buckets.get(x)||[Infinity,-Infinity];b[0]=Math.min(b[0],v);b[1]=Math.max(b[1],v);buckets.set(x,b)}}ctx.strokeStyle=s.color;ctx.globalAlpha=.85;for(const [x,b] of buckets){{const yA=18+(y1-b[1])/(y1-y0)*(h-36),yB=18+(y1-b[0])/(y1-y0)*(h-36);ctx.beginPath();ctx.moveTo(x,yA);ctx.lineTo(x,yB);ctx.stroke()}}ctx.globalAlpha=1}}
ctx.fillStyle='#9baab4';ctx.fillText('pass '+model.points[Math.floor(x0)].pass,52,h-6);ctx.fillText('pass '+model.points[Math.ceil(x1)].pass,w-110,h-6)}}
function pointAt(event){{const r=canvas.getBoundingClientRect(),fraction=Math.max(0,Math.min(1,(event.clientX-r.left-52)/(r.width-64))),i=Math.round(state.left+fraction*(state.right-state.left)),p=model.points[i];if(!p)return;tip.textContent='pass '+p.pass+' · train '+p.train_loss+' · validation '+p.validation_loss+' · L2 '+p.regularization_loss}}
canvas.onpointermove=e=>{{if(state.drag!==null){{const dx=(e.clientX-state.drag)*(state.right-state.left)/canvas.getBoundingClientRect().width;state.left=Math.max(0,state.left-dx);state.right=Math.min(model.points.length-1,state.right-dx);state.drag=e.clientX;draw()}}pointAt(e)}};
canvas.onpointerdown=e=>{{state.drag=e.clientX;canvas.setPointerCapture(e.pointerId)}};canvas.onpointerup=e=>{{state.drag=null;canvas.releasePointerCapture(e.pointerId)}};
canvas.onwheel=e=>{{e.preventDefault();const f=Math.exp(e.deltaY*.001),mid=state.left+(e.offsetX/canvas.getBoundingClientRect().width)*(state.right-state.left);state.left=Math.max(0,mid-(mid-state.left)*f);state.right=Math.min(model.points.length-1,mid+(state.right-mid)*f);if(state.right-state.left<1){{state.left=Math.max(0,mid-.5);state.right=Math.min(model.points.length-1,mid+.5)}}draw()}};
addEventListener('resize',size);size();
</script>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, help="annotation .history.jsonl file")
    parser.add_argument("--output", type=Path, required=True, help="standalone HTML output")
    args = parser.parse_args()
    write_view(args.history, args.output)


if __name__ == "__main__":
    main()
