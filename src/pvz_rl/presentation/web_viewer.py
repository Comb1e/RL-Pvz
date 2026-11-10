"""CPU-only browser frontend for the existing live-view mailboxes.

The process owns no CUDA state. jupyter-server-proxy (or an equivalent
authenticated reverse proxy) can expose its loopback port without changing
the training process or the viewer's selection/history protocol.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from pvz_rl.presentation.live_view import coerce_grad_scale, read_shared_grad_scale
from pvz_rl.presentation.web_assets import BOARD_RENDERER_JS


def _jsonable(value):
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value") and isinstance(value.value, (str, int, float)):
        return value.value
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


HTML = (
    """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>PVZ training viewer</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{font:14px system-ui;background:#12201a;color:#e4eee5;margin:1rem}
#top{display:flex;gap:.8rem;align-items:center;margin-bottom:.7rem;flex-wrap:wrap}#activity{color:#d4b85a;margin:0}
#grad{display:flex;gap:.45rem;align-items:center;margin:0}#grad input{width:4.6rem;background:#182720;color:#e4eee5;border:1px solid #6b8d70;border-radius:4px;padding:.15rem .35rem;font:14px Consolas,monospace}#grad output{font:14px Consolas,monospace}#grad output.now{color:#8aa392}#grad input.bad{border-color:#d85c58}
#boards{display:grid;grid-template-columns:repeat(2,minmax(320px,1fr));gap:.7rem}
article{background:#22352a;border:1px solid #49634f;border-radius:6px;padding:.7rem;min-width:0}
article.focus{grid-column:1/-1}article.result{border-color:#d4b85a}article.selecting{border-style:dashed}
.panel-head{display:flex;gap:.45rem;align-items:center;flex-wrap:wrap;margin-bottom:.45rem}
.panel-title{font-weight:700;margin-right:auto}.board-row{display:grid;grid-template-columns:minmax(0,3fr) minmax(220px,2fr);gap:.5rem;align-items:stretch}.board-wrap{width:100%;background:#142724;border-radius:5px;overflow:hidden}
canvas.board{display:block;width:100%;height:auto;aspect-ratio:1280/820}
.outcome-live{min-width:0;background:#182720;padding:.45rem;border-radius:5px;display:flex;flex-direction:column;gap:.35rem}
.band{display:grid;grid-template-columns:.4rem 1fr;gap:.35rem;align-items:stretch}.swatch{border-radius:2px}.swatch.field{background:#3f9a62}.swatch.breach{background:#d6a448}.swatch.probe{background:#3d8fbf}
.outcome-grid,.breach-grid{display:grid;gap:1px;font-size:10px;text-align:center}.outcome-grid{grid-template-columns:.8rem repeat(10,minmax(0,1fr))}
.outcome-cell{aspect-ratio:1.15;min-height:8px}.outcome-label{align-self:center;color:#b8cdb9;font-size:10px}
.breach-grid{grid-template-columns:.8rem repeat(5,minmax(0,1fr)) .35rem minmax(0,1fr)}
.q-values{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.2rem .4rem;font:12px Consolas,monospace;margin-top:.35rem}
.q-values span{display:flex;gap:.3rem;align-items:center;min-width:0}.q-values i,.dot{width:.55rem;height:.55rem;border-radius:50%;display:inline-block;flex:none}
.q-values .chosen{background:#5c4e28;border-radius:3px;padding:0 .15rem}.q-values .explored{background:#785028;border-radius:3px;padding:0 .15rem}.q-values .unavailable{opacity:.35}.chip{display:flex;gap:.35rem;align-items:center;color:#f4e7c2;font-weight:700;margin-top:.3rem}
.history{overflow:auto;margin-top:.35rem}.history table{border-collapse:collapse;width:100%;font:12px Consolas,monospace;white-space:nowrap}
.history td{border-bottom:1px solid #405846;padding:.2rem .35rem;text-align:left}.history tr.bad td{background:#5c3028}
#settings{margin:0 0 .6rem}#settings summary{list-style:none;cursor:pointer;width:1.2rem}#settings summary::-webkit-details-marker{display:none}#settings summary::before{content:"▸"}#settings[open] summary::before{content:"▾"}.raw{white-space:pre-wrap;max-height:18rem;overflow:auto;color:#d5e4d7;font:12px Consolas,monospace}
button{margin:.1rem;padding:.3rem .6rem;background:#334e3b;color:white;border:1px solid #6b8d70;border-radius:4px;cursor:pointer}button:hover{background:#496b50}
@media(max-width:760px){body{margin:.55rem}#boards{grid-template-columns:1fr}article.focus{grid-column:auto}.panel-title{flex-basis:100%}}
</style><div id=top><div id=activity>…</div><form id=grad onsubmit="submitGrad(event)"><input id=grad-scale value="1" inputmode="decimal" autocomplete=off oninput="this.classList.remove('bad')"><output id=grad-next>1</output><output id=grad-now class=now>1</output></form></div><details id=settings hidden><summary></summary><pre class=raw></pre></details><div id=boards></div>
<script>"""
    + BOARD_RENDERER_JS
    + r"""
const phases=['Starting','Collecting','Updating','Validating','Reporting'];
const pages={},requests={},focused={};let latest=[];
function setActivity(text){document.querySelector('#activity').textContent=text}
function formatScale(value){let n=Number(value);return Number.isFinite(n)&&n>0?String(n):'1'}
function showGrad(d){let input=document.querySelector('#grad-scale');document.querySelector('#grad-next').textContent=formatScale(d.grad_scale_requested);document.querySelector('#grad-now').textContent=formatScale(d.grad_scale_active);if(document.activeElement!==input)input.value=formatScale(d.grad_scale_requested)}
async function submitGrad(event){event.preventDefault();let input=document.querySelector('#grad-scale');try{let r=await fetch('./api/grad-scale',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({value:input.value})});input.classList.toggle('bad',!r.ok)}catch(e){input.classList.add('bad')}}
async function command(x){try{await fetch('./api/command',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(x)})}catch(e){setActivity('…')}}
function pageFor(i,f){let key=[f.generation,f.episode].join(':');return pages[i]&&pages[i].key===key?pages[i].page:(f.history||{start:0,total:0,rows:[]})}
const cardColors=['#8c948e','#f8cf5e','#7fba56','#c58c4d','#d85c58','#c99c59','#82c9cf','#b495e5','#70b55b','#c4a478'];
function depthColor(v,hue){if(!Number.isFinite(v))return '#37403c';let t=Math.max(0,Math.min(1,v));return `rgb(${hue[0][0]+(hue[1][0]-hue[0][0])*t},${hue[0][1]+(hue[1][1]-hue[0][1])*t},${hue[0][2]+(hue[1][2]-hue[0][2])*t})`}
function depthGrid(values,hue,axis){let grid=axis?'<span></span>'+['H',1,2,3,4,5,6,7,8,9].map(c=>`<span class="outcome-label">${c}</span>`).join(''):'';for(let r=0;r<5;r++){grid+=`<span class="outcome-label">${String.fromCharCode(65+r)}</span>`;for(let c=0;c<10;c++){let raw=values&&values[r]&&values[r][c],v=Number.isFinite(raw)?raw:null;grid+=`<span class="outcome-cell" style="background:${depthColor(v,hue)}"></span>`}}return `<div class="outcome-grid">${grid}</div>`}
function breachGrid(classes){let grid='';for(let r=0;r<5;r++){let chosen=classes?Number(classes[r]):NaN;grid+=`<span class="outcome-label">${String.fromCharCode(65+r)}</span>`;for(let c=0;c<5;c++){let on=Number.isFinite(chosen)&&chosen<5&&c<=chosen;grid+=`<span class="outcome-cell" style="background:${on?'#d6a448':'#2a3430'}"></span>`}grid+='<span></span>';let none=chosen===5;grid+=`<span class="outcome-cell" style="background:${none?'#d6a448':'#2a3430'}"></span>`}return `<div class="breach-grid">${grid}</div>`}
function outcomePrediction(f){let d=f.outcome_decision||f.decision||{},known=!!d.effect_prediction_valid,grids=known?d.effect_prediction:null,breach=d.breach_prediction_valid?d.breach_prediction:null;return `<aside class="outcome-live"><div class="band"><i class="swatch field"></i>${depthGrid(grids&&grids[0],[[22,120,70],[190,220,140]],true)}</div><div class="band"><i class="swatch breach"></i>${breachGrid(breach)}</div><div class="band"><i class="swatch probe"></i>${depthGrid(grids&&grids[1],[[28,70,100],[150,210,230]],false)}</div></aside>`}
function dot(i){return i?`<i class="dot" style="background:${cardColors[i]||'#8c948e'}"></i>`:`<i class="dot" style="background:transparent;box-shadow:inset 0 0 0 1.5px #d5e4d7"></i>`}
function qValues(d){if(!scoresRecorded(d))return '';let chosen=branchOf(d.action),explored=n(d.action)!==n(d.greedy_action);return `<div class="q-values">${list(d.q).map((q,i)=>{let off=q===null||q===undefined||d.legal&&d.legal[i]===false,cls=off?'unavailable':i===chosen?(explored?'explored':'chosen'):'';return `<span class="${cls}">${dot(i)}${q===null||q===undefined?'':n(q).toFixed(1)}</span>`}).join('')}</div>`}
function historyTable(h,o,recorded){let rows=objectList(h&&h.rows);if(!rows.length)return '';let rate=Math.max(1,n(o&&o.tick_rate,1));return `<div class="history"><table><tbody>${rows.map(r=>{let bad=r.accepted===false,qs=recorded?list(r.q).map(q=>`<td>${q===null||q===undefined?'':n(q).toFixed(1)}</td>`).join(''):'';return `<tr class="${bad?'bad':''}"><td>${esc(r.sequence)}</td><td>${(n(r.tick)/rate).toFixed(1)}</td><td>${dot(branchOf(r.action))}</td><td>${esc(tileOf(r.action))}</td>${qs}</tr>`}).join('')}</tbody></table></div>`}

function showSettings(d){let box=document.querySelector('#settings'),rows=list(d.learning_settings);if(!rows.length){box.hidden=true;return}box.hidden=false;let text=rows.map(row=>[row[0],...list(row[1])].filter(Boolean).join('  ')).join('\n');let pre=box.querySelector('pre');if(pre.textContent!==text)pre.textContent=text}
function render(){let d=window.liveState||{},packets=objectList(d.packets);latest=packets;setActivity(phases[d.activity]||d.activity||'');showGrad(d);showSettings(d);document.querySelector('#boards').innerHTML=packets.map((p,i)=>{let f=p.frame&&typeof p.frame==='object'?p.frame:{},h=pageFor(i,f),decision=f.decision&&typeof f.decision==='object'?f.decision:{},recorded=scoresRecorded(decision),state=p.state==='selecting'?'selecting':(f.outcome&&f.outcome!=='running'?'result':''),task=f.task||'',stamp='';if(task==='prediction'&&decision.effect_prediction_valid&&f.observation)stamp=' '+(n(decision.tick)/Math.max(1,n(f.observation.tick_rate,1))).toFixed(1);else if(task!=='prediction'&&!['cherry_bomb','cherry_bomb_obstacle','attacker_pre','attacker'].includes(task)&&typeof f.reward==='number')stamp=' '+(f.reward>=0?'+':'')+n(f.reward).toFixed(2)+(f.reward_pending?'~':'');let chip=decision.action>0&&!recorded?`<div class="chip">${dot(branchOf(decision.action))}${esc(tileOf(decision.action))}</div>`:'';let live=!pages[i];return `<article class="${focused[i]?'focus':''} ${state}"><div class="panel-head"><span class="panel-title">${esc(p.env)} ${esc(task)}${esc(stamp)}</span><button onclick="command([${i},${n(p.generation)}])">&lt;&gt;</button><button onclick="focused[${i}]=!focused[${i}];render()">+</button><button onclick="historyPage(${i},-8)">^</button><button onclick="historyPage(${i},8)" style="${live?'background:#5a7a4e':''}">v</button></div><div class="board-row"><div class="board-wrap"><canvas class="board" id="board-${i}"></canvas></div>${outcomePrediction(f)}</div>${chip}${qValues(decision)}${historyTable(h,f.observation||{},recorded)}</article>`}).join('');packets.forEach((p,i)=>{let c=document.querySelector('#board-'+i);if(c)drawBoard(c,p.frame&&p.frame.observation,p.frame||{})})}
async function historyPage(i,delta){let p=latest[i],f=p&&p.frame||{};if(!p||!f.history)return;let old=pageFor(i,f),total=n(old.total),start=Math.max(0,Math.min(total,n(old.start)+delta));if(delta>0&&start>=total){delete pages[i];render();return}let request=(requests[i]||0)+1;requests[i]=request;let query=new URLSearchParams({panel:i,generation:p.generation,env:p.env,episode:f.episode,start,request});try{let r=await fetch('./api/history?'+query,{cache:'no-store'}),d=await r.json();if(d.request===request&&d.generation===p.generation&&d.episode===f.episode&&d.page){pages[i]={key:[p.generation,f.episode].join(':'),page:d.page};render()}}catch(e){setActivity('History reconnecting: '+e)}}
async function refresh(){try{let r=await fetch('./api/state',{cache:'no-store'}),d=await r.json();window.liveState=d;render()}catch(e){setActivity('Viewer reconnecting: '+e)}}
window.addEventListener('resize',()=>{if(window.liveState)render()});setInterval(refresh,500);refresh();
</script>"""
)


class _Handler(BaseHTTPRequestHandler):
    server_version = "pvz-web-viewer/1"

    def _reply(self, status, body, content_type="application/json"):
        raw = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if path in ("", "/index.html"):
            return self._reply(200, HTML, "text/html")
        if path == "/api/state":
            return self._reply(200, json.dumps(self.server.state()))
        if path == "/api/history":
            try:
                request = self.server.parse_history(parse_qs(parsed.query))
            except (KeyError, TypeError, ValueError) as exc:
                return self._reply(400, json.dumps({"error": f"invalid history request: {exc}"}))
            response = self.server.history(request)
            status = 200 if response.get("page") is not None else 504
            return self._reply(status, json.dumps(response))
        return self._reply(404, json.dumps({"error": "not found"}))

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path.rstrip("/")
        if path == "/api/grad-scale":
            return self._submit_grad_scale()
        if path != "/api/command":
            return self._reply(404, json.dumps({"error": "not found"}))
        try:
            length = int(self.headers.get("Content-Length", 0))
            command = json.loads(self.rfile.read(length))
            if (
                not isinstance(command, list)
                or len(command) != 2
                or any(type(value) is not int or value < 0 for value in command)
            ):
                raise ValueError("switch command must be [panel, generation]")
            self.server.commands.put_nowait(command)
        except (ValueError, json.JSONDecodeError):
            return self._reply(400, json.dumps({"error": "invalid command"}))
        except queue.Full:
            return self._reply(429, json.dumps({"error": "command queue full"}))
        return self._reply(202, json.dumps({"accepted": True}))

    def _submit_grad_scale(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            accepted = coerce_grad_scale(body.get("value") if isinstance(body, dict) else None)
            share = self.server.grad_requested
            if accepted is None or share is None:
                raise ValueError("gradient scale must be a finite positive number")
            with share.get_lock():
                share.value = accepted
        except (ValueError, json.JSONDecodeError, TypeError):
            return self._reply(400, json.dumps({"error": "invalid gradient scale"}))
        return self._reply(202, json.dumps({"accepted": accepted}))

    def log_message(self, *_args):
        return


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address,
        commands,
        history_responses,
        activity,
        settings=(),
        grad_requested=None,
        grad_active=None,
    ):
        super().__init__(address, _Handler)
        self.commands, self.history_responses = commands, history_responses
        self.activity, self.settings = activity, settings
        self.grad_requested, self.grad_active = grad_requested, grad_active
        self.packets = []
        self.history_lock = threading.Lock()
        self.pending_history = []

    def state(self):
        requested, active = read_shared_grad_scale(self.grad_requested, self.grad_active)
        return _jsonable(
            {
                "activity": self.activity.value,
                "packets": self.packets,
                "learning_settings": self.settings,
                "grad_scale_requested": requested,
                "grad_scale_active": active,
            }
        )

    @staticmethod
    def parse_history(query):
        def integer(name, *, minimum=0):
            values = query.get(name)
            if not values or len(values) != 1:
                raise ValueError(f"{name} is required")
            value = int(values[0])
            if value < minimum:
                raise ValueError(f"{name} must be non-negative")
            return value

        return {
            name: integer(name)
            for name in ("panel", "generation", "env", "episode", "start", "request")
        }

    def history(self, request):
        """Forward one page request and wait only in this CPU viewer process."""
        with self.history_lock:
            for response in list(self.pending_history):
                if all(response.get(key) == value for key, value in request.items()):
                    self.pending_history.remove(response)
                    return _jsonable(response)
            try:
                self.commands.put_nowait(request)
            except queue.Full:
                return {**request, "page": None, "error": "command queue full"}
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                try:
                    response = self.history_responses.get(timeout=0.05)
                except queue.Empty:
                    continue
                if all(response.get(key) == value for key, value in request.items()):
                    return _jsonable(response)
                self.pending_history.append(response)
                if len(self.pending_history) > 8:
                    self.pending_history.pop(0)
            return {**request, "page": None, "error": "history response timed out"}


def web_viewer_main(
    frames,
    commands,
    history_responses,
    errors,
    closed,
    ready,
    activity,
    size,
    count,
    settings=(),
    port_value=None,
    grad_requested=None,
    grad_active=None,
):
    """Process target; all frame conversion and HTTP work stays on CPU."""
    try:
        requested_port = int(port_value.value) if port_value is not None else 0
        server = _Server(
            ("127.0.0.1", requested_port),
            commands,
            history_responses,
            activity,
            settings,
            grad_requested,
            grad_active,
        )
        if port_value is not None:
            port_value.value = server.server_port
        ready.set()
        server.timeout = 0.1
        while not closed.is_set():
            try:
                server.packets = frames.get_nowait()
            except queue.Empty:
                pass
            server.handle_request()
        server.server_close()
    except Exception as exc:
        try:
            errors.put_nowait(f"{type(exc).__name__}: {exc}")
        except queue.Full:
            pass
