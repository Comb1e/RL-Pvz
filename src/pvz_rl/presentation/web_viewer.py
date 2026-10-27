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
h1{margin:.2rem 0 .5rem;font-size:1.35rem}#activity{color:#d4b85a;margin-bottom:.7rem}
#boards{display:grid;grid-template-columns:repeat(2,minmax(320px,1fr));gap:.7rem}
article{background:#22352a;border:1px solid #49634f;border-radius:6px;padding:.7rem;min-width:0}
article.focus{grid-column:1/-1;border-color:#d4b85a}.panel-head{display:flex;gap:.45rem;align-items:center;flex-wrap:wrap;margin-bottom:.45rem}
.panel-title{font-weight:700;margin-right:auto}.board-wrap{width:100%;background:#142724;border-radius:5px;overflow:hidden}
canvas.board{display:block;width:100%;height:auto;aspect-ratio:1280/820}.caption{color:#eadb9f;margin:.4rem 0}
.decision{color:#d8e6d6;line-height:1.45;min-height:2.8rem}.decision strong{color:#f4e8af}.q-values{display:flex;flex-wrap:wrap;gap:.3rem .7rem;color:#dce8d7;font:12px Consolas,monospace}
.q-values .chosen{color:#fcdf83}.q-values .unavailable{color:#aeb8ae}.history{overflow:auto;margin-top:.4rem}.history table{border-collapse:collapse;width:100%;font:12px Consolas,monospace;white-space:nowrap}
.history th,.history td{border-bottom:1px solid #405846;padding:.25rem .35rem;text-align:left}.history th{color:#e5eddd;background:#304635;position:sticky;top:0}.history td.bad{color:#fab589}
.muted{color:#b8cdb9}details{margin-top:.45rem;color:#bdceb9}summary{cursor:pointer;color:#d8e6d6}.raw{white-space:pre-wrap;max-height:18rem;overflow:auto;color:#d5e4d7;font:12px Consolas,monospace}
button{margin:.1rem;padding:.3rem .6rem;background:#334e3b;color:white;border:1px solid #6b8d70;border-radius:4px;cursor:pointer}button:hover{background:#496b50}
@media(max-width:760px){body{margin:.55rem}#boards{grid-template-columns:1fr}article.focus{grid-column:auto}.panel-title{flex-basis:100%}}
</style><h1>PVZ training viewer</h1><div id=activity>Connecting…</div><div id=boards></div>
<script>"""
    + BOARD_RENDERER_JS
    + r"""
const phases=['Starting training','Live training collection','Learner updating','Validation','Reporting'];
const pages={},requests={},focused={};let latest=[];
function setActivity(text){document.querySelector('#activity').textContent=text}
async function command(x){try{await fetch('./api/command',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(x)})}catch(e){setActivity('Command reconnecting: '+e)}}
function pageFor(i,f){let key=[f.generation,f.episode].join(':');return pages[i]&&pages[i].key===key?pages[i].page:(f.history||{start:0,total:0,rows:[]})}
function decisionText(f){let d=f.decision;if(!d)return '<span>Waiting for the first recorded decision.</span>';let action=actionName(d.action),execution=d.executed_action===undefined?'—':actionName(d.executed_action),status=d.accepted===false?'rejected: '+esc(reasonName(d.reason)):'accepted',exploration=d.exploration?'exploration':'greedy';return `<strong>${esc(exploration)} ${esc(action)}</strong> → ${esc(execution)} · ${esc(status)}${d.penalty===undefined?'':' · penalty '+n(d.penalty).toPrecision(4)}`}
function qValues(d){return list(d&&d.q).map((q,i)=>`<span class="${q===null||q===undefined?'unavailable':i===n(d.action,-1)?'chosen':''}">${esc(names[i]||i)} ${q===null||q===undefined?'unavailable':n(q).toPrecision(5)}</span>`).join('')}
function historyTable(h,o){let rows=objectList(h&&h.rows);if(!rows.length)return '<div class="muted">No plant or dig actions recorded yet.</div>';let rate=Math.max(1,n(o&&o.tick_rate,1)),body=rows.map(r=>{let bad=r.accepted===false,qs=list(r.q).map(q=>`<td>${q===null||q===undefined?'—':n(q).toPrecision(4)}</td>`).join('');return `<tr><td>${esc(r.sequence)}</td><td>${(n(r.tick)/rate).toFixed(2)}</td><td>${esc(actionName(r.action))}</td><td class="${bad?'bad':''}">${bad?'rejected: '+esc(reasonName(r.reason)):'accepted'}</td><td>${r.penalty===undefined?'—':n(r.penalty).toPrecision(4)}</td>${qs}</tr>`}).join('');return `<table><thead><tr><th>#</th><th>time</th><th>proposal</th><th>result</th><th>penalty</th><th colspan="10">Q values (Wait…Dig)</th></tr></thead><tbody>${body}</tbody></table>`}
function render(){let d=window.liveState||{},packets=objectList(d.packets);latest=packets;setActivity('Phase: '+(phases[d.activity]||d.activity)+' | '+new Date().toLocaleTimeString());document.querySelector('#boards').innerHTML=packets.map((p,i)=>{let f=p.frame&&typeof p.frame==='object'?p.frame:{},h=pageFor(i,f),decision=f.decision&&typeof f.decision==='object'?f.decision:{};return `<article class="${focused[i]?'focus':''}"><div class="panel-head"><span class="panel-title">Panel ${i+1} · env ${esc(p.env)} · ${esc(p.state)}</span><button onclick="command([${i},${n(p.generation)}])">switch</button><button onclick="focused[${i}]=!focused[i];render()">${focused[i]?'grid':'focus'}</button><button onclick="historyPage(${i},-8)">older</button><button onclick="historyPage(${i},8)">newer</button></div><div class="board-wrap"><canvas class="board" id="board-${i}"></canvas></div><div class="caption">${f.observation?`Board ${n(f.observation.elapsed_seconds).toFixed(2)}s · sun ${n(f.observation.sun)} · ${esc(f.outcome||f.observation.status||'running')}${typeof f.reward==='number'&&Number.isFinite(f.reward)?' · Reward '+(f.reward>=0?'+':'')+f.reward.toFixed(6)+(f.reward_pending?' (time pending)':''):''}`:'Waiting for collection'}</div><div class="decision">${decisionText(f)}<div class="q-values">${qValues(decision)}</div></div><div class="history">${historyTable(h,f.observation||{})}</div><details><summary>Raw frame JSON</summary><pre class="raw">${esc(json(f))}</pre></details><details><summary>Raw history JSON</summary><pre class="raw">${esc(json(h))}</pre></details><details><summary>Learning settings</summary><pre class="raw">${esc(json(d.learning_settings))}</pre></details></article>`}).join('');packets.forEach((p,i)=>{let c=document.querySelector('#board-'+i);if(c)drawBoard(c,p.frame&&p.frame.observation,p.frame||{})})}
async function historyPage(i,delta){let p=latest[i],f=p&&p.frame||{};if(!p||!f.history)return;let old=pageFor(i,f),total=n(old.total),start=Math.max(0,Math.min(total,n(old.start)+delta)),request=(requests[i]||0)+1;requests[i]=request;let query=new URLSearchParams({panel:i,generation:p.generation,env:p.env,episode:f.episode,start,request});try{let r=await fetch('./api/history?'+query,{cache:'no-store'}),d=await r.json();if(d.request===request&&d.generation===p.generation&&d.episode===f.episode&&d.page){pages[i]={key:[p.generation,f.episode].join(':'),page:d.page};render()}}catch(e){setActivity('History reconnecting: '+e)}}
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
        if urlparse(self.path).path.rstrip("/") != "/api/command":
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

    def log_message(self, *_args):
        return


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, commands, history_responses, activity, settings=()):
        super().__init__(address, _Handler)
        self.commands, self.history_responses = commands, history_responses
        self.activity, self.settings = activity, settings
        self.packets = []
        self.history_lock = threading.Lock()
        self.pending_history = []

    def state(self):
        return _jsonable(
            {
                "activity": self.activity.value,
                "packets": self.packets,
                "learning_settings": self.settings,
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
):
    """Process target; all frame conversion and HTTP work stays on CPU."""
    try:
        requested_port = int(port_value.value) if port_value is not None else 0
        server = _Server(
            ("127.0.0.1", requested_port), commands, history_responses, activity, settings
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
