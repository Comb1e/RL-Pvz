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


HTML = """<!doctype html><meta charset=utf-8><title>PVZ training viewer</title>
<style>body{font:14px system-ui;background:#12201a;color:#e4eee5;margin:1rem}
#boards{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:.7rem}
article{background:#22352a;border:1px solid #49634f;border-radius:6px;padding:.7rem}
article.focus{grid-column:1/-1;border-color:#d4b85a}
pre{white-space:pre-wrap;max-height:22rem;overflow:auto;color:#d5e4d7}
button{margin:.2rem;padding:.3rem .6rem;background:#334e3b;color:white;border:1px solid #6b8d70;border-radius:4px}
</style><h1>PVZ training viewer</h1><div id=activity></div><div id=boards></div>
<script>
const esc=x=>String(x??'').replaceAll('&','&amp;').replaceAll('<','&lt;');
const phases=['Starting training','Live training collection','Learner updating','Validation','Reporting'];
const reasons=['','occupied_tile','card_recharging','insufficient_sun','empty_tile','task_restriction','game_finished','outside_space'];
const pages={},requests={},focused={}; let latest=[];
async function command(x){await fetch('./api/command',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(x)});}
function pageFor(i,f){let key=[f.generation,f.episode].join(':');
 return pages[i]&&pages[i].key===key?pages[i].page:(f.history||{start:0,total:0,rows:[]});}
function summary(f){let d=f.decision||{};return {level:f.level,episode:f.episode,outcome:f.outcome,
 decision:d,q:d.q,proposal:d.action,execution:d.executed_action,rejected:d.accepted===false,
rejection_reason:reasons[d.reason]||d.reason,exploration:d.exploration,observation:f.observation};}
function render(){let d=window.liveState||{},packets=d.packets||[];latest=packets;
 document.querySelector('#activity').textContent='Phase: '+(phases[d.activity]||d.activity)+' | '+new Date().toLocaleTimeString();
 document.querySelector('#boards').innerHTML=packets.map((p,i)=>{let f=p.frame||{},h=pageFor(i,f);
 return `<article class="${focused[i]?'focus':''}"><b>Panel ${i+1} · env ${p.env} · ${esc(p.state)}</b>
 <button onclick="command([${i},${p.generation}])">switch</button><button onclick="focused[${i}]=!focused[${i}];render()">focus</button>
 <button onclick="historyPage(${i},-8)">older</button><button onclick="historyPage(${i},8)">newer</button>
 <pre>${esc(JSON.stringify({...summary(f),history:h,learning_settings:d.learning_settings},null,2))}</pre></article>`}).join('');}
async function historyPage(i,delta){let p=latest[i],f=p&&p.frame||{};if(!p||!f.history)return;
 let old=pageFor(i,f),total=Number(old.total||0),start=Math.max(0,Math.min(total,Number(old.start||0)+delta));
 let request=(requests[i]||0)+1;requests[i]=request;
 let query=new URLSearchParams({panel:i,generation:p.generation,env:p.env,episode:f.episode,start,request});
 try{let r=await fetch('./api/history?'+query,{cache:'no-store'}),d=await r.json();
  if(d.request===request&&d.generation===p.generation&&d.episode===f.episode&&d.page){
   pages[i]={key:[p.generation,f.episode].join(':'),page:d.page};render();}
 }catch(e){document.querySelector('#activity').textContent='History reconnecting: '+e;}}
async function refresh(){try{let r=await fetch('./api/state',{cache:'no-store'}),d=await r.json();window.liveState=d;render();
 }catch(e){document.querySelector('#activity').textContent='Viewer reconnecting: '+e}}
setInterval(refresh,500);refresh();
</script>"""


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
        return _jsonable({"activity": self.activity.value, "packets": self.packets,
                          "learning_settings": self.settings})

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

        return {name: integer(name) for name in
                ("panel", "generation", "env", "episode", "start", "request")}

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


def web_viewer_main(frames, commands, history_responses, errors, closed, ready, activity,
                    size, count, settings=(), port_value=None):
    """Process target; all frame conversion and HTTP work stays on CPU."""
    try:
        requested_port = int(port_value.value) if port_value is not None else 0
        server = _Server(("127.0.0.1", requested_port), commands, history_responses, activity,
                         settings)
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
