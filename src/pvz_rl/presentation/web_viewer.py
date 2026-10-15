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


HTML = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>PVZ training viewer</title>
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
<script>
const esc=x=>String(x??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
const phases=['Starting training','Live training collection','Learner updating','Validation','Reporting'];
const reasons=['','occupied_tile','card_recharging','insufficient_sun','empty_tile','task_restriction','game_finished','outside_space'];
const names=['Wait','Sunflower','Peashooter','Wall-nut','Cherry Bomb','Potato Mine','Snow Pea','Chomper','Repeater','Dig'];
const pages={},requests={},focused={};let latest=[];
const n=(v,d=0)=>Number.isFinite(Number(v))?Number(v):d;
const list=v=>Array.isArray(v)?v:[];
const json=v=>{try{return JSON.stringify(v??{},null,2)}catch(_){return '{}'}};
function normalizeObservation(value){let o=value&&typeof value==='object'?value:{},counts=o.counts&&typeof o.counts==='object'?o.counts:{};return {...o,rows:Math.max(1,n(o.rows,5)),cols:Math.max(1,n(o.cols,9)),units_per_tile:Math.max(1,n(o.units_per_tile,80)),cards:list(o.cards),plants:list(o.plants),zombies:list(o.zombies),projectiles:list(o.projectiles),mowers:list(o.mowers),counts:{defeated:n(counts.defeated),alive:n(counts.alive),remaining:n(counts.remaining)}}}
function setActivity(text){document.querySelector('#activity').textContent=text}
async function command(x){try{await fetch('./api/command',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(x)})}catch(e){setActivity('Command reconnecting: '+e)}}
function pageFor(i,f){let key=[f.generation,f.episode].join(':');return pages[i]&&pages[i].key===key?pages[i].page:(f.history||{start:0,total:0,rows:[]})}
function actionName(v){let i=n(v,-1);return names[i]||('Action '+i)}
function reasonName(v){let i=n(v,-1);return reasons[i]||String(v??'unknown')}
function decisionText(f){let d=f.decision;if(!d)return '<span>Waiting for the first recorded decision.</span>';let action=actionName(d.action),execution=d.executed_action===undefined?'—':actionName(d.executed_action),status=d.accepted===false?'rejected: '+esc(reasonName(d.reason)):'accepted',exploration=d.exploration?'exploration':'greedy';return `<strong>${esc(exploration)} ${esc(action)}</strong> → ${esc(execution)} · ${esc(status)}${d.penalty===undefined?'':' · penalty '+n(d.penalty).toPrecision(4)}`}
function qValues(d){return list(d&&d.q).map((q,i)=>`<span class="${q===null||q===undefined?'unavailable':i===n(d.action,-1)?'chosen':''}">${esc(names[i]||i)} ${q===null||q===undefined?'unavailable':n(q).toPrecision(5)}</span>`).join('')}
function drawPlant(ctx,p,x,y,s){let kind=String(p.plant_type||'plant'),colors={sunflower:'#f8cf5e',peashooter:'#7fba56',wall_nut:'#c58c4d',cherry_bomb:'#d85c58',potato_mine:'#c99c59',snow_pea:'#82c9cf',chomper:'#b495e5',repeater:'#70b55b'},c=colors[kind]||'#9bc477';ctx.fillStyle='#41643d';ctx.fillRect(x-4*s,y+20*s,8*s,22*s);ctx.fillStyle=c;ctx.beginPath();ctx.arc(x,y,24*s,0,Math.PI*2);ctx.fill();if(kind==='sunflower'){ctx.fillStyle='#f8cf5e';for(let i=0;i<8;i++){let a=i*Math.PI/4;ctx.beginPath();ctx.arc(x+Math.cos(a)*28*s,y+Math.sin(a)*28*s,8*s,0,Math.PI*2);ctx.fill()}}ctx.fillStyle='#203826';ctx.beginPath();ctx.arc(x-7*s,y-3*s,3*s,0,Math.PI*2);ctx.arc(x+7*s,y-3*s,3*s,0,Math.PI*2);ctx.fill()}
function drawZombie(ctx,z,x,y,s){let slow=n(z.slow_ticks)>0;ctx.fillStyle=slow?'#8ecbe0':'#9db6a7';ctx.beginPath();ctx.arc(x,y-29*s,20*s,0,Math.PI*2);ctx.fill();ctx.fillStyle='#735f75';ctx.fillRect(x-16*s,y-12*s,32*s,38*s);ctx.strokeStyle='#445d49';ctx.lineWidth=7*s;ctx.beginPath();ctx.moveTo(x-8*s,y+24*s);ctx.lineTo(x-13*s,y+43*s);ctx.moveTo(x+8*s,y+24*s);ctx.lineTo(x+13*s,y+43*s);ctx.stroke();if(!z.headless){ctx.fillStyle='#203826';ctx.beginPath();ctx.arc(x-7*s,y-32*s,3*s,0,Math.PI*2);ctx.arc(x+7*s,y-32*s,3*s,0,Math.PI*2);ctx.fill()}if(n(z.armor)>0){ctx.strokeStyle='#dca16a';ctx.lineWidth=5*s;ctx.strokeRect(x-22*s,y-52*s,44*s,12*s)}if(slow){ctx.strokeStyle='#a9e7f2';ctx.lineWidth=2*s;ctx.strokeRect(x-25*s,y-57*s,50*s,65*s)}}
function drawBoard(canvas,obs,frame){let o=normalizeObservation(obs),cssW=Math.max(320,canvas.clientWidth||960),cssH=Math.round(cssW*820/1280),dpr=window.devicePixelRatio||1;canvas.width=Math.round(cssW*dpr);canvas.height=Math.round(cssH*dpr);let ctx=canvas.getContext('2d');if(!ctx)return;ctx.setTransform(dpr*cssW/1280,0,0,dpr*cssW/1280,0,0);ctx.clearRect(0,0,1280,820);ctx.fillStyle='#142724';ctx.fillRect(0,0,1280,820);ctx.fillStyle='#e9eddc';ctx.font='bold 30px system-ui';ctx.fillText('LAWN LAB',34,48);ctx.font='13px system-ui';ctx.fillStyle='#9db49e';ctx.fillText('DAYTIME DEFENSE',236,45);ctx.fillText(`${String(o.level||'board').toUpperCase().slice(0,22)} / ${n(o.elapsed_seconds).toFixed(1)}s`,694,43);
let rows=Math.max(1,n(o.rows,5)),cols=Math.max(1,n(o.cols,9)),tx=98,ty=96,bx=64,by=226;ctx.fillStyle='#5e814c';ctx.fillRect(bx-7,by-7,cols*tx+14,rows*ty+14);for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){ctx.fillStyle=(r+c)%2?'#a7c383':'#b4cc91';ctx.fillRect(bx+c*tx,by+r*ty,tx,ty);ctx.fillStyle='#5d824d';ctx.font='13px system-ui';if(c===0)ctx.fillText(String.fromCharCode(65+r),bx+6,by+r*ty+20)}for(let c=0;c<cols;c++){ctx.fillStyle='#9db49e';ctx.font='13px system-ui';ctx.fillText(c+1,bx+c*tx+tx/2-4,by-23)}
let units=Math.max(1,n(o.units_per_tile,80));for(let m of list(o.mowers)){if(m.state==='spent')continue;let x=bx+n(m.x)/units*tx-22,y=by+n(m.row)*ty+ty*.65;ctx.fillStyle='#d36351';ctx.fillRect(x-20,y-13,40,22);ctx.fillStyle='#203826';ctx.beginPath();ctx.arc(x-11,y+11,7,0,Math.PI*2);ctx.arc(x+11,y+11,7,0,Math.PI*2);ctx.fill()}
for(let p of list(o.plants)){let x=bx+(n(p.col)+.5)*tx,y=by+(n(p.row)+.53)*ty;drawPlant(ctx,p,x,y,.72);if(n(p.max_health)>0&&n(p.health)<n(p.max_health)){ctx.fillStyle='#596b41';ctx.fillRect(x-24,y+34,48,4);ctx.fillStyle='#368442';ctx.fillRect(x-24,y+34,48*Math.max(0,Math.min(1,n(p.health)/n(p.max_health))),4)}}for(let z of list(o.zombies)){let x=bx+(n(z.x)/units+21/80)*tx,y=by+(n(z.row)+.58)*ty;drawZombie(ctx,z,x,y,.68)}for(let p of list(o.projectiles)){let x=bx+n(p.x)/units*tx,y=by+(n(p.row)+.5)*ty-8;ctx.fillStyle=p.icy?'#afe8f1':'#cce276';ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.fill()}
ctx.fillStyle='#3b3f27';ctx.fillRect(978,20,266,70);ctx.fillStyle='#f8cf5e';ctx.beginPath();ctx.arc(1007,54,13,0,Math.PI*2);ctx.fill();ctx.fillStyle='#f8cf5e';ctx.font='bold 30px system-ui';ctx.fillText(n(o.sun),1034,65);ctx.fillStyle='#9db49e';ctx.font='13px system-ui';ctx.fillText('AUTO SUN',1137,58);ctx.fillStyle='#1e362f';ctx.fillRect(978,105,266,100);ctx.fillStyle='#e9eddc';ctx.font='bold 20px system-ui';ctx.fillText(String(o.status||frame?.outcome||'running').toUpperCase(),998,132);ctx.font='13px system-ui';ctx.fillStyle='#9db49e';ctx.fillText(`wave ${n(o.wave)}/${n(o.total_waves)} · defeated ${n(o.counts&&o.counts.defeated)}`,998,163);ctx.fillText(`alive ${n(o.counts&&o.counts.alive)} · remaining ${n(o.counts&&o.counts.remaining)}`,998,185);
let cards=list(o.cards);for(let i=0;i<Math.min(cards.length,8);i++){let c=cards[i],x=34+i*114;ctx.fillStyle='#1f342e';ctx.fillRect(x,84,105,116);ctx.fillStyle='#d5e4d7';ctx.font='12px system-ui';ctx.fillText(String(i+1),x+9,101);ctx.fillText(String(c.plant_type||'plant').replaceAll('_',' '),x+6,181);ctx.fillStyle='#f8cf5e';ctx.fillText(n(c.cost),x+6,195);if(n(c.cooldown_ticks)>0){ctx.fillStyle='#6a936e';ctx.fillRect(x+8,152,89,4);ctx.fillStyle='#9bd477';ctx.fillRect(x+8,152,89*Math.max(0,1-n(c.cooldown_ticks)/Math.max(1,n(c.recharge_ticks))),4)}}
if(frame&&frame.outcome&&frame.outcome!=='running'){ctx.fillStyle='rgba(20,39,36,.82)';ctx.fillRect(420,370,440,115);ctx.fillStyle='#f4efd8';ctx.font='bold 30px system-ui';ctx.fillText(String(frame.outcome).toUpperCase(),565,420);ctx.font='15px system-ui';ctx.fillStyle='#9db49e';ctx.fillText('latest public board snapshot',530,454)}}
function historyTable(h,o){let rows=list(h&&h.rows);if(!rows.length)return '<div class="muted">No plant or dig actions recorded yet.</div>';let rate=Math.max(1,n(o&&o.tick_rate,1)),body=rows.map(r=>{let bad=r.accepted===false,qs=list(r.q).map(q=>`<td>${q===null||q===undefined?'—':n(q).toPrecision(4)}</td>`).join('');return `<tr><td>${esc(r.sequence)}</td><td>${(n(r.tick)/rate).toFixed(2)}</td><td>${esc(actionName(r.action))}</td><td class="${bad?'bad':''}">${bad?'rejected: '+esc(reasonName(r.reason)):'accepted'}</td><td>${r.penalty===undefined?'—':n(r.penalty).toPrecision(4)}</td>${qs}</tr>`}).join('');return `<table><thead><tr><th>#</th><th>time</th><th>proposal</th><th>result</th><th>penalty</th><th colspan="10">Q values (Wait…Dig)</th></tr></thead><tbody>${body}</tbody></table>`}
function render(){let d=window.liveState||{},packets=list(d.packets);latest=packets;setActivity('Phase: '+(phases[d.activity]||d.activity)+' | '+new Date().toLocaleTimeString());document.querySelector('#boards').innerHTML=packets.map((p,i)=>{let f=p.frame||{},h=pageFor(i,f),decision=f.decision||{};return `<article class="${focused[i]?'focus':''}"><div class="panel-head"><span class="panel-title">Panel ${i+1} · env ${esc(p.env)} · ${esc(p.state)}</span><button onclick="command([${i},${n(p.generation)}])">switch</button><button onclick="focused[${i}]=!focused[i];render()">${focused[i]?'grid':'focus'}</button><button onclick="historyPage(${i},-8)">older</button><button onclick="historyPage(${i},8)">newer</button></div><div class="board-wrap"><canvas class="board" id="board-${i}"></canvas></div><div class="caption">${f.observation?`Board ${n(f.observation.elapsed_seconds).toFixed(2)}s · sun ${n(f.observation.sun)} · ${esc(f.outcome||f.observation.status||'running')}`:'Waiting for collection'}</div><div class="decision">${decisionText(f)}<div class="q-values">${qValues(decision)}</div></div><div class="history">${historyTable(h,f.observation||{})}</div><details><summary>Raw frame JSON</summary><pre class="raw">${esc(json(f))}</pre></details><details><summary>Learning settings</summary><pre class="raw">${esc(json(d.learning_settings))}</pre></details></article>`}).join('');packets.forEach((p,i)=>{let c=document.querySelector('#board-'+i);if(c)drawBoard(c,p.frame&&p.frame.observation,p.frame||{})})}
async function historyPage(i,delta){let p=latest[i],f=p&&p.frame||{};if(!p||!f.history)return;let old=pageFor(i,f),total=n(old.total),start=Math.max(0,Math.min(total,n(old.start)+delta)),request=(requests[i]||0)+1;requests[i]=request;let query=new URLSearchParams({panel:i,generation:p.generation,env:p.env,episode:f.episode,start,request});try{let r=await fetch('./api/history?'+query,{cache:'no-store'}),d=await r.json();if(d.request===request&&d.generation===p.generation&&d.episode===f.episode&&d.page){pages[i]={key:[p.generation,f.episode].join(':'),page:d.page};render()}}catch(e){setActivity('History reconnecting: '+e)}}
async function refresh(){try{let r=await fetch('./api/state',{cache:'no-store'}),d=await r.json();window.liveState=d;render()}catch(e){setActivity('Viewer reconnecting: '+e)}}
window.addEventListener('resize',()=>{if(window.liveState)render()});setInterval(refresh,500);refresh();
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
