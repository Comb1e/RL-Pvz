"""CPU-only, one-map-at-a-time browser annotator."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from pvz_rl.annotation.dataset import LiveAnnotationStore
from pvz_rl.annotation.sampler import ReachableSnapshotSampler
from pvz_rl.config import digest
from pvz_rl.presentation.web_assets import BOARD_RENDERER_JS

DIFFICULTIES = ("easy", "standard", "hard")


ANNOTATION_HTML = (
    """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>PVZ annotation studio</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{font:14px system-ui;background:#12201a;color:#e4eee5;margin:1rem}
h1{margin:.2rem 0 .5rem;font-size:1.35rem}#activity{color:#d4b85a;margin-bottom:.7rem}
#toolbar{display:flex;gap:.4rem;align-items:center;flex-wrap:wrap;background:#22352a;border:1px solid #49634f;border-radius:6px;padding:.55rem;margin-bottom:.7rem}
#toolbar button{margin:.1rem;padding:.35rem .7rem;background:#334e3b;color:white;border:1px solid #6b8d70;border-radius:4px;cursor:pointer}
#toolbar button.active{background:#7c6732;border-color:#eadb9f}#toolbar button.absolute{background:#7d493f;border-color:#f3a08a}#toolbar button.locked{background:#5d5435;border-color:#eadb9f;cursor:not-allowed}#toolbar button:disabled,#difficulty:disabled{opacity:.45;cursor:not-allowed}
#branch-controls{display:flex;gap:.2rem;align-items:center;padding-left:.25rem;border-left:1px solid #49634f}#branch-controls button{font-size:12px;padding:.25rem .5rem}
#status{margin-left:auto;color:#d8e6d6}.board-wrap{max-width:1280px;background:#142724;border-radius:6px;overflow:hidden}
canvas.board{display:block;width:100%;height:auto;aspect-ratio:1280/820;touch-action:none;cursor:crosshair}
#facts{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:.6rem;margin-top:.7rem}
.fact{background:#22352a;border:1px solid #49634f;border-radius:6px;padding:.6rem;min-height:3rem}.fact b{color:#f4e8af}
details{margin-top:.6rem;color:#bdceb9}summary{cursor:pointer;color:#d8e6d6}.raw{white-space:pre-wrap;max-height:22rem;overflow:auto;color:#d5e4d7;font:12px Consolas,monospace}
@media(max-width:760px){body{margin:.55rem}#facts{grid-template-columns:1fr}.board-wrap{width:100%}}
</style><h1>PVZ annotation studio</h1><div id=activity>Choose a difficulty to generate a reachable map.</div>
<div id=toolbar><button id=previous disabled>Previous</button><button id=next disabled>Next</button><button id=current disabled>Return to current</button><button id=reload disabled>Reload map</button><label>Difficulty <select id=difficulty><option value="">Choose…</option></select></label><label id=plant-label>Plant <select id=plant disabled></select></label><div id=branch-controls hidden><button id=wait-action data-action=wait disabled>Wait</button><button id=dig-action data-action=dig disabled>Dig</button></div><button id=add class=active disabled>Paint feasible</button><button id=absolute-mode disabled>Paint absolute-infeasible</button><button id=erase disabled>Erase</button><button id=clear disabled>Clear</button><button id=save disabled>Save and next</button><button id=save-exit disabled>Save and exit</button><button id=discard disabled>Discard state</button><button id=retry hidden>Retry generation</button><span id=status></span></div>
<div class=board-wrap id=board-wrap hidden><canvas class=board id=board></canvas></div><div id=facts></div>
<details><summary>Raw current record JSON</summary><pre class=raw id=raw></pre></details>
<script>"""
    + BOARD_RENDERER_JS
    + r"""
const app={state:null,draft:new Set(),absolute:new Set(),mode:'add',dirty:false,busy:false,error:null,request:0,lastTile:null,exiting:false};
const $=id=>document.getElementById(id);
function setActivity(value){$('activity').textContent=value}
function currentRecord(){return app.state&&app.state.record||null}
function isActionMode(){let r=currentRecord();return !!(app.state&&app.state.annotation_kind==='action'||r&&r.annotation_kind==='action')}
function actionOrder(){let r=currentRecord();return r&&r.action_availability?Object.keys(r.action_availability):[]}
function sortLabels(values){let order=actionOrder(),rank=value=>{let index=order.indexOf(value);return index<0?order.length:index};return [...new Set(values)].sort((a,b)=>rank(a)-rank(b)||String(a).localeCompare(String(b)))}
function selected(){return isActionMode()?sortLabels(app.draft):[...app.draft].sort((a,b)=>a-b)}
function selectedAbsolute(){return isActionMode()?sortLabels(app.absolute):[...app.absolute].sort((a,b)=>a-b)}
function toggleAction(name,mode){let r=currentRecord(),automatic=new Set(list(r&&r.automatic_absolute_infeasible_actions));if(!name||automatic.has(name))return;if(mode==='erase'){app.draft.delete(name);app.absolute.delete(name)}else if(mode==='absolute'){if(app.absolute.has(name))app.absolute.delete(name);else{app.absolute.add(name);app.draft.delete(name)}}else if(app.draft.has(name))app.draft.delete(name);else{app.draft.add(name);app.absolute.delete(name)}app.dirty=true;render()}
function setDisabled(disabled){for(let id of ['plant','wait-action','dig-action','add','absolute-mode','erase','clear','save','save-exit','discard'])$(id).disabled=disabled}
function syncBranchControls(isAction,disabled){let wrap=$('branch-controls'),r=currentRecord(),automatic=new Set(list(r&&r.automatic_absolute_infeasible_actions));wrap.hidden=!isAction||!r;for(let name of ['wait','dig']){let button=$(`${name}-action`),locked=automatic.has(name);button.disabled=disabled||locked;button.classList.toggle('active',isAction&&app.draft.has(name));button.classList.toggle('absolute',isAction&&app.absolute.has(name));button.classList.toggle('locked',locked);button.textContent=locked?`${name[0].toUpperCase()+name.slice(1)} (locked)`:name[0].toUpperCase()+name.slice(1)}}
function render(){
 let s=app.state||{},r=currentRecord(),nav=s.navigation||{},mode=s.mode||'idle',conflict=s.conflict,select=$('difficulty'),isAction=!!(s.annotation_kind==='action'||r&&r.annotation_kind==='action');
 select.textContent='';let placeholder=document.createElement('option');placeholder.value='';placeholder.textContent='Choose…';select.appendChild(placeholder);
 for(let level of list(s.difficulties)){let option=document.createElement('option');option.value=level;option.textContent=level;select.appendChild(option)}
 select.value=s.difficulty||'';select.disabled=app.busy||mode==='conflict';
 let counts=s.counts||{};$('status').textContent=`generated ${n(counts.generated)} · labeled ${n(counts.labeled)} · discarded ${n(counts.discarded)}${n(s.conflicts_remaining)?' · conflicts '+n(s.conflicts_remaining):''}`;
 $('previous').disabled=app.busy||mode==='conflict'||!nav.previous_id;$('next').disabled=app.busy||mode==='conflict'||!nav.next_id;
 $('current').disabled=app.busy||mode==='conflict'||!nav.pending_id||!!r&&r.id===nav.pending_id;
 $('reload').disabled=app.busy||!r;$('retry').hidden=!s.error;$('retry').disabled=app.busy;
 $('plant-label').hidden=!r||isAction;setDisabled(app.busy||!r);syncBranchControls(isAction,app.busy||!r);$('board-wrap').hidden=!r;
 if(!r){$('facts').textContent='';$('raw').textContent='';setActivity(app.error||s.error||(s.difficulty?'Generating a reachable map…':'Choose a difficulty to generate a reachable map.'));return}
 let a=r.availability||{},current=r.difficulty||r.observation&&r.observation.level;
 if(!isAction){$('plant').textContent='';let option=document.createElement('option');option.value=r.plant_type||'';option.textContent=r.plant_type||'action labels';$('plant').appendChild(option)}else $('plant').textContent='';
 $('save').textContent=mode==='conflict'?'Save correction':nav.reviewing?'Save changes':'Save and next';$('save').disabled=app.busy||!availabilityValid();$('save-exit').disabled=app.busy||!availabilityValid();$('discard').disabled=app.busy||!r||mode==='conflict';
 let title=mode==='conflict'?`Duplicate annotation conflict ${n(s.conflicts_index)} / ${n(s.conflicts_remaining)}`:`Map ${n(nav.index)+1} / ${n(nav.total)}`;
 setActivity(app.error||`${title} · ${current} · ${r.status||'pending'}${app.dirty?' · unsaved changes':''}${nav.reviewing?' · reviewing saved map':''}${s.difficulty!==current?' · next '+s.difficulty:''}${s.error?' · generation paused: '+s.error:''}`);
 let boardOptions={annotation:true};if(isAction)boardOptions.action={feasible:selected(),absolute:selectedAbsolute(),automatic:list(r.automatic_absolute_infeasible_actions),availability:r.action_availability||{}};else Object.assign(boardOptions,{selected:selected(),absolute:selectedAbsolute(),automatic:list(r.automatic_absolute_infeasible_tiles),legal:list(a.legal_tiles),mode:app.mode});drawBoard($('board'),r.observation,{outcome:r.observation&&r.observation.status},boardOptions);
 let memberFact=conflict?`<div class=fact><b>Existing annotations</b><br>${list(conflict.members).map(member=>`${esc(member.id)}: [${list(member.feasible_actions||member.feasible_tiles).join(', ')}] · absolute [${list(member.absolute_infeasible_actions||member.absolute_infeasible_tiles).join(', ')}]`).join('<br>')}</div>`:'';
 let boardFact=`<div class=fact><b>Board</b><br>${n(r.observation&&r.observation.sun)} sun · wave ${n(r.observation&&r.observation.wave)}/${n(r.observation&&r.observation.total_waves)}<br>${list(r.observation&&r.observation.plants).length} plants · ${list(r.observation&&r.observation.zombies).length} zombies</div>`;
 let facts=isAction?`<div class=fact><b>Action availability</b><br>${Object.entries(r.action_availability||{}).filter(([,item])=>item&&item.feasible).map(([name])=>esc(name)).join(', ')||'none'} feasible<br>automatic absolute: ${list(r.automatic_absolute_infeasible_actions).map(esc).join(', ')||'none'}</div>${boardFact}`:`<div class=fact><b>Plant</b><br>${esc(r.plant_type)}<br>cost ${n(a.cost)} · sun ${n(a.sun)}</div><div class=fact><b>Availability</b><br>${a.available?'available':'unavailable'} · cooldown ${n(a.cooldown_ticks)}<br>legal plots ${list(a.legal_tiles).length}</div>${boardFact}`;
 $('facts').innerHTML=facts+memberFact;$('raw').textContent=json(conflict?{record:r,conflict}:r)
}
function applyState(data,request,preserveDraft){
 if(data.request!==request||request!==app.request)return;
 let previous=currentRecord(),record=data.record;
 let keep=preserveDraft&&previous&&record&&previous.id===record.id&&n(previous.revision)===n(record.revision)&&data.mode!=='conflict';
 app.state=data;if(!keep){let isAction=data.annotation_kind==='action'||record&&record.annotation_kind==='action';app.draft=data.mode==='conflict'?new Set():new Set(list(record&&(isAction?record.feasible_actions:record.feasible_tiles)));app.absolute=new Set(list(record&&(isAction?record.absolute_infeasible_actions:record.absolute_infeasible_tiles)));app.dirty=false;app.lastTile=null}
}
async function request(path,body=null,preserveDraft=false){
 if(app.busy)return;let request=++app.request;app.busy=true;app.error=null;render();
 try{
  let options=body===null?{cache:'no-store'}:{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({...body,request})};
  let url=body===null?path+(path.includes('?')?'&':'?')+'request='+request:path;
  let response=await fetch(url,options),data=await response.json();
  if(request!==app.request)return;if(!response.ok||data.error)throw Error(data.error||'HTTP '+response.status);
  applyState(data,request,preserveDraft);
 }catch(error){if(request===app.request)app.error='Request failed: '+error+' · Reload map to fetch saved labels.'}
 finally{if(request===app.request){app.busy=false;render()}}
}
function refresh(){let r=currentRecord();return request('./api/state'+(r?'?id='+encodeURIComponent(r.id):''))}
function canLeave(){return !app.dirty||window.confirm('Leave this map and discard unsaved changes?')}
function navigate(direction){let r=currentRecord();if(r&&app.state.mode!=='conflict'&&canLeave())request('./api/navigate',{id:r.id,direction})}
function paintTile(tile,legal){if(tile===null||!legal.has(tile))return;let r=currentRecord(),automatic=new Set(list(r&&r.automatic_absolute_infeasible_tiles)),locked=automatic.has(tile);if(app.mode==='absolute'&&locked)return;if(app.mode==='erase'){app.draft.delete(tile);if(!locked)app.absolute.delete(tile);return}if(app.mode==='absolute'){app.draft.delete(tile);app.absolute.add(tile);return}if(locked)return;app.absolute.delete(tile);app.draft.add(tile)}
function paint(event){let r=currentRecord();if(!r||isActionMode()||app.busy||event.buttons!==1)return;let tile=tileIndexAt($('board'),r.observation,event),legal=new Set((app.state&&app.state.annotation_kind==='tile')?Array.from({length:45},(_,i)=>i):list(r.availability&&r.availability.legal_tiles));if(tile===null)return;let previous=app.lastTile===null?tile:app.lastTile,cols=Math.max(1,n(r.observation&&r.observation.cols,9)),r0=Math.floor(previous/cols),c0=previous%cols,r1=Math.floor(tile/cols),c1=tile%cols,steps=Math.max(Math.abs(r1-r0),Math.abs(c1-c0));for(let step=0;step<=steps;step++){let fraction=steps?step/steps:0;paintTile(Math.round(r0+(r1-r0)*fraction)*cols+Math.round(c0+(c1-c0)*fraction),legal)}app.lastTile=tile;app.dirty=true;render()}
function setMode(mode){app.mode=mode;$('add').classList.toggle('active',mode==='add');$('absolute-mode').classList.toggle('active',mode==='absolute');$('erase').classList.toggle('active',mode==='erase');render()}
function availabilityValid(){let r=currentRecord(),isAction=app.state&&app.state.annotation_kind==='action';if(isAction)return selected().length>0;let a=r&&r.availability||{};return selected().length>0&&a.available===true&&list(a.legal_tiles).length>0&&n(a.cooldown_ticks,1)===0&&n(a.sun,-1)>=n(a.cost,Infinity)&&$('plant').value===r.plant_type}
for(let [id,mode] of [['add','add'],['erase','erase']])$(id).onclick=()=>setMode(mode);
$('absolute-mode').onclick=()=>setMode('absolute');
$('wait-action').onclick=()=>{if(isActionMode())toggleAction('wait',app.mode)};$('dig-action').onclick=()=>{if(isActionMode())toggleAction('dig',app.mode)};
$('difficulty').onchange=()=>{let level=$('difficulty').value,r=currentRecord();if(level)request('./api/difficulty',{difficulty:level,...(r?{id:r.id}:{})},true)};$('clear').onclick=()=>{app.draft.clear();app.dirty=true;render()};
 function saveCurrent(exit){if(!availabilityValid()){setActivity('Cannot save: choose at least one feasible label');return}let r=currentRecord(),isAction=app.state.annotation_kind==='action';app.exiting=exit;if(app.state.mode==='conflict'){let c=app.state.conflict;request('./api/conflict/resolve',{key:c.key,members:list(c.members).map(member=>({id:member.id,revision:n(member.revision)})),...(isAction?{feasible_actions:selected(),absolute_infeasible_actions:selectedAbsolute()}:{tiles:selected(),absolute_infeasible_tiles:selectedAbsolute()}),exit})}else request('./api/resolve',{id:r.id,revision:n(r.revision),status:'labeled',...(isAction?{feasible_actions:selected(),absolute_infeasible_actions:selectedAbsolute()}:{tiles:selected(),absolute_infeasible_tiles:selectedAbsolute()}),exit})}
 $('save').onclick=()=>saveCurrent(false);$('save-exit').onclick=()=>saveCurrent(true);
$('discard').onclick=()=>{let r=currentRecord(),reason=window.prompt('Optional discard reason:','');if(reason!==null)request('./api/resolve',{id:r.id,revision:n(r.revision),status:'discarded',reason:reason||null})};$('retry').onclick=()=>{if(canLeave())request('./api/retry')};
for(let direction of ['previous','next','current'])$(direction).onclick=()=>navigate(direction);
$('reload').onclick=()=>{if(canLeave())refresh()};
let canvas=$('board');canvas.addEventListener('pointerdown',event=>{if(event.button!==0)return;if(isActionMode()){let name=actionCardAt(canvas,currentRecord()&&currentRecord().observation,event);if(name){event.preventDefault();toggleAction(name,app.mode)}return}canvas.setPointerCapture(event.pointerId);app.lastTile=null;paint(event)});canvas.addEventListener('pointermove',paint);canvas.addEventListener('pointerup',event=>{app.lastTile=null;if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId)});canvas.addEventListener('pointercancel',event=>{app.lastTile=null;if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId)});window.addEventListener('resize',()=>{if(app.state)render()});window.addEventListener('beforeunload',event=>{if(app.dirty&&!app.exiting){event.preventDefault();event.returnValue=''}});refresh();
</script>"""
)


class AnnotationSession:
    def __init__(
        self, cfg, output, *, seed=None, family="preset", max_steps=None, max_attempts=100, kind="tile"
    ):
        levels = tuple(cfg["evaluation"]["levels"])
        self.levels = tuple(level for level in DIFFICULTIES if level in levels)
        if not self.levels:
            raise ValueError("configuration has no supported annotation difficulties")
        if kind not in ("tile", "action"):
            raise ValueError("annotation kind must be tile or action")
        self.kind = kind
        self.store = LiveAnnotationStore(
            output,
            plant_types=cfg["environment"]["plants"],
            config_digest=digest(cfg),
            kind=kind,
        )
        self.sampler = ReachableSnapshotSampler(
            cfg, seed=seed, family=family, max_steps=max_steps, max_attempts=max_attempts
        )
        pending = self.store.pending
        pending_level = pending.get("difficulty") if pending else None
        self.difficulty = pending_level or self.store.last_difficulty
        if self.difficulty not in self.levels:
            self.difficulty = None
        self.error = None
        self.conflicts_seen = 0
        if pending is None and not self.store.conflict_groups() and self.difficulty is not None:
            self._generate()
        self.exit_requested = False

    def _generate(self):
        if (
            self.store.pending is not None
            or self.store.conflict_groups()
            or self.difficulty is None
        ):
            return
        try:
            self.store.add_pending(
                self.sampler.sample(self.difficulty, annotation_kind=self.kind),
                difficulty=self.difficulty,
                sampler_seed=self.sampler.seed,
            )
            self.error = None
        except RuntimeError as exc:
            self.error = str(exc)

    def state(self, request=0, *, record_id=None):
        conflicts = self.store.conflict_groups()
        if conflicts:
            group = conflicts[0]
            record = group["record"]
            index = self.store.index(record["id"])
            return {
                "request": int(request),
                "mode": "conflict",
                "annotation_kind": self.kind,
                "difficulties": list(self.levels),
                "difficulty": self.difficulty,
                "current_difficulty": record.get("difficulty"),
                "record": record,
                "conflict": group,
                "conflicts_index": self.conflicts_seen + 1,
                "conflicts_remaining": len(conflicts),
                "navigation": {
                    "index": index,
                    "total": self.store.total,
                    "previous_id": None,
                    "next_id": None,
                    "pending_id": self.store.pending["id"] if self.store.pending else None,
                    "reviewing": False,
                },
                "counts": self.store.counts,
                "error": self.error,
            }
        record = self.store.record(record_id) if record_id is not None else self.store.pending
        if record is None and self.store.records:
            record = self.store.records[-1]
        index = self.store.index(record["id"]) if record else -1
        return {
            "request": int(request),
            "mode": "pending"
            if record and record["status"] == "pending"
            else "review"
            if record
            else "idle",
            "difficulties": list(self.levels),
            "difficulty": self.difficulty,
            "current_difficulty": record.get("difficulty") if record else None,
            "record": record,
            "conflict": None,
            "conflicts_index": 0,
            "conflicts_remaining": 0,
            "navigation": {
                "index": index,
                "total": self.store.total,
                "previous_id": self.store.records[index - 1]["id"] if index > 0 else None,
                "next_id": (
                    self.store.records[index + 1]["id"]
                    if 0 <= index < self.store.total - 1
                    else None
                ),
                "pending_id": self.store.pending["id"] if self.store.pending else None,
                "reviewing": record is not None and record["status"] != "pending",
            },
            "counts": self.store.counts,
            "error": self.error,
            "annotation_kind": self.kind,
        }

    def choose_difficulty(self, difficulty):
        if self.store.conflict_groups():
            raise ValueError("resolve duplicate annotation conflicts before changing difficulty")
        if difficulty not in self.levels:
            raise ValueError(f"unknown annotation difficulty: {difficulty}")
        self.difficulty = difficulty
        self.store.set_difficulty(difficulty)
        self.error = None
        self._generate()

    def retry(self):
        if self.store.conflict_groups():
            raise ValueError("resolve duplicate annotation conflicts before retrying generation")
        self.error = None
        self._generate()

    def navigate(self, body):
        if self.store.conflict_groups():
            raise ValueError("resolve duplicate annotation conflicts before browsing history")
        direction = body.get("direction")
        if direction not in ("previous", "next", "current"):
            raise ValueError("unknown annotation navigation direction")
        navigation = self.state(record_id=body.get("id"))["navigation"]
        if direction == "current":
            record_id = navigation["pending_id"]
        else:
            record_id = navigation[f"{direction}_id"]
        if record_id is None:
            raise ValueError("no annotation map in that direction")
        return record_id

    def resolve(self, body):
        if self.store.conflict_groups():
            raise ValueError("resolve the displayed duplicate annotation conflict first")
        if not isinstance(body.get("id"), str) or body["id"] not in self.store.by_id:
            raise ValueError("stale annotation record")
        record = self.store.record(body["id"])
        status = body.get("status")
        if self.kind == "action":
            if status == "labeled" and not isinstance(body.get("feasible_actions", []), list):
                raise ValueError("labeled actions must be a list")
            feasible = body.get("feasible_actions", [])
            absolute = body.get("absolute_infeasible_actions", [])
            tiles = []
        else:
            if status == "labeled" and not isinstance(body.get("tiles", []), list):
                raise ValueError("labeled tiles must be a list")
            feasible = absolute = None
            tiles = body.get("tiles", [])
            absolute = body.get("absolute_infeasible_tiles")
        reviewing = record["status"] != "pending"
        if reviewing and "revision" not in body:
            raise ValueError("historical edits require an annotation revision")
        save = self.store.revise if reviewing else self.store.resolve
        save(
            record["id"],
            status=status,
            tiles=tiles,
            feasible=feasible,
            absolute=absolute,
            reason=body.get("reason"),
            revision=body.get("revision", 0),
        )
        if reviewing:
            if body.get("exit"):
                self.exit_requested = True
            return record["id"]
        self.error = None
        if body.get("exit"):
            self.exit_requested = True
        else:
            self._generate()
        return None

    def resolve_conflict(self, body):
        if not self.store.conflict_groups():
            raise ValueError("no duplicate annotation conflict is pending")
        key = body.get("key")
        if self.kind == "action":
            self.store.revise_conflict(key, body.get("members"), feasible=body.get("feasible_actions", []), absolute=body.get("absolute_infeasible_actions", []))
        else:
            self.store.revise_conflict(key, body.get("members"), body.get("tiles"), absolute=body.get("absolute_infeasible_tiles"))
        self.conflicts_seen += 1
        if body.get("exit"):
            self.exit_requested = True
        else:
            self.error = None
            self._generate()

    def exit(self):
        self.exit_requested = True


class _AnnotationHandler(BaseHTTPRequestHandler):
    server_version = "pvz-tile-annotator/2"

    def _reply(self, status, body, content_type="application/json"):
        raw = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length))

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.rstrip("/") in ("", "/index.html"):
            return self._reply(200, ANNOTATION_HTML, "text/html")
        if parsed.path.rstrip("/") == "/api/state":
            try:
                request = int(parse_qs(parsed.query).get("request", ["0"])[0])
                record_id = parse_qs(parsed.query).get("id", [None])[0]
                with self.server.lock:
                    result = self.server.session.state(request, record_id=record_id)
                return self._reply(200, json.dumps(result))
            except (TypeError, ValueError, KeyError) as exc:
                return self._reply(400, json.dumps({"error": f"invalid state request: {exc}"}))
        return self._reply(404, json.dumps({"error": "not found"}))

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path.rstrip("/")
        if path not in (
            "/api/difficulty",
            "/api/resolve",
            "/api/conflict/resolve",
            "/api/retry",
            "/api/navigate",
            "/api/exit",
        ):
            return self._reply(404, json.dumps({"error": "not found"}))
        try:
            body = self._body()
            if not isinstance(body, dict):
                raise ValueError("request must be an object")
            request = int(body.get("request", 0))
            with self.server.lock:
                record_id = None
                if path == "/api/difficulty":
                    record_id = body.get("id")
                    if record_id is not None:
                        self.server.session.store.record(record_id)
                    self.server.session.choose_difficulty(body.get("difficulty"))
                elif path == "/api/conflict/resolve":
                    self.server.session.resolve_conflict(body)
                elif path == "/api/resolve":
                    record_id = self.server.session.resolve(body)
                elif path == "/api/navigate":
                    record_id = self.server.session.navigate(body)
                elif path == "/api/exit":
                    self.server.session.exit()
                else:
                    self.server.session.retry()
                result = self.server.session.state(request, record_id=record_id)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            return self._reply(400, json.dumps({"error": str(exc)}))
        response = self._reply(200, json.dumps(result))
        if self.server.session.exit_requested:
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        return response

    def log_message(self, *_args):
        return


class _AnnotationServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, session):
        super().__init__(address, _AnnotationHandler)
        self.session = session
        self.lock = threading.RLock()


def serve_annotations(
    output=None,
    *,
    cfg=None,
    port=0,
    seed=None,
    family="preset",
    max_steps=None,
    max_attempts=100,
    kind="tile",
):
    """Serve the one-map-at-a-time annotator until interrupted."""
    if output is None:
        output = "data/action-annotations.jsonl" if kind == "action" else "data/tile-annotations.jsonl"
    if cfg is None:
        from pvz_rl.config import load_config

        cfg = load_config()
    session = AnnotationSession(
        cfg,
        output,
        seed=seed,
        family=family,
        max_steps=max_steps,
        max_attempts=max_attempts,
        kind=kind,
    )
    server = _AnnotationServer(("127.0.0.1", int(port)), session)
    print(f"Annotation studio: http://127.0.0.1:{server.server_port}/", flush=True)
    print(f"Annotation studio proxy: ./proxy/{server.server_port}/", flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return server.server_port
