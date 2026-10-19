"""Snapshot, discard, browser and unordered tile-label controls."""

import copy
import dataclasses
import json
import random
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import torch
from pvz_game import Place, Status, Wait

from pvz_rl.annotation import sampler as sampler_module
from pvz_rl.annotation.dataset import (
    ANNOTATION_PROTOCOL,
    AnnotationStore,
    LiveAnnotationStore,
    validate_record,
)
from pvz_rl.annotation.sampler import ReachableSnapshotSampler, generate_snapshots
from pvz_rl.config import load_config
from pvz_rl.learning.annotation_fit import fit_annotations, set_ranking_loss
from pvz_rl.learning.checkpoints import inspect_checkpoint
from pvz_rl.learning.training import initial_weights
from pvz_rl.presentation.annotation_viewer import (
    ANNOTATION_HTML,
    AnnotationSession,
    _AnnotationServer,
)


def observation():
    return {
        "api_version": 1,
        "tick": 3,
        "elapsed_seconds": 0.3,
        "tick_rate": 10,
        "units_per_tile": 80,
        "rows": 5,
        "cols": 9,
        "level": "easy",
        "status": "running",
        "sun": 100,
        "cards": [
            {"plant_type": "peashooter", "cost": 100, "cooldown_ticks": 0, "recharge_ticks": 750}
        ],
        "plants": [],
        "zombies": [],
        "projectiles": [],
        "mowers": [{"row": row, "x": 0, "state": "ready"} for row in range(5)],
        "wave": 0,
        "total_waves": 1,
        "counts": {
            "initial_total": 1,
            "spawned": 0,
            "defeated": 0,
            "alive": 1,
            "not_yet_spawned": 1,
            "remaining": 1,
        },
    }


def record(identifier="one"):
    return {
        "protocol": "pvz-rl/tile-annotation-snapshot-v1",
        "id": identifier,
        "group_id": "episode-1",
        "plant_type": "peashooter",
        "observation": observation(),
        "encoded_observation": {
            "entities": [[16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            "globals": [0.0] * 18,
        },
        "availability": {
            "available": True,
            "cost": 100,
            "sun": 100,
            "cooldown_ticks": 0,
            "legal_tiles": [0, 1, 44],
        },
    }


def test_annotation_store_persists_label_and_discard(tmp_path):
    snapshots = tmp_path / "snapshots.jsonl"
    snapshots.write_text(json.dumps(record()) + "\n" + json.dumps(record("two")) + "\n", "utf-8")
    output = tmp_path / "annotations.jsonl"
    store = AnnotationStore(snapshots, output, plant_types=("peashooter",))
    store.commit(0, [44, 0, 44])
    store.discard(1, "meaningless state")
    reloaded = AnnotationStore(snapshots, output, plant_types=("peashooter",))
    assert reloaded.record(0)["feasible_tiles"] == [0, 44]
    assert reloaded.record(1)["status"] == "discarded"
    assert reloaded.pending_index() is None
    manifest = json.loads((tmp_path / "annotations.jsonl.manifest.json").read_text("utf-8"))
    assert manifest["labeled"] == 1 and manifest["discarded"] == 1 and manifest["pending"] == 0
    assert manifest["source_digest"] is None  # hand-built snapshots have no source manifest


def test_annotation_store_rejects_illegal_label(tmp_path):
    snapshots = tmp_path / "snapshots.jsonl"
    snapshots.write_text(json.dumps(record()) + "\n", "utf-8")
    store = AnnotationStore(snapshots, tmp_path / "annotations.jsonl", plant_types=("peashooter",))
    with pytest.raises(ValueError, match="not legal"):
        store.commit(0, [2])


def test_browser_annotation_has_drag_discard_and_relative_protocol():
    for token in (
        "pointerdown",
        "pointermove",
        "setPointerCapture",
        "tileIndexAt",
        "Save and next",
        "Save and exit",
        "Duplicate annotation conflict",
        "./api/conflict/resolve",
        "Discard state",
        "./api/difficulty",
        "./api/resolve",
        "./api/navigate",
        "Return to current",
        "Save changes",
        "Reload map",
        "drawTileOverlay",
        "actionCardAt",
        "drawActionCardOverlay",
        "toggleAction",
        "availability",
        "select id=plant",
        "Raw current record JSON",
    ):
        assert token in ANNOTATION_HTML
    assert "fetch('/api/" not in ANNOTATION_HTML


def test_cli_uses_short_annotation_command():
    from pvz_rl.cli import main

    with pytest.raises(SystemExit) as accepted:
        main(["annotate", "--help"])
    assert accepted.value.code == 0
    with pytest.raises(SystemExit) as removed:
        main(["annotate-tiles", "--help"])
    assert removed.value.code == 2


@pytest.fixture
def history_session(tmp_path):
    session = AnnotationSession(load_config(), tmp_path / "annotations.jsonl", max_steps=1)
    session.sampler.sample = Mock(side_effect=[record(name) for name in ("one", "two", "three")])
    session.choose_difficulty("easy")
    session.resolve({"id": "one", "status": "labeled", "tiles": [0]})
    session.resolve({"id": "two", "status": "discarded", "reason": "too sparse"})
    return session


def test_history_navigation_and_edit_preserve_pending_map(history_session):
    session = history_session
    pending = copy.deepcopy(session.store.pending)
    original = copy.deepcopy(session.store.record("one"))
    assert session.navigate({"id": "three", "direction": "previous"}) == "two"
    assert session.navigate({"id": "two", "direction": "previous"}) == "one"
    assert session.navigate({"id": "one", "direction": "next"}) == "two"
    assert session.navigate({"id": "one", "direction": "current"}) == "three"
    view = session.state(request=7, record_id="one")
    assert view["request"] == 7 and view["navigation"]["reviewing"]
    assert view["navigation"]["index"] == 0
    assert view["navigation"]["previous_id"] is None
    assert (
        session.resolve({"id": "one", "revision": 1, "status": "labeled", "tiles": [44, 1, 44]})
        == "one"
    )
    edited = session.store.record("one")
    assert edited["feasible_tiles"] == [1, 44] and edited["revision"] == 2
    for key in original.keys() - {"feasible_tiles", "revision"}:
        assert edited[key] == original[key]
    assert session.store.pending == pending
    assert session.store.total == 3 and session.sampler.sample.call_count == 3
    reloaded = LiveAnnotationStore(session.store.output_path)
    assert reloaded.record("one") == edited and reloaded.pending == pending
    assert reloaded.manifest["annotation_digest"] == session.store.manifest["annotation_digest"]
    assert reloaded.counts == {"generated": 3, "labeled": 1, "discarded": 1, "pending": 1}


def test_history_can_restore_discarded_maps_and_discard_or_clear_labels(history_session):
    session = history_session
    digest_before = session.store.manifest["annotation_digest"]
    session.resolve({"id": "two", "revision": 1, "status": "labeled", "tiles": [44]})
    assert session.store.record("two")["discard_reason"] is None
    assert session.store.counts["labeled"] == 2
    conflict = session.store.conflict_groups()[0]
    session.resolve_conflict(
        {
            "key": conflict["key"],
            "members": [
                {"id": member["id"], "revision": member["revision"]}
                for member in conflict["members"]
            ],
            "tiles": [44],
        }
    )
    session.resolve({"id": "one", "revision": 2, "status": "discarded", "reason": "rechecked"})
    assert session.store.record("one")["feasible_tiles"] == []
    session.resolve({"id": "two", "revision": 3, "status": "labeled", "tiles": []})
    assert session.store.record("two")["feasible_tiles"] == []
    assert session.store.manifest["annotation_digest"] != digest_before
    resumed = AnnotationSession(load_config(), session.store.output_path, max_steps=1)
    assert resumed.state()["record"]["id"] == "three"
    assert resumed.state(record_id="one")["record"]["discard_reason"] == "rechecked"
    assert resumed.store.counts == {"generated": 3, "labeled": 1, "discarded": 1, "pending": 1}


@pytest.mark.parametrize(
    "changes",
    [
        {"revision": 0},
        {"revision": True},
        {"tiles": [2]},
        {"tiles": [45]},
        {"tiles": [True]},
        {"tiles": [[0]]},
        {"status": "pending"},
        {"id": "missing"},
    ],
)
def test_invalid_historical_edits_leave_storage_unchanged(history_session, changes):
    session = history_session
    before = session.store.output_path.read_bytes(), session.store.manifest_path.read_bytes()
    records = copy.deepcopy(session.store.records)
    body = {"id": "one", "revision": 1, "status": "labeled", "tiles": [1]}
    with pytest.raises(ValueError):
        session.resolve({**body, **changes})
    assert before == (
        session.store.output_path.read_bytes(),
        session.store.manifest_path.read_bytes(),
    )
    assert session.store.records == records


def test_history_rejects_missing_revision_and_out_of_range_navigation(history_session):
    session = history_session
    with pytest.raises(ValueError, match="revision"):
        session.resolve({"id": "one", "status": "labeled", "tiles": [1]})
    for identifier, direction in (("one", "previous"), ("three", "next"), ("one", "unknown")):
        with pytest.raises(ValueError):
            session.navigate({"id": identifier, "direction": direction})
    with pytest.raises(ValueError, match="unknown annotation record"):
        session.state(record_id="missing")


def test_history_remains_editable_after_generation_failure(history_session):
    session = history_session
    session.sampler.sample = Mock(side_effect=RuntimeError("generation attempt limit reached"))
    session.resolve({"id": "three", "status": "labeled", "tiles": [0]})
    assert session.store.pending is None and session.state()["error"]
    assert session.navigate({"id": "three", "direction": "previous"}) == "two"
    session.resolve({"id": "one", "revision": 1, "status": "labeled", "tiles": [44]})
    assert session.state(record_id="one")["record"]["feasible_tiles"] == [44]
    assert session.store.pending is None and session.sampler.sample.call_count == 1
    assert session.state()["error"] == "generation attempt limit reached"


def _live_annotation_row(identifier, status, tiles):
    row = record(identifier)
    row.update(
        protocol=ANNOTATION_PROTOCOL, status=status, feasible_tiles=list(tiles), discard_reason=None
    )
    if status == "pending":
        row["difficulty"] = "easy"
    return row


def test_conflict_groups_are_deterministic_and_exclude_compatible_or_inactive_rows(tmp_path):
    output = tmp_path / "annotations.jsonl"
    rows = [
        _live_annotation_row("a", "labeled", [0]),
        _live_annotation_row("b", "labeled", [1]),
        _live_annotation_row("c", "labeled", [44]),
        _live_annotation_row("shared-a", "labeled", [0]),
        _live_annotation_row("shared-b", "labeled", [0, 1]),
        _live_annotation_row("pending", "pending", []),
        _live_annotation_row("discarded", "discarded", []),
    ]
    rows[3]["encoded_observation"]["globals"][0] = 1.0
    rows[4]["encoded_observation"]["globals"][0] = 1.0
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", "utf-8")
    store = LiveAnnotationStore(output, plant_types=("peashooter",))
    groups = store.conflict_groups()
    assert [group["record_ids"] for group in groups] == [["a", "b", "c"]]
    assert groups[0]["members"][0]["feasible_tiles"] == [0]


def test_conflict_revision_is_atomic_and_applies_one_label_set_to_all_members(tmp_path):
    output = tmp_path / "annotations.jsonl"
    rows = [_live_annotation_row("a", "labeled", [0]), _live_annotation_row("b", "labeled", [1])]
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", "utf-8")
    store = LiveAnnotationStore(output, plant_types=("peashooter",))
    group = store.conflict_groups()[0]
    members = [{"id": item["id"], "revision": item["revision"]} for item in group["members"]]
    store.revise_conflict(group["key"], members, [44, 44, 0])
    assert store.conflict_groups() == []
    assert [store.record(identifier)["feasible_tiles"] for identifier in ("a", "b")] == [
        [0, 44],
        [0, 44],
    ]
    assert [store.record(identifier)["revision"] for identifier in ("a", "b")] == [1, 1]
    before = output.read_bytes(), store.manifest_path.read_bytes()
    with pytest.raises(ValueError, match="stale annotation conflict"):
        store.revise_conflict(group["key"], members, [0])
    assert before == (output.read_bytes(), store.manifest_path.read_bytes())


def test_conflict_session_starts_with_queue_and_advances_to_pending(tmp_path):
    output = tmp_path / "annotations.jsonl"
    rows = [_live_annotation_row("a", "labeled", [0]), _live_annotation_row("b", "labeled", [1])]
    pending = _live_annotation_row("pending", "pending", [])
    rows.append(pending)
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", "utf-8")
    session = AnnotationSession(load_config(), output, max_steps=1)
    session.sampler.sample = Mock(side_effect=AssertionError("conflict review sampled a map"))
    state = session.state()
    assert state["mode"] == "conflict" and state["record"]["id"] == "a"
    conflict = state["conflict"]
    session.resolve_conflict(
        {
            "key": conflict["key"],
            "members": [
                {"id": member["id"], "revision": member["revision"]}
                for member in conflict["members"]
            ],
            "tiles": [0],
        }
    )
    state = session.state()
    assert state["mode"] == "pending" and state["record"]["id"] == "pending"


def test_save_and_exit_persists_current_record_without_generating_next(history_session):
    session = history_session
    pending = session.store.pending
    calls = session.sampler.sample.call_count
    session.resolve(
        {
            "id": pending["id"],
            "revision": pending.get("revision", 0),
            "status": "labeled",
            "tiles": [0],
            "exit": True,
        }
    )
    assert session.exit_requested
    assert session.store.pending is None
    assert session.sampler.sample.call_count == calls
    assert session.store.record(pending["id"])["status"] == "labeled"


def test_legacy_completed_record_without_revision_can_be_edited(tmp_path):
    output = tmp_path / "annotations.jsonl"
    old = record()
    old.update(protocol=ANNOTATION_PROTOCOL, status="labeled", feasible_tiles=[0])
    output.write_text(json.dumps(old) + "\n", "utf-8")
    session = AnnotationSession(load_config(), output, max_steps=1)
    assert session.state()["record"]["id"] == "one"
    session.resolve({"id": "one", "revision": 0, "status": "labeled", "tiles": [1]})
    reloaded = LiveAnnotationStore(output)
    assert reloaded.record("one")["revision"] == 1
    assert reloaded.record("one")["feasible_tiles"] == [1] and reloaded.total == 1


def test_history_http_protocol_and_stale_save(history_session):
    server = _AnnotationServer(("127.0.0.1", 0), history_session)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/"

    def post(path, **body):
        with urlopen(
            Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}),
            timeout=5,
        ) as response:
            return json.load(response)

    try:
        with urlopen(base + "api/state?id=one&request=10", timeout=5) as response:
            assert json.load(response)["record"]["feasible_tiles"] == [0]
        previous = post("api/navigate", id="three", direction="previous", request=11)
        assert previous["request"] == 11 and previous["record"]["id"] == "two"
        changed = post("api/difficulty", id="one", difficulty="hard", request=12)
        assert changed["record"]["id"] == "one" and changed["difficulty"] == "hard"
        saved = post("api/resolve", id="one", revision=1, status="labeled", tiles=[44], request=13)
        assert saved["record"]["id"] == "one" and saved["record"]["revision"] == 2
        with pytest.raises(HTTPError) as failure:
            post("api/resolve", id="one", revision=1, status="labeled", tiles=[1], request=14)
        assert failure.value.code == 400
        assert "stale annotation revision" in json.load(failure.value)["error"]
        active = post("api/navigate", id="one", direction="current", request=15)
        assert active["record"]["id"] == "three" and active["record"]["difficulty"] == "easy"
        assert history_session.sampler.sample.call_count == 3
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_conflict_http_resolution_and_save_exit_shuts_down_server(tmp_path):
    output = tmp_path / "annotations.jsonl"
    rows = [_live_annotation_row("a", "labeled", [0]), _live_annotation_row("b", "labeled", [1])]
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", "utf-8")
    session = AnnotationSession(load_config(), output)
    server = _AnnotationServer(("127.0.0.1", 0), session)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/"
    try:
        conflict = session.state()["conflict"]
        body = {
            "key": conflict["key"],
            "members": [
                {"id": member["id"], "revision": member["revision"]}
                for member in conflict["members"]
            ],
            "tiles": [0],
            "exit": True,
            "request": 1,
        }
        with urlopen(
            Request(
                base + "api/conflict/resolve",
                json.dumps(body).encode(),
                {"Content-Type": "application/json"},
            ),
            timeout=5,
        ) as response:
            assert json.load(response)["request"] == 1
        thread.join(timeout=5)
        assert not thread.is_alive()
        reloaded = LiveAnnotationStore(output)
        assert reloaded.conflict_groups() == []
        assert reloaded.record("a")["feasible_tiles"] == [0]
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_annotation_browser_preserves_drafts_and_ignores_stale_replies():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = ANNOTATION_HTML.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    harness = r"""
const vm=require('node:vm'),assert=require('node:assert/strict');
let source=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
let elements=new Map(),context={console,Set,URLSearchParams,encodeURIComponent};
function element(){return {value:'',textContent:'',innerHTML:'',hidden:false,disabled:false,appendChild(){},
 classList:{toggle(){}},addEventListener(){},getContext(){return null},hasPointerCapture(){return false}}}
context.document={getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id)},createElement:element};
context.window={addEventListener(){},confirm(){return false}};
// Avoid opening a network connection: exercise the browser state machine directly.
context.fetch=()=>new Promise(()=>{});vm.createContext(context);vm.runInContext(source,context);
vm.runInContext(`
 drawBoard=()=>{};
 app.request=10;app.busy=false;app.state={record:{id:'one',revision:1,feasible_tiles:[0]}};
 app.draft=new Set([0,44]);app.dirty=true;
 applyState({request:10,record:{id:'one',revision:1,feasible_tiles:[0]}},10,true);
 if(!app.draft.has(44)||!app.dirty)throw Error('difficulty change lost draft');
 app.request=12;
 applyState({request:11,record:{id:'other',revision:1,feasible_tiles:[1]}},11,false);
 if(currentRecord().id!=='one'||!app.draft.has(44))throw Error('stale reply replaced board');
 if(canLeave()!==false)throw Error('unsaved navigation not guarded');
 applyState({request:12,record:{id:'one',revision:2,feasible_tiles:[1]}},12,false);
 if(app.dirty||selected().join(',')!=='1')throw Error('save did not refresh draft');
 app.state.navigation={reviewing:true,index:0,total:3};render();
 if($('save').textContent!=='Save changes')throw Error('history save mode missing');
 app.state.record={id:'two',revision:1,status:'discarded',feasible_tiles:[]};
 applyState({request:12,record:{id:'two',revision:1,status:'discarded',feasible_tiles:[]}},12,false);
 if(app.draft.size!==0)throw Error('discarded map did not start empty');
 app.state={annotation_kind:'action',record:{id:'action',annotation_kind:'action',action_availability:{sunflower:{},wait:{},dig:{}},automatic_absolute_infeasible_actions:['wait']}};
 app.draft=new Set();app.absolute=new Set();app.dirty=false;render=()=>{};
 let cardCanvas={getBoundingClientRect(){return {left:0,top:0,width:1280,height:820}}};
 let cardObs={cards:[{plant_type:'sunflower'}]};
 if(actionCardAt(cardCanvas,cardObs,{clientX:40,clientY:90})!=='sunflower')throw Error('plant card hit test failed');
 if(actionCardAt(cardCanvas,cardObs,{clientX:250,clientY:90})!==null)throw Error('card hit escaped plant bounds');
 toggleAction('sunflower','add');
 if(!app.draft.has('sunflower')||app.absolute.has('sunflower'))throw Error('feasible card selection failed');
 toggleAction('sunflower','absolute');
 if(app.draft.has('sunflower')||!app.absolute.has('sunflower'))throw Error('absolute card selection failed');
 toggleAction('sunflower','erase');
 if(app.draft.has('sunflower')||app.absolute.has('sunflower'))throw Error('card erase failed');
 toggleAction('wait','add');
 if(app.draft.has('wait'))throw Error('automatic absolute action was editable');
 app.draft=new Set(['wait']);app.absolute=new Set();app.dirty=false;
 paint({buttons:1,clientX:500,clientY:400});
 if(!app.draft.has('wait')||app.draft.size!==1)throw Error('action canvas triggered tile painting');
 `,context);
"""
    result = subprocess.run(
        [node, "-e", harness], input=json.dumps(script), text=True, capture_output=True, timeout=15
    )
    assert result.returncode == 0, result.stderr


def test_live_session_generates_resolves_and_resumes_pending(tmp_path):
    cfg = load_config()
    output = tmp_path / "data" / "tile-annotations.jsonl"
    session = AnnotationSession(cfg, output, seed=91, family="diagnostic", max_steps=20)
    assert session.state()["record"] is None
    session.choose_difficulty("easy")
    first = session.state()["record"]
    assert first["status"] == "pending"
    assert first["difficulty"] == "easy"
    assert session.store.manifest["sampler_seed"] == 91
    session.choose_difficulty("hard")
    assert session.state()["record"]["difficulty"] == "easy"
    assert session.state()["difficulty"] == "hard"
    session.resolve({"id": first["id"], "status": "discarded", "reason": "not useful"})
    second = session.state()["record"]
    assert second is not None and second["id"] != first["id"]
    assert second["difficulty"] == "hard"
    assert json.loads(output.read_text("utf-8").splitlines()[0])["status"] == "discarded"
    resumed = AnnotationSession(cfg, output, seed=91, family="diagnostic", max_steps=20)
    assert resumed.state()["record"]["id"] == second["id"]
    resumed.resolve(
        {
            "id": second["id"],
            "status": "labeled",
            "tiles": second["availability"]["legal_tiles"][:1],
        }
    )
    assert resumed.state()["record"]["difficulty"] == "hard"
    counts = LiveAnnotationStore(output, plant_types=tuple(cfg["environment"]["plants"])).counts
    assert counts["labeled"] == 1 and counts["discarded"] == 1 and counts["pending"] == 1


def test_annotation_sampler_uses_entropy_by_default_and_explicit_seed_is_reproducible(monkeypatch):
    monkeypatch.setattr(sampler_module.secrets, "randbits", Mock(side_effect=[101, 202]))
    first = ReachableSnapshotSampler(max_steps=1, max_attempts=1)
    second = ReachableSnapshotSampler(max_steps=1, max_attempts=1)
    assert first.seed == 101 and second.seed == 202
    explicit_a = ReachableSnapshotSampler(seed=303, max_steps=1, max_attempts=1)
    explicit_b = ReachableSnapshotSampler(seed=303, max_steps=1, max_attempts=1)
    assert explicit_a.seed == explicit_b.seed == 303


def test_record_validation_rejects_missing_full_observation():
    broken = copy.deepcopy(record())
    broken["observation"].pop("cards")
    with pytest.raises(ValueError, match="required fields"):
        validate_record(broken)


@pytest.mark.parametrize("x", [-1, 0, 760])
def test_snapshot_preserves_zombies_near_house_and_their_health(x):
    row = record()
    row["observation"]["zombies"] = [
        {
            "id": 3,
            "zombie_type": "basic",
            "row": 0,
            "x": x,
            "health": 230,
            "armor": 0,
            "state": "walking",
            "slow_ticks": 0,
            "has_pole": False,
            "timer_ticks": 0,
            "headless": False,
        }
    ]
    validate_record(row)
    assert row["observation"]["zombies"][0]["health"] == 230
    for invalid in (True, -1.5, "-1", float("nan")):
        row["observation"]["zombies"][0]["x"] = invalid
        with pytest.raises(ValueError, match="zombie.x"):
            validate_record(row)


def test_empty_entity_encoding_is_valid_for_empty_boards():
    empty = record()
    empty["encoded_observation"] = {"entities": [], "globals": [0.0] * 18}
    validate_record(empty)


def test_unordered_set_loss_ignores_positive_order_and_duplicates():
    values = torch.arange(45, dtype=torch.float32).reshape(1, 45)
    first = set_ranking_loss(values, [[1, 2, 2, 3]], margin=1.0)
    second = set_ranking_loss(values, [[3, 1, 2]], margin=1.0)
    assert torch.equal(first, second)


def test_sampler_writes_only_usable_plants_and_full_observations(tmp_path):
    path = tmp_path / "snapshots.jsonl"
    generate_snapshots(
        path,
        cfg=load_config(),
        count=2,
        seed=77,
        levels=["easy"],
        family="diagnostic",
        max_steps=20,
    )
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    assert len(rows) == 2
    for row in rows:
        validate_record(row, plant_types=load_config()["environment"]["plants"])
        availability = row["availability"]
        card = next(c for c in row["observation"]["cards"] if c["plant_type"] == row["plant_type"])
        assert availability["available"] and availability["sun"] >= card["cost"]
        assert availability["cooldown_ticks"] == 0 and availability["legal_tiles"]


def test_sampler_diversifies_easy_states_with_delayed_sparse_planting():
    sampler = ReachableSnapshotSampler(load_config(), seed=101)
    assert sampler.max_steps == 20000
    rows = [sampler.sample("easy") for _ in range(4)]
    assert all(0 <= row["source"]["simulation_steps"] <= 20000 for row in rows)
    observations = [row["observation"] for row in rows]
    assert all(o["status"] == "running" and len(o["plants"]) <= 8 for o in observations)
    assert any(o["zombies"] for o in observations)
    assert len({p["plant_type"] for o in observations for p in o["plants"]}) >= 3
    assert len({o["sun"] for o in observations}) > 1
    assert len({tuple(c["cooldown_ticks"] for c in o["cards"]) for o in observations}) > 1


@pytest.mark.parametrize("limit", [1, 20000])
@pytest.mark.parametrize("endpoint", ["zero", "maximum"])
def test_random_simulation_limit_includes_both_boundaries(limit, endpoint):
    rng = Mock()
    rng.randint.return_value = 0 if endpoint == "zero" else limit
    assert sampler_module._warmup_steps(rng, limit) == rng.randint.return_value
    rng.randint.assert_called_once_with(0, limit)


def test_sampler_plant_attempts_follow_step_intervals_and_replay_exactly(monkeypatch):
    from pvz_game.config import resolve_level

    from pvz_rl.annotation.sampler import _public_observation
    from pvz_rl.envs.action_timing import ActionPhaseGame

    sampler = ReachableSnapshotSampler(seed=101, max_attempts=1)
    monkeypatch.setattr(sampler_module, "_warmup_steps", lambda *_: 4000)
    attempts, actions = [], []
    original_action, original_step = sampler_module._rollout_action, sampler.env.step
    original_reset = sampler.env.reset
    scenarios = []

    def reset(*, seed, options):
        # Keep enough legal sun in this replay fixture to test timing without
        # conditioning the result on the separate endpoint-availability rule.
        scenario = dataclasses.replace(
            resolve_level("easy", sampler.env.rules, random.Random(seed)), initial_sun=1000
        )
        scenarios.append(scenario)
        return original_reset(seed=seed, options={**options, "scenario": scenario})

    def attempt(env, rng, **kwargs):
        attempts.append(len(actions))
        return original_action(env, rng, **kwargs)

    def step(encoded):
        action = sampler.env.codec.decode(encoded)
        assert sampler.env.game.validate_action(action).accepted
        actions.append(action)
        return original_step(encoded)

    monkeypatch.setattr(sampler_module, "_rollout_action", attempt)
    monkeypatch.setattr(sampler.env, "step", step)
    monkeypatch.setattr(sampler.env, "reset", reset)
    row = sampler.sample("easy")
    assert len(actions) == row["source"]["simulation_steps"] == 4000
    assert attempts and 500 <= attempts[0] <= 1500
    assert all(500 <= b - a <= 1500 for a, b in zip(attempts, attempts[1:]))
    replay = ActionPhaseGame(sampler.env.rules)
    replay.reset(scenarios[0], row["source"]["seed"])
    for action in actions:
        replay.step(action, ticks=int(isinstance(action, Wait)))
    assert _public_observation(replay.observe()) == row["observation"]
    assert (
        sampler.env.encoder.encode(replay.observe())["entities"].tolist()
        == row["encoded_observation"]["entities"]
    )


def test_sampler_respects_sparse_plant_cap(monkeypatch):
    cfg = load_config()
    cfg["training"]["annotation"].update(
        sampling_plant_interval_steps=[1, 1], sampling_max_plants=1
    )
    sampler = ReachableSnapshotSampler(cfg, max_attempts=1)
    monkeypatch.setattr(sampler_module, "_warmup_steps", lambda *_: 3000)
    original_step = sampler.env.step

    def step(action):
        result = original_step(action)
        assert len(sampler.env.public.plants) <= 1
        return result

    monkeypatch.setattr(sampler.env, "step", step)
    assert len(sampler.sample("easy")["observation"]["plants"]) == 1


@pytest.fixture
def rollout_env():
    kinds = load_config()["environment"]["plants"]
    legal = [Wait(), *(Place(kind, 0, 0) for kind in kinds)]
    legal.extend(Place(kinds[0], 1, col) for col in range(9))
    return SimpleNamespace(
        codec=SimpleNamespace(plants=tuple(kinds), cols=9),
        public=SimpleNamespace(
            sun=100,
            plants=[],
            cards=[SimpleNamespace(plant_type=kind, cost=100, cooldown_ticks=0) for kind in kinds],
        ),
        public_board=lambda: SimpleNamespace(legal=legal),
    )


def test_random_plant_choice_covers_wait_and_species_without_tile_count_bias(rollout_env):
    env = rollout_env
    kinds, legal = env.codec.plants, env.public_board().legal
    rng = random.Random(101)
    actions = [sampler_module._rollout_action(env, rng) for _ in range(900)]
    counts = {
        kind: sum(isinstance(a, Place) and a.plant_type == kind for a in actions) for kind in kinds
    }
    counts[None] = sum(isinstance(a, Wait) for a in actions)
    assert all(60 <= count <= 140 for count in counts.values())
    assert all(action in legal for action in actions)
    env.public_board = lambda: SimpleNamespace(legal=list(reversed(legal)))
    rng = random.Random(101)
    assert [sampler_module._rollout_action(env, rng) for _ in range(900)] == actions
    env.public_board = lambda: SimpleNamespace(legal=[Wait()])
    assert isinstance(sampler_module._rollout_action(env, rng), Wait)


def test_random_plant_choice_draws_each_of_nine_options_before_checking_availability(rollout_env):
    env = rollout_env
    options = (None, *env.codec.plants)
    assert len(options) == 9
    for kind in options:
        rng = Mock()
        rng.choice.side_effect = [kind, 0]
        action = sampler_module._rollout_action(env, rng)
        assert rng.choice.call_args_list[0].args == (options,)
        assert action == (Wait() if kind is None else Place(kind, 0, 0))
        assert rng.choice.call_count == (1 if kind is None else 2)


@pytest.mark.parametrize(
    "blocked", ["cooldown", "insufficient_sun", "absent_card", "no_tiles", "cap"]
)
def test_random_plant_choice_waits_without_redrawing_blocked_species(rollout_env, blocked):
    env = rollout_env
    kind = env.codec.plants[0]
    card = env.public.cards[0]
    max_plants = None
    if blocked == "cooldown":
        card.cooldown_ticks = 1
    elif blocked == "insufficient_sun":
        card.cost = env.public.sun + 1
    elif blocked == "absent_card":
        env.public.cards.remove(card)
    elif blocked == "no_tiles":
        legal = [
            a for a in env.public_board().legal if not isinstance(a, Place) or a.plant_type != kind
        ]
        env.public_board = lambda: SimpleNamespace(legal=legal)
    else:
        max_plants = 1
        env.public.plants = [object()]
    # Other species remain usable: a failed draw must not redirect to them.
    rng = Mock()
    rng.choice.return_value = kind
    assert isinstance(sampler_module._rollout_action(env, rng, max_plants=max_plants), Wait)
    rng.choice.assert_called_once_with((None, *env.codec.plants))


@pytest.mark.parametrize(
    "key,value",
    [
        ("sampling_plant_interval_steps", [1500, 500]),
        ("sampling_plant_interval_steps", [True, 1500]),
        ("sampling_plant_interval_steps", [0, 1500]),
        ("sampling_plant_interval_steps", [500]),
        ("sampling_max_plants", 0),
        ("sampling_max_plants", 45),
        ("sampling_max_plants", True),
    ],
)
def test_annotation_sampling_settings_reject_invalid_ranges(key, value):
    from pvz_rl.config import validate_config

    cfg = load_config()
    cfg["training"]["annotation"][key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(cfg)


@pytest.mark.parametrize("status", ["lost", "won", "truncated"])
def test_sampler_retries_finished_games_including_final_step(monkeypatch, status):
    sampler = ReachableSnapshotSampler(max_steps=1, max_attempts=2)
    targets = iter([1, 0])
    monkeypatch.setattr(sampler_module, "_warmup_steps", lambda *_: next(targets))

    def finish(_action):
        sampler.env.state = status

    monkeypatch.setattr(sampler.env, "step", finish)
    row = sampler.sample("easy")
    assert sampler.episodes == 2
    assert row["observation"]["status"] == "running"
    assert row["source"]["simulation_steps"] == row["source"]["tick"] == 0


def test_sampler_does_not_return_pre_loss_frames_and_bounds_retries(monkeypatch):
    sampler = ReachableSnapshotSampler(max_steps=10, max_attempts=2)
    monkeypatch.setattr(sampler_module, "_warmup_steps", lambda *_: 10)
    steps = []

    def lose(action):
        steps.append(action)
        sampler.env.public = SimpleNamespace(status=Status.LOST)
        sampler.env.state = "lost"

    monkeypatch.setattr(sampler.env, "step", lose)
    with pytest.raises(RuntimeError, match="2 attempts of 10 steps"):
        sampler.sample("easy")
    assert len(steps) == sampler.episodes == 2


def test_sampler_retries_unusable_endpoint_without_extending_limit(monkeypatch):
    sampler = ReachableSnapshotSampler(max_steps=1, max_attempts=2)
    monkeypatch.setattr(sampler_module, "_warmup_steps", lambda *_: 1)
    original_available = sampler_module._available
    count = 0

    def available(env):
        nonlocal count
        count += 1
        return [] if count == 1 else original_available(env)

    monkeypatch.setattr(sampler_module, "_available", available)
    row = sampler.sample("easy")
    assert sampler.episodes == 2
    assert row["source"]["simulation_steps"] == row["source"]["tick"] == 1


def test_annotation_fit_emits_weight_only_initialization(tmp_path):
    snapshots = tmp_path / "snapshots.jsonl"
    generate_snapshots(
        snapshots,
        cfg=load_config(),
        count=1,
        seed=78,
        levels=["easy"],
        family="diagnostic",
        max_steps=10,
    )
    annotations = tmp_path / "annotations.jsonl"
    store = AnnotationStore(
        snapshots,
        annotations,
        plant_types=tuple(load_config()["environment"]["plants"]),
    )
    legal = store.record(0)["availability"]["legal_tiles"]
    store.commit(0, legal)
    output = tmp_path / "tile-init.pt"
    fit_annotations(annotations, output, cfg=load_config(), device="cpu", passes=1, batch_size=1)
    assert inspect_checkpoint(output, weights_only=True)["initialization_type"] == "annotation"
    weights, metadata = initial_weights(output, load_config())
    assert metadata["type"] == "annotation" and weights


def test_top1_hinge_independent_values_gradients_and_legal_ties():
    from pvz_rl.learning.annotation_fit import _metrics

    values = torch.tensor([[2.0, 0.0, 3.0, 100.0], [4.0, 1.0, 7.0, 0.0]], requires_grad=True)
    legal = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)
    # Only the best positive matters; invalid 100 cannot become a negative.
    loss = set_ranking_loss(values, [[1, 0, 1], [2]], legal_masks=legal)
    assert loss.item() == 1
    loss.backward()
    torch.testing.assert_close(values.grad, torch.tensor([[-0.5, 0, 0.5, 0], [0, 0, 0, 0]]))
    tied = torch.tensor([[2.0, 2.0, 99.0]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    assert _metrics(tied, [[1]], mask)["hits"] == 0
    assert _metrics(tied, [[0, 1]], mask)["hits"] == 1
    assert set_ranking_loss(tied, [[1, 0]], legal_masks=mask) == 0
    with pytest.raises(ValueError, match="empty"):
        set_ranking_loss(tied, [[]], legal_masks=mask)


def test_top1_loss_partition_weights_and_unordered_labels():
    values = torch.tensor([[0.0, 1.0, 4.0], [4.0, 2.0, 1.0], [1.0, 0.0, 2.0]], requires_grad=True)
    labels = [[0, 1, 1], [0, 2], [1]]
    whole = set_ranking_loss(values, labels)
    partial = (
        set_ranking_loss(values[:2], labels[:2]) * (2 / 3)
        + set_ranking_loss(values[2:], labels[2:]) / 3
    )
    torch.testing.assert_close(whole, partial)
    torch.testing.assert_close(
        torch.autograd.grad(whole, values, retain_graph=True)[0],
        torch.autograd.grad(partial, values)[0],
    )
    assert whole == set_ranking_loss(values, [[1, 0], [2, 0], [1]])


def test_identical_model_inputs_reject_empty_or_collectively_contradictory_sets():
    from pvz_rl.learning.annotation_fit import _validate_labels

    rows = [record(str(i)) for i in range(3)]
    for row, labels in zip(rows, [[0, 1], [1, 44], [0, 44]], strict=True):
        row["feasible_tiles"] = labels
    with pytest.raises(ValueError, match="contradictory.*0, 1, 2"):
        _validate_labels(rows)
    rows[0]["feasible_tiles"] = []
    with pytest.raises(ValueError, match="empty.*0"):
        _validate_labels(rows)


@pytest.fixture
def fitting_dataset(tmp_path):
    from pvz_rl.learning.annotation_fit import _batch, _load_initial_model

    cfg = load_config()
    sampler = ReachableSnapshotSampler(cfg, max_steps=1, max_attempts=4)
    base = sampler.sample("easy")
    torch.manual_seed(101)
    model = _load_initial_model(cfg, "cpu", None).eval()
    with torch.no_grad():
        values, _ = _batch(model, [{**base, "feasible_tiles": [0]}], cfg, "cpu")
    legal = base["availability"]["legal_tiles"]
    best = max(legal, key=lambda tile: (float(values[0, tile]), -tile))
    chosen = next(tile for tile in reversed(legal) if tile != best)
    records = []
    for index in range(3):
        row = copy.deepcopy(base)
        row.update(
            id=f"fit-{index}",
            group_id="same-game",
            protocol=ANNOTATION_PROTOCOL,
            status="labeled",
            feasible_tiles=[chosen],
        )
        records.append(row)
    path = tmp_path / "fit.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records), "utf-8")
    return path, records, cfg, model.state_dict()


def test_annotation_incomplete_cap_exact_resume_and_frozen_species(
    tmp_path, monkeypatch, fitting_dataset
):
    from pvz_rl.learning import annotation_fit as module

    path, rows, cfg, initial = fitting_dataset
    full_output = tmp_path / "full.pt"
    arguments = dict(cfg=cfg, device="cpu", passes=3, batch_size=2, learning_rate=1e-20)
    full = fit_annotations(path, full_output, **arguments)
    assert full["status"] == "incomplete" and full["passes_completed"] == 3
    assert full["test"] is None and full["checkpoint"] is None and not full_output.exists()
    reference = torch.load(full["progress"], weights_only=True)
    partial_output = tmp_path / "partial.pt"
    original = module._run_split
    attempts = 0

    def interrupted(*args, **kwargs):
        nonlocal attempts
        if kwargs.get("train"):
            attempts += 1
            if attempts == 2:
                raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_run_split", interrupted)
    partial = fit_annotations(path, partial_output, **arguments)
    assert partial["stop_reason"] == "interrupted" and partial["passes_completed"] == 1
    assert not partial_output.exists()
    with pytest.raises(ValueError, match="weight-only checkpoint protocol"):
        inspect_checkpoint(partial["progress"], weights_only=True)
    monkeypatch.setattr(module, "_run_split", original)
    resumed = fit_annotations(path, partial_output, resume_from=partial["progress"], **arguments)
    saved = torch.load(resumed["progress"], weights_only=True)
    assert saved["passes_completed"] == saved["optimizer_updates"] == 3
    for name, value in saved["model"].items():
        torch.testing.assert_close(value, reference["model"][name], rtol=0, atol=0)
        if name.startswith("species_"):
            torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    for key, state in saved["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, reference["optimizer"]["state"][key][name], rtol=0, atol=0
            )
    assert len(Path(resumed["history"]).read_text().splitlines()) == 3
    with pytest.raises(ValueError, match="settings.*mismatch"):
        fit_annotations(
            path, partial_output, resume_from=partial["progress"], **{**arguments, "batch_size": 3}
        )
    rows[0]["group_id"] = "changed-group"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")
    with pytest.raises(ValueError, match="dataset.*mismatch"):
        fit_annotations(path, partial_output, resume_from=partial["progress"], **arguments)


def test_annotation_unlimited_default_and_zero_pass_publication(tmp_path, fitting_dataset):
    path, rows, cfg, _ = fitting_dataset
    assert cfg["training"]["annotation"]["passes"] == 0
    for row in rows:
        row["feasible_tiles"] = row["availability"]["legal_tiles"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")
    output = tmp_path / "qualified.pt"
    result = fit_annotations(path, output, cfg=cfg, device="cpu")
    assert result["status"] == "complete" and result["optimizer_updates"] == 0
    inspected = inspect_checkpoint(output, weights_only=True)
    assert inspected["payload"]["convergence"] == {"verified": True, "hits": 3, "records": 3}
    weights, _ = initial_weights(output, cfg)
    assert weights


def test_annotation_interrupt_before_initial_evaluation_retains_zero_pass(
    tmp_path, monkeypatch, fitting_dataset
):
    from pvz_rl.learning import annotation_fit as module

    path, _, cfg, _ = fitting_dataset

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "_run_split", interrupt)
    result = fit_annotations(path, tmp_path / "interrupted-initial.pt", cfg=cfg, device="cpu")
    assert result["stop_reason"] == "interrupted"
    assert result["passes_completed"] == 0 and result["checkpoint"] is None
    saved = torch.load(result["progress"], weights_only=True)
    assert saved["status"] == "incomplete" and saved["passes_completed"] == 0
    assert saved["optimizer"]["state"] == {} and saved["train"] is None
