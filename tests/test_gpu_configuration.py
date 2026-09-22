"""Hardware selection must not silently reinterpret an archived protocol."""

import argparse
import json

import pytest

from pvz_rl.cli import configured
from pvz_rl.config import load_config, research_config, simulator
from pvz_rl.gpu_benchmark import recommend


def args(**kwargs):
    return argparse.Namespace(**{"command": "train", "config": None, **kwargs})


def test_new_shared_run_defaults_and_explicit_parallelism():
    cfg = configured(args())
    assert simulator(cfg) == "cuda"
    assert cfg["training"]["n_envs"] == 128
    assert cfg["training"]["rollout_size"] == 16384
    for count in (32, 64, 128):
        cfg = configured(args(n_envs=count))
        assert cfg["training"]["rollout_size"] == count * 128


@pytest.mark.parametrize(
    "options", [{"simulator": "cpu"}, {"device": "cpu"}, {"condition": "hybrid"}]
)
def test_cpu_and_hybrid_requests_are_rejected(options):
    with pytest.raises(ValueError, match="removed"):
        configured(args(**options))


def test_explicit_legacy_decision_budget_still_uses_cuda():
    cfg = configured(args(steps=4096))
    assert simulator(cfg) == "cuda"
    assert cfg["training"]["budget_unit"] == "decisions"


def test_hardware_recommendations_and_override_conflicts(tmp_path):
    path = tmp_path / "recommendation.json"
    path.write_text(
        json.dumps(dict(n_envs=64, device="cuda", simulator="cuda", rollout_steps_per_env=128))
    )
    cfg = configured(args(hardware=path))
    assert simulator(cfg) == "cuda" and cfg["training"]["rollout_size"] == 8192
    cfg = configured(args(hardware=path, n_envs=32))
    assert cfg["training"]["rollout_size"] == 4096
    with pytest.raises(ValueError, match="conflict"):
        configured(args(hardware=path, rollout_size=1234))
    path.write_text(json.dumps(dict(n_envs=None, device="cuda")))
    with pytest.raises(ValueError, match="no usable"):
        configured(args(hardware=path))


def test_resume_uses_saved_protocol_and_instrumentation_is_optional(tmp_path):
    saved = load_config()
    (tmp_path / "metadata.json").write_text(json.dumps({"config": saved}))
    cfg = configured(args(resume=tmp_path / "latest.zip"))
    assert research_config(cfg) == research_config(saved)
    cfg["simulation"] = {"backend": "cuda", "profile": True}
    assert research_config(cfg) == research_config(saved)
    cfg["simulation"]["backend"] = "cpu"
    assert research_config(cfg) != research_config(saved)


def test_promotion_requires_complete_stable_trials_and_correctness():
    rows = [
        dict(profile=name, state="complete", decisions_per_second=rate)
        for name, rates in {
            "cuda-32": [150, 151, 152],
            "cuda-64": [155, 156, 157],
            "cuda-128": [100, 300, 500],
        }.items()
        for rate in rates
    ]
    # 32 is within 5% of stable 64; the unstable 128 profile cannot win.
    result = recommend(rows, parity_passed=True, provisional=False)
    assert result["n_envs"] == 32 and result["promote_default"]
    assert not recommend(rows)["promote_default"]
    assert not recommend(rows[:-1], parity_passed=True)["promote_default"]
    assert recommend([])["n_envs"] is None
