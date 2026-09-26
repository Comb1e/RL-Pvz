"""Independent complete-game return, controller and interruption controls."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from pvz_rl.config import load_config
from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.learning.cohort import ActorTransaction, CohortPhase, RoleSchedule
from pvz_rl.learning.cuda_buffer import CompleteGameBuffer
from pvz_rl.learning.cuda_ppo import CudaCompleteGamePPO
from pvz_rl.learning.training import build_model, vector_env
from pvz_rl.policy.event_memory import EventMemory
from pvz_rl.policy.grouped_policy import GroupedDistribution, controller_choice


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("epsilon", [0.0, 0.1, 1.0])
def test_plant_mixture_enumeration_gradients_kl_and_unused_arguments(device, epsilon):
    torch.manual_seed(20)
    logits = torch.randn(4, 368, device=device, dtype=torch.float64, requires_grad=True)
    masks = torch.zeros(4, A.size, device=device, dtype=torch.bool)
    masks[:, 0] = True
    masks[0, 1:10] = True
    masks[0, 46:51] = True
    masks[1, 361:] = True  # no planting; dummy conditionals have zero mass
    masks[2, 92] = True
    masks[3, 1:361] = True
    dist = GroupedDistribution(epsilon).proba_distribution(logits, masks)
    reference = np.zeros((4, 360))
    raw = logits.detach().cpu().numpy()
    legal = masks.cpu().numpy()[:, 1:361].reshape(4, 8, 45)
    for b in range(4):
        species = np.flatnonzero(legal[b].any(-1))
        if not len(species):
            continue
        weights = np.exp(raw[b, species] - raw[b, species].max())
        weights /= weights.sum()
        for k, w in zip(species, weights):
            tiles = np.flatnonzero(legal[b, k])
            values = raw[b, 8 + 45 * k + tiles]
            probability = np.exp(values - values.max())
            probability /= probability.sum()
            reference[b, 45 * k + tiles] = (1 - epsilon) * w * probability + epsilon / len(
                species
            ) / len(tiles)
    np.testing.assert_allclose(dist.probs.detach().cpu(), reference, rtol=1e-12, atol=1e-14)
    positive = reference > 0
    entropy = -np.where(positive, reference * np.log(np.maximum(reference, 1e-300)), 0).sum(-1)
    np.testing.assert_allclose(dist.entropy().detach().cpu(), entropy, atol=1e-12)
    new = GroupedDistribution(epsilon).proba_distribution(
        logits + torch.randn_like(logits) * 0.2, masks
    )
    expected = np.where(
        positive,
        reference
        * (
            np.log(np.maximum(reference, 1e-300))
            - np.log(np.maximum(new.probs.detach().cpu().numpy(), 1e-300))
        ),
        0,
    ).sum(-1)
    np.testing.assert_allclose(dist.kl_divergence(new).detach().cpu(), expected, atol=1e-12)
    actions = torch.tensor([1, 0, 92, 80], device=device)
    logs = dist.log_prob(actions)
    (-logs.sum()).backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1].abs().sum() == 0
    # Mixture's species likelihood depends on the selected species' tile map only.
    assert logits.grad[0, 8 + 45 :].abs().sum() == 0
    if epsilon == 1:
        assert logits.grad.abs().sum() == 0
    with pytest.raises(ValueError, match="legal"):
        GroupedDistribution().proba_distribution(logits.detach(), torch.zeros_like(masks))
    with pytest.raises(ValueError, match="finite"):
        GroupedDistribution().proba_distribution(torch.full_like(logits, float("nan")), masks)


def test_greedy_values_initial_ties_and_legality():
    values = torch.zeros(5, 47)
    values[:, 2:] = -2
    masks = torch.ones(5, 406, dtype=torch.bool)
    values[1, 0] = -0.1
    values[2, 9] = 2
    masks[3, 0] = False
    masks[4, 1:361] = False
    assert controller_choice(values, masks).tolist() == [0, 1, 9, 1, 0]


@pytest.mark.parametrize("ram", [0, 1 << 20])
def test_host_storage_actual_returns_context_causality_and_disk_restore(tmp_path, ram):
    cfg = load_config()
    cfg["training"]["n_envs"] = 2
    from pvz_game import Rules

    memory = EventMemory(cfg, Rules(), 2, "cpu")
    buffer = CompleteGameBuffer(tmp_path / "buffer", 286, 48, 2, ram_bytes=ram, block_rows=4)
    contexts = []
    for step in range(5):
        obs = torch.zeros(2, 286)
        obs[:, 270] = step
        masks = torch.zeros(2, 406, dtype=torch.bool)
        masks[:, 0] = True
        ids = torch.arange(2 * step, 2 * step + 2)
        c = memory.observe(
            obs, masks, torch.zeros(2), torch.full((2,), step == 0), torch.full((2,), step), ids
        )
        contexts.append(c.clone())
        rows = np.zeros(2, dtype=buffer.dtype)
        rows["observation"] = obs
        rows["mask"] = np.packbits(masks.numpy(), axis=-1, bitorder="little")
        rows["tick"] = step
        rows["reset"] = step == 0
        rows["env"] = [0, 1]
        rows["active"] = [True, step < 3]
        rows["reward"] = [step + 1, 10 * (step + 1)]
        rows["baseline"] = 0.5
        rows["ids"] = memory.ids
        rows["counts"] = memory.counts
        rows["starts"] = memory.starts
        buffer.append(rows)
    errors = buffer.finalize()
    data = buffer.take(np.arange(10))
    np.testing.assert_array_equal(data["target"][::2], [15, 14, 12, 9, 5])
    np.testing.assert_array_equal(data["target"][1:6:2], [60, 50, 30])
    assert errors["wait"]["count"] == 8 and errors["plant"]["mse"] is None
    context = buffer.batch([8, 9], "cpu")["context"]
    for name in ("tokens", "valid", "counts", "starts"):
        torch.testing.assert_close(getattr(context, name), getattr(contexts[-1], name))
    state = buffer.save(tmp_path / "saved")
    restored = CompleteGameBuffer.restore(tmp_path / "saved", state, tmp_path / "restore")
    np.testing.assert_array_equal(restored.take(np.arange(10)), data)
    assert restored.ram_used <= ram
    restored.close()
    buffer.close()


def tiny_config():
    c = load_config()
    c["training"].update(n_envs=2, n_epochs=1, batch_size=128, role_phase_games=2)
    c["environment"]["cutoff_seconds"] = 1
    c["visualization"].update(live_enabled=False, enabled=False, demos=False)
    c["training"]["performance"].update(compile_kernels=False, telemetry=False)
    return c


def test_wait_losses_enable_planting_and_separate_optimizer_phases(tmp_path):
    c = tiny_config()
    env = vector_env(c, "masked", 101)
    model = build_model(c, "masked", env, 101)
    model.set_logger(configure(str(tmp_path), []))
    actor = [p.clone() for p in model.policy.actor_parameters()]
    model.learn(1)
    assert model.num_timesteps == 200 and model.training_games == 2
    assert model.prefit_errors["wait"]["mse"] == 4
    assert model.prefit_errors["plant"]["count"] == 0
    assert all(torch.equal(a, b) for a, b in zip(actor, model.policy.actor_parameters()))
    assert model.policy.critic_optimizer.state and not model.policy.optimizer.state
    obs = env.reset()
    actions, _ = model.policy.sample_actions(obs, env.action_masks())
    assert bool(((actions > 0) & (actions < A.dig_start)).all())
    model.learn(1, reset_num_timesteps=False)
    assert model.prefit_errors["plant"]["count"] > 0
    assert model.policy.optimizer.state
    env.close()


class InterruptOnce(BaseCallback):
    def __init__(self, after):
        super().__init__()
        self.after = after

    def _on_step(self):
        return self.n_calls < self.after


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_state_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("interrupt_after", [20, None])
def test_interrupted_collection_resume_matches_uninterrupted_optimizer_and_rng(
    tmp_path, interrupt_after
):
    c = tiny_config()
    e = vector_env(c, "masked", 102)
    full = build_model(c, "masked", e, 102)
    full.learn(1)
    if interrupt_after is None:
        full.learn(1, reset_num_timesteps=False)
    expected = deepcopy(full.policy.state_dict())
    optim = deepcopy(full.policy.critic_optimizer.state_dict())
    hashes = [e.batch.state_hash(i) for i in range(2)]
    e.close()
    e = vector_env(c, "masked", 102)
    partial = build_model(c, "masked", e, 102)
    if interrupt_after is None:
        partial.learn(1)
        assert partial.phase == CohortPhase.IDLE
    else:
        with pytest.raises(KeyboardInterrupt):
            partial.learn(1, callback=InterruptOnce(interrupt_after))
        assert partial.phase == CohortPhase.COLLECT
    partial.save(tmp_path / "interrupted.zip")
    e.close()
    resume_cfg = deepcopy(c)
    resume_cfg["training"]["performance"]["token_cache_gib"] = 0
    e = vector_env(resume_cfg, "masked", 102)
    resumed = CudaCompleteGamePPO.load(tmp_path / "interrupted.zip", env=e, device="cuda")
    resumed.learn(1, reset_num_timesteps=False)
    assert_state_equal(expected, resumed.policy.state_dict())
    assert_state_equal(optim, resumed.policy.critic_optimizer.state_dict())
    assert hashes == [e.batch.state_hash(i) for i in range(2)]
    assert resumed.training_games == (4 if interrupt_after is None else 2)
    assert resumed.cohort_metrics["cache_hits"] == 0
    e.close()


def test_actor_transaction_restores_adam_and_leaves_critic(tmp_path):
    c = tiny_config()
    e = vector_env(c, "masked", 101)
    m = build_model(c, "masked", e, 101)
    obs = e.reset()
    mask = e.action_masks()
    action = torch.tensor([1, 1], device="cuda")
    for _ in range(2):
        loss = -m.policy.get_distribution(obs, mask).log_prob(action).mean()
        m.policy.optimizer.zero_grad()
        loss.backward()
        m.policy.optimizer.step()
    before = deepcopy(m.policy.state_dict())
    state = deepcopy(m.policy.optimizer.state_dict())
    tx = ActorTransaction(m.policy, 0.01)
    loss = -m.policy.get_distribution(obs, mask).log_prob(action).mean()
    m.policy.optimizer.zero_grad()
    loss.backward()
    m.policy.optimizer.step()
    tx.check(0.1)
    assert tx.state["rejected"]
    assert_state_equal(before, m.policy.state_dict())
    assert_state_equal(state, m.policy.optimizer.state_dict())
    e.close()


@pytest.mark.parametrize("phase", [CohortPhase.CRITIC, CohortPhase.ACTOR])
def test_interrupted_optimizer_phase_resumes_exactly(tmp_path, monkeypatch, phase):
    import signal

    cfg = tiny_config()

    def make():
        env = vector_env(cfg, "masked", 103)
        m = build_model(cfg, "masked", env, 103)
        m.roles = RoleSchedule(phase.value, 0, "fixed")
        with torch.no_grad():
            m.policy.value_net.bias[1] = 0.1
        return env, m

    env, full = make()
    full.learn(1)
    expected = deepcopy(full.policy.state_dict())
    actor = deepcopy(full.policy.optimizer.state_dict())
    critic = deepcopy(full.policy.critic_optimizer.state_dict())
    env.close()
    env, partial = make()
    name = "_actor_step" if phase == CohortPhase.ACTOR else "_critic_step"
    original = getattr(partial, name)
    triggered = []

    def interrupt(data):
        original(data)
        if not triggered:
            triggered.append(True)
            signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(partial, name, interrupt)
    with pytest.raises(KeyboardInterrupt):
        partial.learn(1)
    monkeypatch.delattr(partial, name)
    partial.save(tmp_path / "partial.zip")
    env.close()
    env = vector_env(cfg, "masked", 103)
    m = CudaCompleteGamePPO.load(tmp_path / "partial.zip", env=env, device="cuda")
    m.learn(0, reset_num_timesteps=False)
    assert_state_equal(expected, m.policy.state_dict())
    assert_state_equal(actor, m.policy.optimizer.state_dict())
    assert_state_equal(critic, m.policy.critic_optimizer.state_dict())
    env.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_conditional_ppo_loss_gradients_and_adam_against_enumeration(device):
    from types import SimpleNamespace

    torch.manual_seed(47)
    actual = torch.nn.Parameter(torch.randn(4, 368, device=device, dtype=torch.float64) * 0.2)
    reference = torch.nn.Parameter(actual.detach().clone())
    masks = torch.zeros(4, 406, device=device, dtype=torch.bool)
    masks[:, [0, 1, 2, 46, 47]] = True
    actions = torch.tensor([1, 47, 2, 46], device=device)
    advantages = torch.tensor([1.0, -0.3, -1.0, 0.5], device=device, dtype=torch.float64)
    dist = GroupedDistribution(0.1).proba_distribution(actual, masks)
    old_logs = dist.log_prob(actions).detach() + torch.tensor([0.0, 0.5, -0.5, 0.0], device=device)
    policy = SimpleNamespace(
        action_dist=dist,
        exploration_settings={"plant_coef": 0.001, "tile_coef": 0.001},
        optimizer=torch.optim.Adam([actual], lr=1e-4, eps=1e-5),
        actor_parameters=lambda: [actual],
        get_distribution=lambda *args: dist.proba_distribution(actual, masks),
    )
    control_optimizer = torch.optim.Adam([reference], lr=1e-4, eps=1e-5)
    model = SimpleNamespace(
        policy=policy,
        target_kl=0,
        normalize_advantage=False,
        clip_range=0.2,
        max_grad_norm=0.5,
        _stats={"actor_attempted_steps": 0, "actor_retained_steps": 0},
        logger=configure(format_strings=[]),
    )
    data = {
        "observation": None,
        "masks": masks,
        "context": None,
        "action": actions,
        "log_prob": old_logs,
        "advantage": advantages,
    }
    for _ in range(2):
        # Enumerate the four legal planting pairs without the implementation's
        # distribution, masking, conditional-entropy or probability helpers.
        species = reference[:, :2].softmax(-1)
        tiles = torch.stack((reference[:, 8:10], reference[:, 53:55]), 1).softmax(-1)
        joint = 0.9 * species[:, :, None] * tiles + 0.025
        plant_mass = joint.sum(-1)
        tile_mass = joint / plant_mass[:, :, None]
        selected = torch.stack((joint[0, 0, 0], joint[1, 1, 1], joint[2, 0, 1], joint[3, 1, 0]))
        ratio = (selected.log() - old_logs).exp()
        surrogate = torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages)
        normalization = np.log(2)
        plant_entropy = -(plant_mass * plant_mass.log()).sum(-1) / normalization
        tile_entropy = -(tile_mass * tile_mass.log()).sum(-1).mean(-1) / normalization
        loss = -surrogate.mean() - 0.001 * (plant_entropy + tile_entropy).mean()
        control_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([reference], 0.5)
        control_optimizer.step()
        CudaCompleteGamePPO._actor_step(model, data)
        torch.testing.assert_close(actual.grad, reference.grad, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-10)
        assert_state_equal(
            policy.optimizer.state_dict()["param_groups"],
            control_optimizer.state_dict()["param_groups"],
        )
        for key, value in policy.optimizer.state[actual].items():
            torch.testing.assert_close(
                value, control_optimizer.state[reference][key], atol=1e-12, rtol=1e-10
            )


def test_atomic_checkpoint_failure_preserves_previous_archive(tmp_path, monkeypatch):
    from zipfile import ZipFile

    env = vector_env(tiny_config(), "masked", 101)
    model = build_model(tiny_config(), "masked", env, 101)
    path = tmp_path / "checkpoint.zip"
    model.save(path)
    before = path.read_bytes()
    original = ZipFile.writestr

    def fail(self, name, *args, **kwargs):
        if name == "cohort-state.pt":
            raise OSError("controlled disk failure")
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "writestr", fail)
    with pytest.raises(OSError, match="controlled disk failure"):
        model.save(path)
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.zip")) == [path]
    monkeypatch.undo()
    CudaCompleteGamePPO.load(path, device="cuda")
    # Incomplete/corrupt archives cannot silently start a different RNG stream.
    damaged = tmp_path / "damaged.zip"
    with ZipFile(path) as inp, ZipFile(damaged, "w") as out:
        for name in inp.namelist():
            if name != "cohort-state.pt":
                out.writestr(name, inp.read(name))
    with pytest.raises(ValueError, match="runtime state"):
        CudaCompleteGamePPO.load(damaged, device="cuda")
    env.close()


def test_resume_at_game_ceiling_finishes_pending_optimization(tmp_path, monkeypatch):
    import signal

    from pvz_rl.learning.training import load_policy, train

    cfg = tiny_config()
    cfg["training"]["total_games"] = 2
    train(cfg, "masked", 101, tmp_path / "full", validation_limit=1)
    expected, _ = load_policy(tmp_path / "full/final.zip", "cuda")
    original = CudaCompleteGamePPO._critic_step
    triggered = []

    def interrupt(self, data):
        original(self, data)
        if not triggered:
            triggered.append(True)
            signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(CudaCompleteGamePPO, "_critic_step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        train(cfg, "masked", 101, tmp_path / "partial", validation_limit=1)
    monkeypatch.setattr(CudaCompleteGamePPO, "_critic_step", original)
    train(
        cfg,
        "masked",
        101,
        tmp_path / "resumed",
        validation_limit=1,
        resume=tmp_path / "partial/interrupted.zip",
    )
    actual, _ = load_policy(tmp_path / "resumed/final.zip", "cuda")
    assert actual._n_updates == expected._n_updates == 1
    assert actual.training_games == 2
    assert_state_equal(actual.policy.state_dict(), expected.policy.state_dict())
    assert_state_equal(
        actual.policy.critic_optimizer.state_dict(), expected.policy.critic_optimizer.state_dict()
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_balanced_critic_update_matches_independent_group_means(device):
    from types import SimpleNamespace

    actual = torch.nn.Parameter(torch.zeros(6, 47, device=device, dtype=torch.float64))
    reference = torch.nn.Parameter(actual.detach().clone())
    actions = torch.tensor([0, 0, 0, 1, 46, 361], device=device)
    targets = torch.arange(1, 7, device=device, dtype=torch.float64)
    opt = torch.optim.Adam([actual], lr=3e-4, eps=1e-5)
    ref_opt = torch.optim.Adam([reference], lr=3e-4, eps=1e-5)
    model = SimpleNamespace(
        policy=SimpleNamespace(
            predict_values=lambda *_: actual,
            critic_optimizer=opt,
            critic_parameters=lambda: [actual],
        ),
        _buffer=SimpleNamespace(group_counts=np.array([3, 2, 1])),
        device=device,
        batch_size=6,
        max_grad_norm=0.5,
        _stats={"critic_optimizer_steps": 0},
        logger=configure(format_strings=[]),
    )
    for _ in range(2):
        loss = (
            (reference[:3, 0] - targets[:3]).square().mean()
            + (reference[3:5, 1] - targets[3:5]).square().mean()
            + (reference[5, 2] - targets[5]).square()
        ) / 3
        ref_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([reference], 0.5)
        ref_opt.step()
        CudaCompleteGamePPO._critic_step(
            model, dict(observation=None, context=None, action=actions, target=targets)
        )
        torch.testing.assert_close(actual.grad, reference.grad, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
        for key, value in opt.state[actual].items():
            torch.testing.assert_close(value, ref_opt.state[reference][key], atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(actual[:, 3:]) == 0  # Unselected outputs receive no regression.


@pytest.mark.parametrize("limit", [0, -32, 255, 257, 1.5, True])
def test_role_configuration_rejects_incomplete_cohorts(limit):
    from pvz_rl.config import validate_config

    cfg = load_config()
    cfg["training"]["role_phase_games"] = limit
    with pytest.raises(ValueError, match="role_phase_games"):
        validate_config(cfg)


def test_eight_cohort_roles_stage_reset_and_no_double_credit():
    roles = RoleSchedule()
    roles.enter_stage("saving")
    for role in ("critic", "actor", "critic"):
        for cohort in range(8):
            assert roles.role == role and roles.games == 32 * cohort
            roles.enter_stage("saving")
            roles.complete(32, 256)
    roles.enter_stage("easy")
    assert (roles.role, roles.games, roles.stage) == ("critic", 0, "easy")


def test_no_plant_actor_cohort_skips_both_optimizers_but_credits_role():
    cfg = tiny_config()
    env = vector_env(cfg, "masked", 101)
    try:
        model = build_model(cfg, "masked", env, 101)
        model.roles = RoleSchedule("actor", 0, "fixed")
        before = deepcopy(model.policy.state_dict())
        model.learn(1)
        assert_state_equal(before, model.policy.state_dict())
        assert not model.policy.optimizer.state and not model.policy.critic_optimizer.state
        assert model.cohort_metrics["actor_skip_reason"] == "no_planting_samples"
        assert model.roles.role == "critic" and model.roles.games == 0
    finally:
        env.close()


@pytest.mark.parametrize("cache_blocks", [0, 1, 8])
@pytest.mark.parametrize("ram", [0, 1 << 20])
def test_raw_token_cache_prefetch_and_partial_batches_match_independent_control(
    tmp_path, cache_blocks, ram
):
    from pvz_rl.learning.transfers import BatchPrefetch

    buffer = CompleteGameBuffer(tmp_path / "trajectory", 286, 48, 2, block_rows=8, ram_bytes=ram)
    rng = np.random.default_rng(141)
    rows = np.zeros(42, dtype=buffer.dtype)
    rows["observation"] = rng.normal(size=(42, 286))
    rows["previous"] = rng.integers(0, 406, 42)
    rows["tick"] = np.arange(42) * 3001
    rows["reset"][:2] = True
    rows["env"] = np.arange(42) % 2
    rows["active"] = True
    rows["mask"] = 255
    rows["action"] = 1
    rows["reward"] = rng.normal(size=42)
    for i in range(42):
        valid = min(i // 2 + 1, 48)
        rows["ids"][i, :valid] = np.arange(i, i - valid * 2, -2)
        rows["counts"][i, :valid] = 1
        rows["starts"][i, :valid] = rows["tick"][rows["ids"][i, :valid]]
    buffer.append(rows)
    buffer.finalize()
    buffer.configure_cache("cuda", cache_blocks * 8 * 289 * 4 / 1024**3)
    assert buffer.cache.size == min(42, 8 * cache_blocks)
    order = rng.permutation(42)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    prefetch = BatchPrefetch(buffer, order, 11, "cuda")
    try:
        for at, batch in enumerate(prefetch):
            ids = order[at * 11 : (at + 1) * 11]
            expected = np.zeros((len(ids), 48, 289), np.float32)
            for j, row in enumerate(ids):
                for k in range(48):
                    if rows["counts"][row, k]:
                        source = rows[rows["ids"][row, k]]
                        expected[j, k, :286] = source["observation"]
                        expected[j, k, 286:] = [source["previous"], source["reset"], source["tick"]]
            np.testing.assert_array_equal(batch["context"].tokens.cpu(), expected)
            control = buffer.batch(ids, "cpu")
            # Compare authoritative rows without using batch/context reconstruction.
            torch.testing.assert_close(
                batch["target"].cpu(), torch.from_numpy(buffer.take(ids)["target"].copy())
            )
            if control:
                torch.testing.assert_close(batch["context"].tokens.cpu(), control["context"].tokens)
    finally:
        prefetch.close()
        buffer.close()
    assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(
        cuda_rng, torch.cuda.get_rng_state()
    )


def test_token_cache_reserves_two_gib_and_falls_back(tmp_path, monkeypatch):
    from pvz_rl.learning.transfers import VRAM_RESERVE, TokenCache

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (VRAM_RESERVE, 8 * 1024**3))
    cache = TokenCache("cuda", 8, 289, 2 * 1024**3)
    cache.append(torch.zeros(2, 289, device="cuda"))
    assert cache.full and not cache.blocks and cache.size == 0


@pytest.mark.parametrize("role", ["critic", "actor"])
def test_cached_prefetched_update_and_inactive_optimizer_match_host_control(
    tmp_path, role, monkeypatch
):
    from pvz_rl.learning.transfers import BatchPrefetch

    results = []
    for cached in (False, True):
        cfg = tiny_config()
        cfg["training"]["performance"]["token_cache_gib"] = 2 if cached else 0
        env = vector_env(cfg, "masked", 105)
        model = build_model(cfg, "masked", env, 105)
        model.roles = RoleSchedule(role, 0, "fixed")
        with torch.no_grad():
            model.policy.value_net.bias[1] = 0.1
        inactive = model.policy.critic_optimizer if role == "actor" else model.policy.optimizer
        obs = env.reset()
        if role == "actor":
            warm_loss = (model.policy.predict_values(obs)[:, 1] - 1).square().mean()
        else:
            warm_loss = (
                -model.policy.get_distribution(obs, env.action_masks())
                .log_prob(torch.ones(2, device="cuda"))
                .mean()
            )
        inactive.zero_grad()
        warm_loss.backward()
        inactive.step()
        assert inactive.state  # Freeze populated moments, not just an empty Adam.
        before = deepcopy(inactive.state_dict())
        parameters = (
            model.policy.critic_parameters() if role == "actor" else model.policy.actor_parameters()
        )
        unchanged = [p.detach().clone() for p in parameters]
        # Synchronous host preparation is an independent transport control, not
        # a production scheduler or selectable training mode.
        original = BatchPrefetch.__next__
        if not cached:

            def host_next(prefetch):
                data = original(prefetch)
                torch.cuda.synchronize()
                return data

            monkeypatch.setattr(BatchPrefetch, "__next__", host_next)
        model.learn(1)
        assert_state_equal(before, inactive.state_dict())
        parameters = (
            model.policy.critic_parameters() if role == "actor" else model.policy.actor_parameters()
        )
        for a, b in zip(unchanged, parameters):
            assert torch.equal(a, b)
        results.append(
            (
                deepcopy(model.policy.state_dict()),
                deepcopy(model.policy.optimizer.state_dict()),
                deepcopy(model.policy.critic_optimizer.state_dict()),
                torch.cuda.get_rng_state(),
            )
        )
        env.close()
        monkeypatch.undo()
    assert_state_equal(results[0], results[1])


def test_previous_optimizer_protocol_inference_allowed_resume_rejected(tmp_path):
    cfg = tiny_config()
    env = vector_env(cfg, "masked", 101)
    try:
        model = build_model(cfg, "masked", env, 101)
        obs = env.reset()
        mask = env.action_masks()
        expected = model.predict(
            obs.cpu().numpy(), deterministic=True, action_masks=mask.cpu().numpy()
        )[0]
        model.optimizer_protocol = "complete_game_mc_conditional_ppo_v1"
        model.save(tmp_path / "previous.zip")
        loaded = CudaCompleteGamePPO.load(tmp_path / "previous.zip", env=env, device="cuda")
        actual = loaded.predict(
            obs.cpu().numpy(), deterministic=True, action_masks=mask.cpu().numpy()
        )[0]
        np.testing.assert_array_equal(actual, expected)
        with pytest.raises(ValueError, match="optimizer protocol"):
            loaded.learn(1)
    finally:
        env.close()


def test_interruption_at_role_transition_neither_repeats_nor_skips_credit(tmp_path, monkeypatch):
    import signal

    cfg = tiny_config()
    env = vector_env(cfg, "masked", 101)
    model = build_model(cfg, "masked", env, 101)
    original = model.roles.complete

    def interrupt(games, limit):
        original(games, limit)
        signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(model.roles, "complete", interrupt)
    with pytest.raises(KeyboardInterrupt):
        model.learn(1)
    monkeypatch.undo()
    assert model.phase == CohortPhase.IDLE and model.roles.role == "actor"
    model.save(tmp_path / "boundary.zip")
    env.close()
    env = vector_env(cfg, "masked", 101)
    try:
        loaded = CudaCompleteGamePPO.load(tmp_path / "boundary.zip", env=env, device="cuda")
        assert loaded.roles.role == "actor" and loaded.roles.games == 0
        loaded.learn(1, reset_num_timesteps=False)
        assert loaded.cohort_metrics["training_role"] == "actor"
        assert loaded.roles.role == "critic" and loaded.roles.games == 0
        assert loaded.training_games == 4
    finally:
        env.close()


def test_token_storage_math_and_transport_settings_do_not_change_resume_protocol():
    from pvz_rl.learning.training_requirements import resume_protocol

    cfg = load_config()
    changed = deepcopy(cfg)
    changed["training"]["performance"]["token_cache_gib"] = 0
    assert resume_protocol(cfg, "masked") == resume_protocol(changed, "masked")
    from pvz_game import Rules

    width = EventMemory(cfg, Rules(), 1, "cpu").width
    amount = 32768 * width * np.dtype(np.float32).itemsize
    assert 2 * 1024**3 // amount == 56
    assert amount * 56 == 2121269248
    assert width * 4 * 1024 * 48 == 56819712


def test_archived_parallelism_can_be_read_without_new_role_configuration():
    from pvz_rl.config import validate_config
    from pvz_rl.learning.training_requirements import require_cuda_training

    cfg = load_config()
    cfg["training"].pop("role_phase_games")
    cfg["training"]["n_envs"] = 1024
    validate_config(cfg)  # Existing inference/report configuration.
    with pytest.raises(ValueError, match="role_phase_games"):
        require_cuda_training(cfg, runtime=False)
