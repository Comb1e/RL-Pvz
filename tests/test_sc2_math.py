"""Independent complete-game return, controller and interruption controls."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from pvz_rl.config import load_config
from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.learning.cohort import ActorTransaction, CohortPhase
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
    c["training"].update(n_envs=2, n_epochs=1, batch_size=128)
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
    e = vector_env(c, "masked", 102)
    resumed = CudaCompleteGamePPO.load(tmp_path / "interrupted.zip", env=e, device="cuda")
    resumed.learn(1, reset_num_timesteps=False)
    assert_state_equal(expected, resumed.policy.state_dict())
    assert_state_equal(optim, resumed.policy.critic_optimizer.state_dict())
    assert hashes == [e.batch.state_hash(i) for i in range(2)]
    assert resumed.training_games == (4 if interrupt_after is None else 2)
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
