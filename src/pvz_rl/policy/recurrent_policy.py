"""SB3 lifecycle adapter with exactly the demonstration model's weight names."""

from types import SimpleNamespace

import torch

from pvz_rl.policy.transformer_lstm import TransformerLSTMPolicy


class RecurrentQPolicy(TransformerLSTMPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *,
        features_extractor_kwargs,
        exploration_epsilon=0.0,
        tile_exploration_epsilon=0.0,
    ):
        super().__init__(features_extractor_kwargs["layout_cfg"])
        self.observation_space, self.action_space = observation_space, action_space
        self.features_extractor = SimpleNamespace(cfg=self.cfg, layout=self.layout)
        self.exploration_epsilon = exploration_epsilon
        self.tile_exploration_epsilon = tile_exploration_epsilon
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1), eps=1e-8)

    @property
    def device(self):
        return next(self.parameters()).device

    def set_training_mode(self, mode):
        self.train(mode)

    def enable_compilation(self):
        # The recurrent sequence path uses the fused PyTorch LSTM directly.
        pass

    @torch.no_grad()
    def predict(
        self, observation, state=None, episode_start=None, deterministic=True, action_masks=None
    ):
        from pvz_rl.envs.encoding import collate_observations
        from pvz_rl.policy.runner import PolicyRunner
        from pvz_rl.policy.sequential_q import observation_tile_masks

        vectorized = not isinstance(observation, dict)
        obs = collate_observations(observation, self.device)
        if state is None:
            state = PolicyRunner(self, self.cfg, self.layout.rules, len(obs), self.device)
        if episode_start is not None:
            state.reset(torch.as_tensor(episode_start, device=self.device).bool())
        masks = (
            observation_tile_masks(obs)
            if action_masks is None
            else torch.as_tensor(action_masks, device=self.device, dtype=torch.bool).reshape(
                len(obs), -1
            )
        )
        actions = state.decide(obs, masks, None, deterministic=deterministic)[0].cpu().numpy()
        return (actions if vectorized else actions[0]), state
