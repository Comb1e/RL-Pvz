"""SB3 lifecycle adapter sharing the demonstration model's exact weight names."""

from types import SimpleNamespace

import torch

from pvz_rl.envs.encoding import collate_observations
from pvz_rl.policy.two_stage_transformer import TwoStageTransformerPolicy


class TwoStageQPolicy(TwoStageTransformerPolicy):
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
        self.optimizer = torch.optim.Adam(self.master_parameters(), lr=lr_schedule(1), eps=1e-8)

    def set_training_mode(self, mode):
        self.train(mode)

    @torch.no_grad()
    def predict(
        self, observation, state=None, episode_start=None, deterministic=True, action_masks=None
    ):
        vectorized = not isinstance(observation, dict)
        observations = collate_observations(observation, self.device)
        masks = (
            None
            if action_masks is None
            else torch.as_tensor(action_masks, device=self.device, dtype=torch.bool).reshape(
                len(observations), -1
            )
        )
        actions, _ = self.decide(observations, action_masks=masks, deterministic=deterministic)
        result = actions.cpu().numpy()
        return (result if vectorized else result[0]), None
