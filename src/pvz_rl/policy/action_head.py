"""Ten action scores from CLS and five continuous lane features."""

from torch import nn

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.policy.entity_attention import stable_norm
from pvz_rl.policy.mlp import build_mlp


class ActionHead(nn.Module):
    def __init__(self, width, hidden_width, hidden_layers):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mlp = build_mlp((A.rows + 1) * width, hidden_width, A.plant_types + 2, hidden_layers)

    def forward(self, memory):
        if memory.ndim != 3 or memory.shape[1:] != (A.rows + 1, self.norm.normalized_shape[0]):
            raise ValueError("Action head requires CLS followed by five lane vectors")
        return self.mlp(stable_norm(self.norm, memory).flatten(1))
