"""Five lane queries over the current board, before any action is chosen."""

import torch
from torch import nn

from pvz_rl.policy.entity_attention import attend, stable_norm


class EffectHead(nn.Module):
    """Cross-attend the encoder memory and emit one outcome vector per lane.

    The queries are learned lane embeddings. They do not see the action, the
    chosen cell, or these outputs.
    """

    def __init__(self, query_width, memory_width, effect_width, heads, query_chunk):
        super().__init__()
        if query_width % heads:
            raise ValueError("policy.effect.width must be divisible by policy.effect.heads")
        self.heads, self.query_chunk = heads, query_chunk
        self.effect_width = effect_width
        self.lane_queries = nn.Embedding(5, query_width)
        self.query_norm = nn.LayerNorm(query_width)
        self.memory_norm = nn.LayerNorm(memory_width)
        self.query = nn.Linear(query_width, query_width)
        self.key_value = nn.Linear(memory_width, query_width * 2)
        self.output = nn.Linear(query_width, query_width)
        self.mlp = nn.Sequential(
            nn.Linear(query_width, query_width), nn.GELU(), nn.Linear(query_width, effect_width)
        )

    def forward(self, memory, valid):
        if memory.ndim != 3 or valid.shape != memory.shape[:2]:
            raise ValueError("Effect head requires encoder memory and its valid mask")
        index = torch.arange(5, device=memory.device)
        queries = self.lane_queries(index).to(dtype=memory.dtype)
        queries = queries[None].expand(memory.shape[0], -1, -1).contiguous()
        batch, length, width = queries.shape
        q = self.query(stable_norm(self.query_norm, queries)).reshape(batch, length, self.heads, -1)
        kv = self.key_value(stable_norm(self.memory_norm, memory)).reshape(
            batch, memory.shape[1], 2, self.heads, -1
        )
        attended = attend(
            q.transpose(1, 2),
            kv[:, :, 0].transpose(1, 2),
            kv[:, :, 1].transpose(1, 2),
            valid[:, None, None, :],
            query_chunk=self.query_chunk,
        )
        hidden = queries + self.output(attended.transpose(1, 2).reshape(batch, length, width))
        return self.mlp(hidden)
