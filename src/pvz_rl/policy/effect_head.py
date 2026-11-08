"""One forecast query over the executed action, chosen cell, and board memory."""

import torch
from torch import nn

from pvz_rl.policy.entity_attention import attend, stable_norm


class EffectHead(nn.Module):
    """Fuse the action and tile vectors, then cross-attend the full board memory.

    Wait and a rejected action pass ``empty_tile`` instead of a tile-decoder row.
    """

    def __init__(self, width, effect_width, heads, query_chunk):
        super().__init__()
        if width % heads:
            raise ValueError("shared_width must be divisible by effect attention heads")
        self.heads, self.query_chunk = heads, query_chunk
        self.effect_width = effect_width
        self.empty_tile = nn.Parameter(torch.zeros(width))
        self.fuse = nn.Linear(width * 2, width)
        self.query_norm, self.memory_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, width * 2)
        self.output = nn.Linear(width, width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 5 * effect_width)
        )

    def forward(self, action, tile, memory, valid):
        squeeze = action.ndim == 2 and tile.ndim == 2
        if action.ndim == 2:
            action = action[:, None, :]
        if tile.ndim == 2:
            tile = tile[:, None, :]
        if action.shape[1] == 1 and tile.shape[1] != 1:
            action = action.expand(-1, tile.shape[1], -1)
        elif tile.shape[1] == 1 and action.shape[1] != 1:
            tile = tile.expand(-1, action.shape[1], -1)
        if action.shape[:2] != tile.shape[:2] or action.shape[-1] != tile.shape[-1]:
            raise ValueError("Action and tile vectors must broadcast over the same queries")
        queries = self.fuse(torch.cat((action, tile), -1))
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
        logits = self.mlp(hidden).reshape(batch, length, 5, self.effect_width)
        return logits.squeeze(1) if squeeze else logits
