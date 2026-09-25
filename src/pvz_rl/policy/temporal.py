"""Gated causal attention over a bounded public event history.

Only the current query is evaluated. Attention is O(memory), not O(ticks²).
Raw memories are re-encoded by current weights, avoiding stale learned KV caches
between PPO updates. This is a project-specific sparse Transformer, not GTrXL.
"""

import math

import torch
from torch import nn

from pvz_rl.envs.actions import ActionSchema as A


class GatedMemoryBlock(nn.Module):
    def __init__(self, spec):
        super().__init__()
        d = spec["model_width"]
        self.heads, self.depth = spec["heads"], d // spec["heads"]
        self.norm = nn.LayerNorm(d)
        self.query, self.key, self.value = (nn.Linear(d, d) for _ in range(3))
        self.output = nn.Linear(d, d)
        self.feed_norm = nn.LayerNorm(d)
        self.feed = nn.Sequential(
            nn.Linear(d, spec["feedforward_width"]),
            nn.ReLU(),
            nn.Linear(spec["feedforward_width"], d),
        )
        self.attn_gate, self.feed_gate = nn.Linear(d, d), nn.Linear(d, d)
        for gate in (self.attn_gate, self.feed_gate):
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, -spec["gate_bias"])

    def forward(self, current, memory, valid):
        b, length, _ = memory.shape
        q = self.query(self.norm(current)).view(b, self.heads, 1, self.depth)
        normalized = self.norm(memory)
        k = self.key(normalized).view(b, length, self.heads, self.depth).transpose(1, 2)
        v = self.value(normalized).view(b, length, self.heads, self.depth).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.depth)
        weights = scores.masked_fill(~valid[:, None, None, :], -torch.inf).softmax(-1)
        result = (weights @ v).reshape(b, -1)
        current = current + self.attn_gate(current).sigmoid() * self.output(result).relu()
        current = (
            current + self.feed_gate(current).sigmoid() * self.feed(self.feed_norm(current)).relu()
        )
        return current, weights.mean(1).squeeze(1)


class TemporalEncoder(nn.Module):
    def __init__(self, layout, spec, plant_dim, state_dim):
        super().__init__()
        self.layout, self.spec = layout, spec
        width = 45 * (plant_dim + state_dim + 1) + layout.size - 135
        d = spec["model_width"]
        self.observation = nn.Linear(width, d)
        self.action = nn.Embedding(A.size, d)
        self.reset = nn.Linear(1, d, bias=False)
        # Relative time, covered time span and number of represented observations.
        self.time = nn.Sequential(nn.Linear(3, d), nn.Tanh())
        self.blocks = nn.ModuleList(GatedMemoryBlock(spec) for _ in range(spec["layers"]))

    def forward(self, context, plant_types, plant_states):
        raw = context.tokens
        b, length, _ = raw.shape
        obs = raw[..., : self.layout.size]
        grid = obs[..., :135].reshape(b, length, 45, 3)
        plants = torch.cat(
            (plant_types(grid[..., 0].long()), grid[..., 1:2], plant_states(grid[..., 2].long())),
            -1,
        ).flatten(2)
        # Elapsed seconds remains a public scalar. Entity/card countdowns are absent.
        encoded = self.observation(torch.cat((plants, obs[..., 135:]), -1))
        encoded = encoded + self.action(raw[..., -3].long()) + self.reset(raw[..., -2:-1])
        current_index = self.spec["local_tokens"] - 1
        now = raw[:, current_index, -1:]
        age = (now - raw[..., -1]).clamp_min(0) / self.layout.rules.game["tick_rate"]
        span = (raw[..., -1] - context.starts).clamp_min(0) / self.layout.rules.game["tick_rate"]
        timing = torch.stack((age.log1p(), span.log1p(), context.counts.clamp_min(0).log1p()), -1)
        memory = encoded + self.time(timing)
        valid = context.valid & (raw[..., -1] <= now)
        current = encoded[:, current_index]
        for block in self.blocks:
            current, weights = block(current, memory, valid)
        self.last_attention = weights.detach()
        return current
