"""Shared entity embeddings and exact, bounded bidirectional attention."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from pvz_rl.envs.encoding import EntityBatch


class AttentionBlock(nn.Module):
    def __init__(self, width, heads, feedforward, query_chunk):
        super().__init__()
        self.heads, self.query_chunk = heads, query_chunk
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.output = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.ff = nn.Sequential(
            nn.Linear(width, feedforward), nn.GELU(), nn.Linear(feedforward, width)
        )
        self.force_fallback = False

    def attention(self, q, k, v, mask):
        # Chunk queries, never keys: every query still sees every present entity.
        outputs = []
        for start in range(0, q.shape[-2], self.query_chunk):
            query = q[..., start : start + self.query_chunk, :]

            def attend(query, key, value, valid):
                if not self.force_fallback:
                    return F.scaled_dot_product_attention(query, key, value, attn_mask=valid)
                logits = (query @ key.transpose(-2, -1)) * query.shape[-1] ** -0.5
                weights = logits.masked_fill(~valid, -torch.inf).softmax(-1)
                return weights @ value

            # The math backend would otherwise retain quadratic score tensors.
            if torch.is_grad_enabled() and self.training:
                output = checkpoint(attend, query, k, v, mask, use_reentrant=False)
            else:
                output = attend(query, k, v, mask)
            outputs.append(output)
        return torch.cat(outputs, dim=-2)

    def forward(self, x, valid):
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, -1)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        attended = self.attention(q, k, v, valid[:, None, None, :])
        x = x + self.output(attended.transpose(1, 2).reshape(batch, length, width))
        x = x + self.ff(self.norm2(x))
        return x.masked_fill(~valid[..., None], 0)


class EntityTransformer(nn.Module):
    """n physical entities plus one global and 45 coordinate readout tokens."""

    def __init__(self, layout, spec):
        super().__init__()
        self.layout = layout
        self.width = width = spec["entity_width"]
        self.microbatch = spec["encoder_microbatch"]
        self.token_budget = spec["encoder_token_budget"]
        self.type_embedding = nn.Embedding(len(layout.types) + 1, width, padding_idx=0)
        self.state_embedding = nn.Embedding(len(layout.states) + 1, width, padding_idx=0)
        self.numeric = nn.Linear(9, width)
        self.input_norm = nn.LayerNorm(width)
        self.global_projection = nn.Linear(layout.global_width, width)
        self.tile_marker = nn.Parameter(torch.zeros(width))
        coordinates = torch.zeros(layout.rows * layout.cols, 9)
        index = torch.arange(len(coordinates))
        coordinates[:, 0] = (index // layout.cols) / max(1, layout.rows - 1)
        units = layout.rules.game["units_per_tile"]
        coordinates[:, 1] = (
            (index % layout.cols) * units + units // 2 - layout.rules.game["house_x"]
        ) / layout.position_scale
        self.register_buffer("tile_coordinates", coordinates)
        self.layers = nn.ModuleList(
            [
                AttentionBlock(
                    width,
                    spec["transformer_heads"],
                    spec["transformer_feedforward"],
                    spec["attention_query_chunk"],
                )
                for _ in range(spec["transformer_layers"])
            ]
        )
        self.norm = nn.LayerNorm(width)

    def embed(self, batch):
        kind, state, numeric = self.layout.normalize(batch)
        result = self.input_norm(
            self.type_embedding(kind)
            + self.state_embedding(state)
            + self.numeric(numeric.to(self.numeric.weight.dtype))
        )
        return result.masked_fill(~batch.entity_mask[..., None], 0)

    def _encode(self, entities, mask, globals_):
        batch = EntityBatch(entities, mask, globals_)
        embedded = self.embed(batch)
        tiles = self.numeric(self.tile_coordinates) + self.tile_marker
        readouts = torch.cat(
            (
                self.global_projection(globals_.to(self.global_projection.weight.dtype))[:, None],
                tiles[None].expand(len(batch), -1, -1),
            ),
            1,
        )
        x = torch.cat((readouts, embedded), 1)
        valid = torch.cat(
            (torch.ones(readouts.shape[:2], dtype=torch.bool, device=x.device), mask), 1
        )
        for layer in self.layers:
            x = layer(x, valid)
        # Only these outputs cross the recurrent boundary; entity activations
        # are reconstructed during backward by the outer checkpoint.
        return self.norm(x[:, : readouts.shape[1]])

    def forward(self, observations):
        length = observations.entities.shape[-2] + 46
        size = max(1, min(self.microbatch, self.token_budget // length))
        results = []
        for start in range(0, len(observations), size):
            part = observations[start : start + size]
            # Trim batch padding after splitting, without dropping real rows.
            occupied = part.entity_mask.any(0)
            last = torch.nonzero(occupied).flatten()
            count = int(last[-1]) + 1 if len(last) else 0
            args = (part.entities[:, :count], part.entity_mask[:, :count], part.globals)
            if self.training and torch.is_grad_enabled():
                value = checkpoint(self._encode, *args, use_reentrant=False)
            else:
                value = self._encode(*args)
            results.append(value)
        readouts = torch.cat(results, 0)
        return readouts[:, 0], readouts[:, 1:]
