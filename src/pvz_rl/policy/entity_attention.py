"""Shared entity embeddings and exact, bounded bidirectional attention."""

import re

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from pvz_rl.envs.encoding import EntityBatch


def feature_dtype(device):
    return torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type) else None


def stable_norm(layer, value):
    # Preserve float64 mathematical controls; BF16 reductions use FP32.
    dtype = value.dtype
    with torch.autocast(value.device.type, enabled=False):
        return layer(value.float() if dtype == torch.bfloat16 else value).to(dtype)


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
        supported = True
        if q.is_cuda and not self.force_fallback:
            params = torch.backends.cuda.SDPAParams(q, k, v, mask, 0.0, False, False)
            supported = torch.backends.cuda.can_use_efficient_attention(
                params
            ) or torch.backends.cuda.can_use_flash_attention(params)
        if not self.force_fallback and supported:
            # PyTorch selects the available efficient implementation on this device.
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        # Chunk queries, never keys: every query still sees every present entity.
        outputs = []
        for start in range(0, q.shape[-2], self.query_chunk):
            query = q[..., start : start + self.query_chunk, :]

            with torch.autocast(q.device.type, enabled=False):
                dtype = torch.float32 if q.dtype == torch.bfloat16 else q.dtype
                logits = (query.to(dtype) @ k.to(dtype).transpose(-2, -1)) * q.shape[-1] ** -0.5
                output = (logits.masked_fill(~mask, -torch.inf).softmax(-1) @ v.to(dtype)).to(
                    q.dtype
                )
            outputs.append(output)
        return torch.cat(outputs, dim=-2)

    def forward(self, x, valid):
        batch, length, width = x.shape
        qkv = self.qkv(stable_norm(self.norm1, x)).reshape(batch, length, 3, self.heads, -1)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        attended = self.attention(q, k, v, valid[:, None, None, :])
        x = x + self.output(attended.transpose(1, 2).reshape(batch, length, width))
        x = x + self.ff(stable_norm(self.norm2, x))
        return x.masked_fill(~valid[..., None], 0)


class EntityTransformer(nn.Module):
    """n physical entities plus one global and 45 coordinate readout tokens."""

    def __init__(self, layout, spec):
        super().__init__()
        self.layout = layout
        self.width = width = spec["entity_width"]
        self.microbatch = spec["encoder_microbatch"]
        self.token_budget = spec["encoder_token_budget"]
        self.activation_checkpointing = False
        self._compiled_encode = None
        self._compiled_shapes = {}
        self.compilation_status = "disabled"
        self.compilation_error = None
        self.register_buffer("health_scales", torch.tensor(layout.health_scales), persistent=False)
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

    def enable_compilation(self):
        """Request lazy compilation for fixed encoder shapes.

        Compilation is deliberately limited to the tensor-only encoder.  Ragged
        storage, recurrent state management and simulator code stay eager.  A
        backend failure disables the wrapper and keeps the exact eager path.
        """
        if not hasattr(torch, "compile"):
            self.compilation_status = "unavailable"
            self.compilation_error = "torch.compile is unavailable"
            return
        if not torch.cuda.is_available():
            self.compilation_status = "unavailable"
            self.compilation_error = "CUDA is unavailable"
            return
        self._compiled_encode = torch.compile(self._encode, dynamic=False, mode="reduce-overhead")
        self._compiled_shapes.clear()
        self.compilation_status = "requested"
        self.compilation_error = None

    def _encode_with_optional_compile(self, args):
        if self._compiled_encode is None:
            return self._encode(*args)
        shape = (int(args[0].shape[0]), int(args[0].shape[1]))
        # Dynamic ragged shapes would trigger an unbounded stream of compiler
        # graphs.  Compile at most four observed fixed shapes per run and use
        # eager execution for additional buckets.
        if shape not in self._compiled_shapes and len(self._compiled_shapes) >= 4:
            return self._encode(*args)
        try:
            value = self._compiled_encode(*args)
            self._compiled_shapes[shape] = True
            self.compilation_status = "compiled"
            return value
        except Exception as exc:  # backend availability is device/build dependent
            self._compiled_encode = None
            self._compiled_shapes.clear()
            self.compilation_status = "fallback"
            detail = str(exc).splitlines()[0] if str(exc) else "backend compilation failed"
            self.compilation_error = re.sub(r"https?://\S+", "<url>", detail)[:256]
            return self._encode(*args)

    def embed(self, batch):
        kind, state, numeric = self.layout.normalize(batch, health_scales=self.health_scales)
        dtype = feature_dtype(batch.device) or self.numeric.weight.dtype
        accumulation = torch.float32 if dtype == torch.bfloat16 else dtype
        result = stable_norm(
            self.input_norm,
            self.type_embedding(kind).to(dtype).to(accumulation)
            + self.state_embedding(state).to(dtype).to(accumulation)
            + self.numeric(numeric.to(self.numeric.weight.dtype)).to(accumulation),
        )
        return result.masked_fill(~batch.entity_mask[..., None], 0)

    def _encode(self, entities, mask, globals_):
        batch = EntityBatch(entities, mask, globals_, validated=True)
        embedded = self.embed(batch)
        tiles = self.numeric(self.tile_coordinates)
        tiles = tiles + self.tile_marker.to(tiles.dtype)
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
        return stable_norm(self.norm, x[:, : readouts.shape[1]])

    def forward(self, observations):
        self.layout.validate(observations)
        length = observations.entities.shape[-2] + 46
        size = max(1, min(self.microbatch, self.token_budget // length))
        results = []
        for start in range(0, len(observations), size):
            part = observations[start : start + size]
            # Collation determines padding once; never read GPU lengths in this loop.
            args = part.tensors()
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                value = checkpoint(self._encode, *args, use_reentrant=False)
            elif self.training and torch.is_grad_enabled():
                value = self._encode_with_optional_compile(args)
            else:
                value = self._encode(*args)
            results.append(value)
        readouts = torch.cat(results, 0)
        return readouts[:, 0], readouts[:, 1:]
