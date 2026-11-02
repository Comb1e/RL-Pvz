"""Parallel, action-conditioned spatial queries over one public board memory."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from pvz_rl.policy.entity_attention import AttentionBlock, SpatialTransformer, stable_norm


class DecoderBlock(nn.Module):
    def __init__(self, width, heads, feedforward, query_chunk):
        super().__init__()
        self.self_attention = AttentionBlock(width, heads, feedforward, query_chunk)
        self.query_norm, self.memory_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, width * 2)
        self.output = nn.Linear(width, width)
        self.heads = heads

    def forward(self, queries, memory, valid):
        queries = self.self_attention(
            queries, torch.ones(queries.shape[:2], device=queries.device, dtype=torch.bool)
        )
        b, length, width = queries.shape
        q = self.query(stable_norm(self.query_norm, queries)).reshape(b, length, self.heads, -1)
        kv = self.key_value(stable_norm(self.memory_norm, memory)).reshape(
            b, memory.shape[1], 2, self.heads, -1
        )
        attended = self.self_attention.attention(
            q.transpose(1, 2),
            kv[:, :, 0].transpose(1, 2),
            kv[:, :, 1].transpose(1, 2),
            valid[:, None, None, :],
        )
        return queries + self.output(attended.transpose(1, 2).reshape(b, length, width))


class TileDecoder(nn.Module):
    # Share the bounded graph lifecycle with the board encoder. The graph takes
    # tensor memory explicitly; no cached board survives a decision or update.
    enable_compilation = SpatialTransformer.enable_compilation
    collection_metrics = SpatialTransformer.collection_metrics
    configure_collection = SpatialTransformer.configure_collection
    release_collection = SpatialTransformer.release_collection
    release_retired = SpatialTransformer.release_retired
    release_fitting = SpatialTransformer.release_fitting
    disable_compilation = SpatialTransformer.disable_compilation
    _encode_with_optional_compile = SpatialTransformer._encode_with_optional_compile

    def __init__(self, layout, spec):
        super().__init__()
        from pvz_rl.policy.cudagraph_backend import CollectionGraphCache

        self.layout, self.stage = layout, "tile"
        self.width, self.token_overhead = spec["shared_width"], 45
        self.microbatch, self.token_budget = (
            spec["encoder_microbatch"],
            spec["encoder_token_budget"],
        )
        self.activation_checkpointing = self.fitting = False
        self._compiled_encode = self._fitting_backend = None
        self._compiled_shapes = {}
        self._fitting_inference_cache = CollectionGraphCache(6, 1024)
        self._collection_cache = CollectionGraphCache(8, 512)
        self.compilation_status, self.compilation_backend, self.compilation_error = (
            "disabled",
            None,
            None,
        )
        self.configure_collection(layout.cfg["training"]["performance"])
        self.branch_embedding = nn.Embedding(10, self.width)
        self.coordinates = nn.Linear(2, self.width)
        indices = torch.arange(45)
        self.register_buffer(
            "tile_coordinates",
            torch.stack(
                (
                    (indices // layout.cols) / max(1, layout.rows - 1),
                    (
                        (indices % layout.cols) * layout.rules.game["units_per_tile"]
                        + layout.rules.game["units_per_tile"] // 2
                        - layout.rules.game["house_x"]
                    )
                    / layout.position_scale,
                ),
                -1,
            ),
        )
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    self.width,
                    spec["attention_heads"],
                    spec["feedforward_width"],
                    spec["attention_query_chunk"],
                )
                for _ in range(spec["tile_decoder_blocks"])
            ]
        )
        self.norm = nn.LayerNorm(self.width)

    def _encode(self, memory, valid, branches):
        queries = self.coordinates(self.tile_coordinates.to(self.coordinates.weight.dtype))[None]
        queries = queries + self.branch_embedding(branches)[:, None]
        for layer in self.layers:
            queries = layer(queries, memory, valid)
        return stable_norm(self.norm, queries)

    def forward(self, memory, valid, branches):
        size = max(1, min(self.microbatch, self.token_budget // (memory.shape[1] + 45)))
        results = []
        for start in range(0, len(memory), size):
            args = (
                memory[start : start + size],
                valid[start : start + size],
                branches[start : start + size],
            )
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                result = checkpoint(self._encode, *args, use_reentrant=False)
            else:
                result = self._encode_with_optional_compile(args)
            results.append(result)
        return torch.cat(results) if results else memory.new_empty((0, 45, self.width))
