"""Ten action queries cross-attending one public board memory."""

import torch
from torch import nn

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.policy.entity_attention import SpatialTransformer, stable_norm
from pvz_rl.policy.tile_decoder import DecoderBlock, run_queries


class ActionDecoder(nn.Module):
    """One learned query per action id. Output is ``[batch, 10, width]``."""

    enable_compilation = SpatialTransformer.enable_compilation
    collection_metrics = SpatialTransformer.collection_metrics
    release_collection = SpatialTransformer.release_collection
    release_retired = SpatialTransformer.release_retired
    release_fitting = SpatialTransformer.release_fitting
    disable_compilation = SpatialTransformer.disable_compilation
    _encode_with_optional_compile = SpatialTransformer._encode_with_optional_compile

    def configure_collection(self, performance):
        SpatialTransformer.configure_collection(self, performance)
        self.collection_bucket_rows = performance.get("collection_tile_buckets", False)

    def __init__(self, layout, spec):
        super().__init__()
        from pvz_rl.policy.cudagraph_backend import CollectionGraphCache

        self.layout, self.stage = layout, "action"
        action, execution = spec["action"], spec["execution"]
        self.width, self.token_overhead = action["width"], A.plant_types + 2
        self.microbatch, self.token_budget = execution["microbatch"], execution["token_budget"]
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
        if self.width % action["heads"]:
            raise ValueError("policy.action.width must be divisible by policy.action.heads")
        self.action_queries = nn.Embedding(self.token_overhead, self.width)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    self.width,
                    action["heads"],
                    action["feedforward"],
                    execution["query_chunk"],
                    spec["board"]["width"],
                )
                for _ in range(action["blocks"])
            ]
        )
        self.norm = nn.LayerNorm(self.width)

    def _encode(self, memory, valid):
        index = torch.arange(self.token_overhead, device=memory.device)
        queries = self.action_queries(index).to(dtype=memory.dtype)
        queries = queries[None].expand(memory.shape[0], -1, -1).contiguous()
        for layer in self.layers:
            queries = layer(queries, memory, valid)
        return stable_norm(self.norm, queries)

    def forward(self, memory, valid):
        return run_queries(self, memory, valid)
