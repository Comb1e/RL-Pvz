"""Shared entity embeddings and exact, bounded bidirectional attention."""

import re
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from pvz_rl.policy.cudagraph_backend import CollectionGraphCache, FittingGraphBackend


def feature_dtype(device):
    return torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type) else None


def attend(q, k, v, mask, *, query_chunk, force_fallback=False):
    """Scaled dot-product attention. A bool mask keeps positions marked true."""
    supported = True
    # SDPAParams is a pybind11 object.  Constructing it during a Dynamo
    # trace creates a graph break before the actual tensor attention runs.
    # The compiled path delegates backend selection to SDPA directly; the
    # capability probe remains available for the eager path.
    compiling = bool(getattr(torch.compiler, "is_compiling", lambda: False)())
    if q.is_cuda and not force_fallback and not compiling:
        params = torch.backends.cuda.SDPAParams(q, k, v, mask, 0.0, False, False)
        supported = torch.backends.cuda.can_use_efficient_attention(
            params
        ) or torch.backends.cuda.can_use_flash_attention(params)
    if not force_fallback and supported:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    # Chunk queries, never keys: every query still sees every present entity.
    outputs = []
    for start in range(0, q.shape[-2], query_chunk):
        query = q[..., start : start + query_chunk, :]
        with torch.autocast(q.device.type, enabled=False):
            dtype = torch.float32 if q.dtype == torch.bfloat16 else q.dtype
            logits = (query.to(dtype) @ k.to(dtype).transpose(-2, -1)) * q.shape[-1] ** -0.5
            output = (logits.masked_fill(~mask, -torch.inf).softmax(-1) @ v.to(dtype)).to(q.dtype)
        outputs.append(output)
    return torch.cat(outputs, dim=-2)


def stable_norm(layer, value):
    # Preserve float64 mathematical controls; BF16 reductions use FP32.
    dtype = value.dtype
    with torch.autocast(value.device.type, enabled=False):
        norm_dtype = torch.float32 if dtype == torch.bfloat16 else dtype
        return F.layer_norm(
            value.to(norm_dtype),
            layer.normalized_shape,
            layer.weight.to(norm_dtype),
            layer.bias.to(norm_dtype),
            layer.eps,
        ).to(dtype)


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
        return attend(
            q, k, v, mask, query_chunk=self.query_chunk, force_fallback=self.force_fallback
        )

    def forward(self, x, valid):
        batch, length, width = x.shape
        qkv = self.qkv(stable_norm(self.norm1, x)).reshape(batch, length, 3, self.heads, -1)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        attended = self.attention(q, k, v, valid[:, None, None, :])
        x = x + self.output(attended.transpose(1, 2).reshape(batch, length, width))
        x = x + self.ff(stable_norm(self.norm2, x))
        return x.masked_fill(~valid[..., None], 0)


class SpatialTransformer(nn.Module):
    """Public-board memory encoder.

    Token 0 is the global board summary. The next five tokens are the previous
    decision's forecast, supplied by the caller. Entity tokens follow them.
    """

    def __init__(self, layout, spec, stage):
        super().__init__()
        if stage != "board":
            raise ValueError("Unknown spatial Transformer stage")
        self.layout = layout
        self.stage = stage
        self.token_overhead = 6
        self.width = width = spec["shared_width"]
        self.microbatch = spec["encoder_microbatch"]
        self.token_budget = spec["encoder_token_budget"]
        self.activation_checkpointing = False
        self.fitting = False
        self._fitting_inference_cache = CollectionGraphCache(6, 1024)
        self._compiled_encode = None
        self._compiled_shapes = {}
        self._fitting_backend = None
        performance = layout.cfg["training"].get("performance", {})
        self.compile_collection = performance.get("compile_collection", False)
        self.compile_fitting = performance.get("compile_fitting", False)
        self._collection_cache = CollectionGraphCache(
            performance.get("collection_graph_cache_entries", 8),
            performance.get("collection_vram_headroom_mib", 512),
        )
        self.compilation_backend = None
        self.compilation_status = "disabled"
        self.compilation_error = None
        self.register_buffer("health_scales", torch.tensor(layout.health_scales), persistent=False)
        # Keep all rule-derived normalization constants as tensor buffers.  The
        # compiled encoder must not reach back through ObservationEncoder.rules:
        # that object is a pybind11 wrapper owned by the simulator and Dynamo
        # cannot trace it.  These buffers preserve the exact public normalization
        # while keeping the graph tensor-only.
        self.register_buffer(
            "row_scale", torch.tensor(float(max(1, layout.rows - 1))), persistent=False
        )
        self.register_buffer(
            "position_origin", torch.tensor(float(layout.rules.game["house_x"])), persistent=False
        )
        self.register_buffer(
            "position_scale_buffer",
            torch.tensor(float(layout.position_scale)),
            persistent=False,
        )
        self.register_buffer(
            "armor_scale_buffer",
            torch.tensor(float(layout.armor_scale)),
            persistent=False,
        )
        self.register_buffer(
            "tick_rate", torch.tensor(float(layout.rules.game["tick_rate"])), persistent=False
        )
        self.register_buffer(
            "projectile_damage",
            torch.tensor(float(layout.rules.plants["peashooter"]["damage"])),
            persistent=False,
        )
        self.type_embedding = nn.Embedding(len(layout.types) + 1, width, padding_idx=0)
        self.state_embedding = nn.Embedding(len(layout.states) + 1, width, padding_idx=0)
        self.numeric = nn.Linear(9, width)
        self.input_norm = nn.LayerNorm(width)
        self.global_projection = nn.Linear(layout.global_width, width)
        self.layers = nn.ModuleList(
            [
                AttentionBlock(
                    width,
                    spec["attention_heads"],
                    spec["feedforward_width"],
                    spec["attention_query_chunk"],
                )
                for _ in range(spec["encoder_blocks"])
            ]
        )
        self.norm = nn.LayerNorm(width)

    def enable_compilation(self):
        """Request lazy compilation for fixed encoder shapes.

        Compilation is deliberately limited to the tensor-only encoder.  Ragged
        storage, action selection and simulator code stay eager.  A
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
        if self._compiled_encode is not None:
            return
        self._fitting_backend = FittingGraphBackend()
        self._compiled_encode = self._fitting_backend.compile(self._encode)
        self._compiled_shapes.clear()
        self.compilation_backend = "cudagraphs_owned"
        self.compilation_status = "requested"
        self.compilation_error = None

    @property
    def collection_metrics(self):
        """Cumulative admission/fallback counters plus the current live entry count."""
        return self._collection_cache.metrics

    def configure_collection(self, performance):
        """Refresh execution settings at a quiescent boundary, including after resume."""
        self.compile_collection = performance.get("compile_collection", False)
        self.compile_fitting = performance.get("compile_fitting", False)
        entries = performance.get("collection_graph_cache_entries", 8)
        headroom = performance.get("collection_vram_headroom_mib", 512)
        if any(type(value) is not int or value < 0 for value in (entries, headroom)):
            raise ValueError("Collection graph settings must be nonnegative integers")
        cache = self._collection_cache
        if (entries, headroom * 1024**2) != (cache.limit, cache.headroom_bytes):
            self.release_collection()
            cache.limit = entries
            cache.headroom_bytes = headroom * 1024**2

    def release_collection(self):
        """Quiescent FIT entry: release collection buffers without disabling compilation."""
        self._collection_cache.release()

    def release_retired(self):
        """Drain completed evictions after the existing collection handoff, without waiting."""
        return self._collection_cache.release_retired()

    def release_fitting(self, *, recompile=False):
        """Quiescent collection entry, after all fitting backwards and stream work finish.

        Captures re-record lazily; in-place parameter updates keep traced graphs valid.
        Recompiling every cohort retained compiler artifacts and grew fitting time, so
        only shape-changing fallbacks request a fresh phase cache.
        """
        if self._fitting_backend is not None:
            self._fitting_backend.release()
        self._fitting_inference_cache.release()
        self.fitting = False
        if recompile and self._compiled_encode is not None:
            self._compiled_shapes.clear()
            self._compiled_encode = self._fitting_backend.compile(self._encode)
            self.compilation_status = "requested"

    def disable_compilation(self, exc):
        """Record one backend failure and retain the exact eager encoder."""
        if self.compilation_status == "fallback":
            return
        self._compiled_encode = None
        self._compiled_shapes.clear()
        self.release_collection()
        self.compilation_status = "fallback"
        detail = str(exc).splitlines()[0] if str(exc) else "backend compilation failed"
        self.compilation_error = re.sub(r"https?://\S+", "<url>", detail)[:256]
        warnings.warn(
            f"Encoder compilation failed; using eager execution: {self.compilation_error}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _encode_with_optional_compile(self, args):
        enabled = self.compile_fitting if self.fitting else self.compile_collection
        if self._compiled_encode is None or not enabled:
            return self._encode(*args)
        if not torch.is_grad_enabled():
            if not args[0].is_cuda:
                return self._encode(*args)
            try:
                cache = self._fitting_inference_cache if self.fitting else self._collection_cache
                value = cache(self._encode, args)
                if cache.captures:
                    self.compilation_status = "compiled"
                return value
            except Exception as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
                self.disable_compilation(exc)
            return self._encode(*args)
        shape = (int(args[0].shape[0]), int(args[0].shape[1]))
        # Dynamic ragged shapes would trigger an unbounded stream of compiler
        # graphs.  Compile at most six masked entity buckets per phase and use
        # eager execution for additional buckets.
        limit = 6 if self.fitting else 4
        if shape not in self._compiled_shapes and len(self._compiled_shapes) >= limit:
            return self._encode(*args)
        try:
            value = self._compiled_encode(*args)
            self._compiled_shapes[shape] = True
            self.compilation_status = "compiled"
            return value
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:  # backend availability is device/build dependent
            self.disable_compilation(exc)
            return self._encode(*args)

    def _normalize_tensors(self, entities, entity_mask):
        """Normalize validated records without touching simulator Python objects."""
        raw = entities
        kind, state, valid = raw[..., 0].long(), raw[..., 1].long(), entity_mask
        kind = torch.where(valid, kind, 0)
        state = torch.where(valid, state, 0)
        values = torch.where(valid[..., None], raw[..., 2:], 0).float()
        values[..., 0] /= self.row_scale
        values[..., 1] = (values[..., 1] - self.position_origin) / self.position_scale_buffer
        values[..., 2] /= self.health_scales[kind]
        values[..., 3] /= self.armor_scale_buffer
        values[..., 4:6] /= self.tick_rate
        values[..., 8] /= self.projectile_damage
        return kind, state, values

    def _embed_tensors(self, entities, entity_mask):
        kind, state, numeric = self._normalize_tensors(entities, entity_mask)
        dtype = feature_dtype(entities.device) or self.numeric.weight.dtype
        accumulation = torch.float32 if dtype == torch.bfloat16 else dtype
        result = stable_norm(
            self.input_norm,
            self.type_embedding(kind).to(dtype).to(accumulation)
            + self.state_embedding(state).to(dtype).to(accumulation)
            + self.numeric(numeric.to(self.numeric.weight.dtype)).to(accumulation),
        )
        return result.to(dtype).masked_fill(~entity_mask[..., None], 0)

    def embed(self, batch):
        """Embed a public EntityBatch; retained for diagnostics and math tests."""
        return self._embed_tensors(batch.entities, batch.entity_mask)

    def _encode(self, entities, mask, globals_, branches, forecast_tokens, forecast_valid):
        del branches
        embedded = self._embed_tensors(entities, mask)
        with torch.autocast(globals_.device.type, enabled=False):
            readouts = self.global_projection(globals_.to(self.global_projection.weight.dtype))[
                :, None
            ]
        forecast_tokens = forecast_tokens.to(dtype=embedded.dtype)
        forecast_tokens = forecast_tokens.masked_fill(~forecast_valid[:, None, None], 0)
        lane_valid = forecast_valid[:, None].expand(-1, 5)
        x = torch.cat((readouts, forecast_tokens, embedded), 1)
        valid = torch.cat(
            (
                torch.ones(readouts.shape[:2], dtype=torch.bool, device=x.device),
                lane_valid,
                mask,
            ),
            1,
        )
        for layer in self.layers:
            x = layer(x, valid)
        return stable_norm(self.norm, x)

    def forward(self, observations, branches=None, forecast=None):
        self.layout.validate(observations)
        if len(observations.shape) != 1:
            raise ValueError("Spatial inference requires flat board batches")
        if branches is None:
            branches = torch.zeros(len(observations), dtype=torch.long, device=observations.device)
        if branches.shape != (len(observations),) or branches.dtype != torch.long:
            raise ValueError("Branches must be an int64 board vector")
        if not len(observations):
            return self.numeric.weight.new_empty(0, self.token_overhead, self.width)
        if forecast is None:
            tokens = self.numeric.weight.new_zeros(len(observations), 5, self.width)
            forecast_valid = torch.zeros(
                len(observations), dtype=torch.bool, device=observations.device
            )
        else:
            tokens, forecast_valid = forecast
            if tokens.shape != (len(observations), 5, self.width) or forecast_valid.shape != (
                len(observations),
            ):
                raise ValueError("Forecast tokens must be [batch, 5, width] with one valid bit")
        length = observations.entities.shape[-2] + self.token_overhead
        size = max(1, min(self.microbatch, self.token_budget // length))
        results = []
        for start in range(0, len(observations), size):
            part = observations[start : start + size]
            args = (
                *part.tensors(),
                branches[start : start + size],
                tokens[start : start + size],
                forecast_valid[start : start + size],
            )
            logical = len(part)
            compile_args = args
            if (
                self._compiled_encode is not None
                and (torch.is_grad_enabled() or self.fitting)
                and logical < size
            ):
                padding = size - logical
                compile_args = tuple(
                    torch.cat((value, value.new_zeros((padding, *value.shape[1:]))), dim=0)
                    for value in args
                )
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                value = checkpoint(self._encode, *args, use_reentrant=False)
            elif self._compiled_encode is not None:
                value = self._encode_with_optional_compile(compile_args)
            else:
                value = self._encode(*args)
            results.append(value[:logical])
        return torch.cat(results, 0)
