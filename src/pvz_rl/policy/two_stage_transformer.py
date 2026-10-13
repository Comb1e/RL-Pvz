"""Stateless, independent species and conditional-placement Q Transformers."""

from contextlib import contextmanager

import numpy as np
import torch
from pvz_game import Rules
from torch import nn

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.policy.active_batch import ActiveBatchPlan, encode_active
from pvz_rl.policy.entity_attention import SpatialTransformer
from pvz_rl.policy.sequential_q import (
    POLICY_SIGNATURE,
    action_parts,
    observation_tile_masks,
    select_q_actions,
)


class TwoStageTransformerPolicy(nn.Module):
    protocol = POLICY_SIGNATURE

    def __init__(self, cfg):
        super().__init__()
        from pvz_rl.envs.encoding import ObservationEncoder

        self.cfg = cfg
        self.layout = ObservationEncoder(cfg, Rules())
        self.fit_precision = "bf16_master"
        spec = cfg["policy"]
        self.species_encoder = SpatialTransformer(self.layout, spec, "species")
        self.tile_encoder = SpatialTransformer(self.layout, spec, "tile")
        self.species_head = nn.Sequential(
            nn.Linear(spec["species_width"], 128), nn.ReLU(), nn.Linear(128, 10)
        )
        self.tile_head = nn.Sequential(
            nn.Linear(spec["tile_width"], 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self._master_parameters = []
        self._master_names = []
        self._initialize_masters()
        self._set_runtime_dtype(torch.bfloat16)

    @property
    def encoders(self):
        return self.species_encoder, self.tile_encoder

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def compilation_status(self):
        statuses = [encoder.compilation_status for encoder in self.encoders]
        for status in ("fallback", "unavailable", "requested", "compiled"):
            if status in statuses:
                return status
        return "disabled"

    @property
    def compilation_backend(self):
        return next(
            (
                encoder.compilation_backend
                for encoder in self.encoders
                if encoder.compilation_backend
            ),
            None,
        )

    @property
    def compilation_error(self):
        return next(
            (encoder.compilation_error for encoder in self.encoders if encoder.compilation_error),
            None,
        )

    def enable_compilation(self):
        for encoder in self.encoders:
            encoder.enable_compilation()

    def _initialize_masters(self):
        self._master_names = []
        self._master_parameters = []
        for name, parameter in self.named_parameters():
            self._master_names.append(name)
            self._master_parameters.append(parameter.detach().float().clone().requires_grad_(True))

    def _set_runtime_dtype(self, dtype):
        if dtype is not torch.bfloat16:
            raise ValueError("The two-stage policy only supports BF16 runtime parameters")
        for parameter in super().parameters():
            parameter.data = parameter.data.to(dtype)

    def _apply(self, fn):
        result = super()._apply(fn)
        for master in getattr(self, "_master_parameters", ()):
            master.data = fn(master.data).float()
            if master.grad is not None:
                master.grad = fn(master.grad).float()
        return result

    def master_parameters(self):
        return tuple(self._master_parameters)

    def requires_grad_(self, requires_grad=True):
        super().requires_grad_(requires_grad)
        for master in self._master_parameters:
            master.requires_grad_(requires_grad)
        return self

    def runtime_parameters(self):
        return tuple(super().parameters())

    def master_state_dict(self):
        return {
            name: master.detach().clone()
            for name, master in zip(self._master_names, self._master_parameters, strict=True)
        }

    def runtime_state_dict(self):
        return super().state_dict()

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for name, master in zip(self._master_names, self._master_parameters, strict=True):
            if name in state:
                state[name] = master.detach().clone()
        return state

    @torch.no_grad()
    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        # Weight-only initialization carries authoritative FP32 masters.  Do
        # not round-trip those values through BF16 runtime parameters; teacher
        # and recovery runtime states are BF16 and intentionally follow the
        # cast runtime values instead.
        for name, master, parameter in zip(
            self._master_names, self._master_parameters, super().parameters(), strict=True
        ):
            source = state_dict.get(name)
            if isinstance(source, torch.Tensor) and source.dtype == torch.float32:
                master.copy_(source.to(device=master.device))
            else:
                master.copy_(parameter.float())
        return result

    @torch.no_grad()
    def _sync_masters_from_runtime(self):
        for master, parameter in zip(self._master_parameters, super().parameters(), strict=True):
            master.copy_(parameter.float())

    @torch.no_grad()
    def sync_runtime_from_masters(self):
        for master, parameter in zip(self._master_parameters, super().parameters(), strict=True):
            parameter.copy_(master.to(dtype=torch.bfloat16))

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for master in self._master_parameters:
            if set_to_none:
                master.grad = None
            elif master.grad is None:
                master.grad = torch.zeros_like(master)
            else:
                master.grad.zero_()

    def clear_runtime_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def accumulate_master_gradients(self):
        for master, parameter in zip(self._master_parameters, super().parameters(), strict=True):
            if parameter.grad is None:
                continue
            if master.grad is None:
                master.grad = torch.zeros_like(master)
            master.grad.add_(parameter.grad.float())

    @property
    def runtime_dtype(self):
        return torch.bfloat16

    @contextmanager
    def fitting_precision(self, precision):
        if precision != "bf16_master":
            raise ValueError("Only bf16_master fitting is supported")
        yield

    def _encode(self, encoder, observations, branches=None, plan=None):
        return (
            encoder(observations, branches)
            if plan is None
            else encode_active(encoder, observations, plan, branches)
        )

    def species_values(self, observations, *, plan=None):
        observations = collate_observations(observations, self.device)
        encoded = self._encode(self.species_encoder, observations, plan=plan)[:, 0]
        if plan is None:
            return self.species_head(encoded)
        rows = torch.as_tensor(plan.original_slot_ids, device=observations.device)
        result = encoded.new_zeros(len(observations), 10)
        return result.index_copy(0, rows, self.species_head(encoded[rows]))

    def _request_plan(self, plan, rows):
        """Bound tile padding by published live widths without copying request IDs."""
        if plan is None:
            return None
        width = max((bucket.entity_width for bucket in plan.buckets), default=0)
        return ActiveBatchPlan.from_counts(
            np.ones(len(rows), dtype=bool),
            np.full(len(rows), width, dtype=np.int64),
            self.layout.max_entities,
            self.tile_encoder.microbatch,
            self.tile_encoder.token_budget,
            token_overhead=self.tile_encoder.token_overhead,
        )

    def conditional_tiles(self, observations, branches, *, rows=None, plan=None):
        observations = collate_observations(observations, self.device)
        branches = torch.as_tensor(branches, device=self.device, dtype=torch.long)
        if branches.ndim != 1:
            raise ValueError("Conditional branches must be a vector")
        if rows is None:
            if len(branches) != len(observations):
                raise ValueError("Conditional branches must match the board batch")
            rows = torch.arange(len(branches), device=self.device)
        else:
            rows = torch.as_tensor(rows, device=self.device, dtype=torch.long)
            if rows.shape != branches.shape:
                raise ValueError("Requested rows and branches must have matching shapes")
        result = next(self.parameters()).new_zeros(len(branches), A.tiles)
        requested = torch.nonzero(branches > 0, as_tuple=False).flatten()
        if not len(requested):
            return result
        pairs = rows[requested] * 10 + branches[requested]
        unique, inverse = torch.unique(pairs, sorted=True, return_inverse=True)
        source_rows, source_branches = unique // 10, unique % 10
        request_plan = self._request_plan(plan, source_rows)
        encoded = self._encode(
            self.tile_encoder, observations[source_rows], source_branches, request_plan
        )[:, 2:]
        values = self.tile_head(encoded).squeeze(-1)
        return result.index_copy(0, requested, values[inverse])

    def selected_values(self, observations, actions, *, species=None, tiles=None):
        observations = collate_observations(observations, self.device)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        if actions.shape != (len(observations),):
            raise ValueError("Selected actions must match the board batch")
        branches, locations = action_parts(actions)
        nonwait = torch.nonzero(branches > 0, as_tuple=False).flatten()
        waits = torch.nonzero(branches == 0, as_tuple=False).flatten()
        result = next(self.parameters()).new_zeros(len(actions))
        if len(waits):
            wait_values = (
                self.species_values(observations[waits])[:, 0]
                if species is None
                else species[waits, 0]
            )
            result = result.index_copy(0, waits, wait_values)
        if len(nonwait):
            requested_tiles = (
                self.conditional_tiles(observations, branches[nonwait], rows=nonwait)
                if tiles is None
                else tiles[nonwait]
            )
            chosen = requested_tiles.gather(1, locations[nonwait, None]).flatten()
            result = result.index_copy(0, nonwait, chosen)
        return result

    def decide(
        self,
        observations,
        *,
        action_masks=None,
        deterministic=True,
        active=None,
        active_plan=None,
    ):
        observations = collate_observations(observations, self.device)
        species = self.species_values(observations, plan=active_plan)
        if action_masks is None:
            action_masks = observation_tile_masks(observations)
        if action_masks.shape != (len(observations), A.size):
            raise ValueError("Action masks must match the board batch")
        branches = species.argmax(-1)
        if active_plan is not None and active is None:
            active = torch.as_tensor(active_plan.compact_indices >= 0, device=self.device)
        if active is not None:
            branches = torch.where(active, branches, 0)
        tiles = self.conditional_tiles(observations, branches, plan=active_plan)
        actions, _, _, details = select_q_actions(
            species,
            tiles,
            action_masks,
            deterministic=deterministic,
            tile_exploration_epsilon=getattr(self, "tile_exploration_epsilon", 0.0),
            active=active,
        )
        details["observations"] = observations
        return actions, details
