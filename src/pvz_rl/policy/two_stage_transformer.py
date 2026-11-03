"""Shared public-board encoding with conditional spatial Q and reach readouts."""

from contextlib import contextmanager

import torch
from pvz_game import Rules
from torch import nn

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.actions import dig_branch_available
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.learning.forecast_labels import (
    EFFECT_WIDTH,
    THREAT_WIDTH,
    effect_summary,
    penetration_map,
    threat_summary,
)
from pvz_rl.policy.active_batch import encode_active
from pvz_rl.policy.entity_attention import SpatialTransformer
from pvz_rl.policy.sequential_q import (
    POLICY_SIGNATURE,
    action_parts,
    observation_tile_masks,
    select_q_actions,
)
from pvz_rl.policy.tile_decoder import TileDecoder


class TwoStageTransformerPolicy(nn.Module):
    protocol = POLICY_SIGNATURE

    def __init__(self, cfg):
        super().__init__()
        from pvz_rl.envs.encoding import ObservationEncoder

        self.cfg = cfg
        self.layout = ObservationEncoder(cfg, Rules())
        self.fit_precision = "bf16_master"
        spec = cfg["policy"]
        self.board_encoder = SpatialTransformer(self.layout, spec, "board")
        self.tile_decoder = TileDecoder(self.layout, spec)
        width = spec["shared_width"]
        self.species_head = nn.Sequential(nn.Linear(width + 30, 128), nn.ReLU(), nn.Linear(128, 10))
        self.tile_head = nn.Sequential(nn.Linear(width + 20, 128), nn.ReLU(), nn.Linear(128, 1))
        self.threat_head = nn.Linear(width, 5 * THREAT_WIDTH)
        self.effect_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 5 * EFFECT_WIDTH)
        )
        self.null_projection = nn.Linear(width, width)
        self.injection_enabled = cfg["training"].get("forecasts", {}).get("injection_active", False)
        self._master_parameters = []
        self._master_names = []
        self._initialize_masters()
        self._set_runtime_dtype(torch.bfloat16)

    @property
    def species_encoder(self):
        return self.board_encoder

    @property
    def tile_encoder(self):
        return self.tile_decoder

    @property
    def encoders(self):
        return self.board_encoder, self.tile_decoder

    @staticmethod
    def _eligible(name, stage):
        if stage == "effect_pretrain":
            return name.startswith(
                ("board_encoder.", "tile_decoder.", "effect_head.", "null_projection.")
            )
        if stage == "tile_reach":
            return not name.startswith("species_head.")
        if stage == "tile":
            return not name.startswith(
                ("species_head.", "threat_head.", "effect_head.", "null_projection.")
            )
        return name.startswith(("board_encoder.", "species_head."))

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

    def master_parameters(self, stage=None):
        if stage is None:
            return tuple(self._master_parameters)
        if stage not in ("species", "tile", "tile_reach", "effect_pretrain"):
            raise ValueError("Parameter stage must be species, tile or tile_reach")
        return tuple(
            master
            for name, master in zip(self._master_names, self._master_parameters, strict=True)
            if self._eligible(name, stage)
        )

    def requires_grad_(self, requires_grad=True):
        super().requires_grad_(requires_grad)
        for master in self._master_parameters:
            master.requires_grad_(requires_grad)
        return self

    def runtime_parameters(self, stage=None):
        parameters = tuple(super().parameters())
        if stage is None:
            return parameters
        if stage not in ("species", "tile", "tile_reach", "effect_pretrain"):
            raise ValueError("Parameter stage must be species, tile or tile_reach")
        return tuple(
            parameter
            for name, parameter in zip(self._master_names, parameters, strict=True)
            if self._eligible(name, stage)
        )

    def master_state_dict(self):
        return {
            name: master.detach().clone()
            for name, master in zip(self._master_names, self._master_parameters, strict=True)
        }

    def runtime_state_dict(self):
        return super().state_dict()

    def checkpoint_state_dict(self):
        """Borrow authoritative tensors for immediate immutable CPU staging only."""
        state = super().state_dict()
        for name, master in zip(self._master_names, self._master_parameters, strict=True):
            if name in state:
                state[name] = master.detach()
        return state

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
        torch._foreach_copy_(list(super().parameters()), self._master_parameters)

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
        destinations, sources = [], []
        for master, parameter in zip(self._master_parameters, super().parameters(), strict=True):
            if parameter.grad is None:
                continue
            if master.grad is None:
                master.grad = parameter.grad.float()
            else:
                destinations.append(master.grad)
                sources.append(parameter.grad.float())
        if destinations:
            torch._foreach_add_(destinations, sources)

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

    def encode_board(self, observations, *, plan=None):
        observations = collate_observations(observations, self.device)
        memory = self._encode(self.board_encoder, observations, plan=plan)
        valid = torch.cat(
            (
                torch.ones((len(observations), 1), device=self.device, dtype=torch.bool),
                observations.entity_mask,
            ),
            1,
        )
        return memory, valid

    @contextmanager
    def board_context(self, observations, *, plan=None):
        # Explicit call-scoped memory, never an across-decision policy cache.
        prior = getattr(self, "_board_context", None)
        self._board_context = (observations, self.encode_board(observations, plan=plan))
        try:
            yield self._board_context[1]
        finally:
            self._board_context = prior

    def _memory(self, observations, plan=None):
        cached = getattr(self, "_board_context", None)
        if cached is not None and cached[0] is observations:
            return cached[1]
        return self.encode_board(observations, plan=plan)

    def species_values(self, observations, *, plan=None, memory=None):
        observations = collate_observations(observations, self.device)
        memory = self._memory(observations, plan) if memory is None else memory
        return self.action_values(memory)

    def threat_values(self, observations=None, *, memory=None):
        if memory is None:
            memory = self._memory(observations)
        return self.threat_head(memory[0][:, 0]).reshape(-1, 5, THREAT_WIDTH)

    def action_values(self, memory, *, rows=None):
        global_features = memory[0][:, 0] if rows is None else memory[0][rows, 0]
        logits = self.threat_head(global_features).reshape(-1, 5, THREAT_WIDTH)
        summaries = threat_summary(logits).detach()
        if not self.injection_enabled:
            summaries = torch.zeros_like(summaries)
        return self.species_head(
            torch.cat((global_features, summaries.to(global_features.dtype)), -1)
        )

    def effect_values(self, features):
        return self.effect_head(features).reshape(*features.shape[:-1], 5, EFFECT_WIDTH)

    def null_features(self, memory, branches=None):
        pooled = memory[0][:, 0]
        if branches is None:
            branches = torch.zeros(len(pooled), dtype=torch.long, device=pooled.device)
        return self.null_projection(pooled + self.tile_decoder.branch_embedding(branches))

    def tile_values(self, features, *, null=None):
        summaries = effect_summary(self.effect_values(features)).detach()
        if null is not None:
            summaries = summaries - effect_summary(self.effect_values(null)).detach()[:, None]
        if not self.injection_enabled:
            summaries = torch.zeros_like(summaries)
        return self.tile_head(torch.cat((features, summaries.to(features.dtype)), -1)).squeeze(-1)

    def tile_features(self, observations, branches, *, rows=None, plan=None, memory=None):
        observations = collate_observations(observations, self.device)
        branches = torch.as_tensor(branches, device=self.device, dtype=torch.long)
        rows = (
            torch.arange(len(branches), device=self.device)
            if rows is None
            else torch.as_tensor(rows, device=self.device, dtype=torch.long)
        )
        if branches.ndim != 1 or rows.shape != branches.shape:
            raise ValueError("Tile branches and rows must be matching vectors")
        memory = self._memory(observations, plan) if memory is None else memory
        return self.tile_decoder(memory[0][rows], memory[1][rows], branches)

    def forecast_display(self, memory):
        logits = self.threat_values(memory=memory)
        cells = penetration_map(logits[..., 22:33])
        # Viewers accept a common cell/home-logit carrier; these are derived raw
        # predictions, never clamped by public facts or training labels.
        home = logits[..., 39].float().sigmoid()[..., None]
        return torch.logit(torch.cat((cells, home), -1).clamp(1e-7, 1 - 1e-7))

    @staticmethod
    def reach_map(logits):
        return logits.float()[..., :9].sigmoid()

    @staticmethod
    def home_map(logits):
        return logits.float()[..., 9].sigmoid()

    def conditional_tiles(
        self, observations, branches, *, rows=None, plan=None, memory=None, return_reach=False
    ):
        observations = collate_observations(observations, self.device)
        branches = torch.as_tensor(branches, device=self.device, dtype=torch.long)
        rows = (
            torch.arange(len(branches), device=self.device)
            if rows is None
            else torch.as_tensor(rows, device=self.device, dtype=torch.long)
        )
        result = next(self.parameters()).new_zeros(len(branches), A.tiles)
        requested = torch.nonzero(branches > 0, as_tuple=False).flatten()
        reach = result.new_zeros((len(branches), 5, 10)) if return_reach else None
        if not len(requested):
            return (result, reach) if return_reach else result
        features = self.tile_features(
            observations, branches[requested], rows=rows[requested], plan=plan, memory=memory
        )
        memory = self._memory(observations, plan) if memory is None else memory
        null = self.null_features((memory[0][rows[requested]], memory[1][rows[requested]]))
        result = result.index_copy(0, requested, self.tile_values(features, null=null))
        if return_reach:
            reach = self.forecast_display((memory[0][rows], memory[1][rows]))
            return result, reach
        return result

    def fitting_tiles(self, observations, branches, *, rows):
        observations = collate_observations(observations, self.device)
        memory = self._memory(observations)
        features = self.tile_features(observations, branches, rows=rows, memory=memory)
        null = self.null_features((memory[0][rows], memory[1][rows]))
        return self.tile_values(features, null=null)

    def collection_tiles(
        self, observations, branches, *, plan=None, memory=None, return_reach=False
    ):
        return self.conditional_tiles(
            observations, branches, plan=plan, memory=memory, return_reach=return_reach
        )

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
        branch_usable=None,
        supplied_branches=None,
    ):
        observations = collate_observations(observations, self.device)
        memory = self.encode_board(observations, plan=active_plan)
        species = self.species_values(observations, memory=memory)
        if action_masks is None:
            action_masks = observation_tile_masks(observations)
        if action_masks.shape != (len(observations), A.size):
            raise ValueError("Action masks must match the board batch")
        branches = species.argmax(-1) if supplied_branches is None else supplied_branches
        if branches.shape != (len(observations),) or branches.dtype != torch.long:
            raise ValueError("Supplied branches must be an int64 vector matching the board batch")
        if supplied_branches is not None:
            branches = branches.to(device=self.device)
            supplied_valid = ((branches >= 0) & (branches < 10)).all()
            branches = branches.clamp(0, 9)
        if active_plan is not None and active is None:
            active = torch.as_tensor(active_plan.compact_indices >= 0, device=self.device)
        if active is not None:
            branches = torch.where(active, branches, 0)
        if branch_usable is None:
            branch_usable = torch.ones(
                (len(observations), 10), dtype=torch.bool, device=observations.device
            )
        else:
            branch_usable = torch.as_tensor(
                branch_usable, device=observations.device, dtype=torch.bool
            )
            if branch_usable.shape != (len(observations), 10):
                raise ValueError("Branch availability must have shape [batch, 10]")
        # Public plant count includes all plants, even when entity truncation
        # drops a record. Also enforce the branch check for weight-only callers
        # that have no simulator availability tensor. Never mutate a borrowed one.
        branch_usable = branch_usable.clone()
        branch_usable[:, -1] = dig_branch_available(
            observations.globals[:, self.layout.global_fields["plant_count"]],
            branch_usable[:, -1],
        )
        selected_usable = branch_usable.gather(1, branches[:, None]).flatten()
        tile_rows = (branches > 0) & selected_usable
        requests = torch.where(tile_rows, branches, torch.zeros_like(branches))
        tiles, reach = self.collection_tiles(
            observations, requests, plan=active_plan, memory=memory, return_reach=True
        )
        actions, _, _, details = select_q_actions(
            species,
            tiles,
            action_masks,
            deterministic=deterministic,
            tile_exploration_epsilon=getattr(self, "tile_exploration_epsilon", 0.0),
            active=active,
            supplied_branches=branches if supplied_branches is not None else None,
        )
        details["branch_usable"] = branch_usable
        if supplied_branches is not None:
            details["valid"] &= supplied_valid
        details["tile_inference_skipped"] = (branches > 0) & ~tile_rows
        details["observations"] = observations
        details["board_memory"] = memory
        details["reach_logits"] = self.forecast_display(memory)
        details["threat_logits"] = self.threat_values(memory=memory)
        return actions, details
