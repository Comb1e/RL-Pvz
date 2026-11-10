"""Shared public-board encoding with action and tile decoders and outcome readouts."""

from contextlib import contextmanager

import torch
from pvz_game import Rules
from torch import nn

from pvz_rl.config import absolute_outcomes
from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.envs.outcome_math import ABSOLUTE_SIZES
from pvz_rl.learning.forecast_labels import EFFECT_WIDTH, effect_forecast
from pvz_rl.policy.action_decoder import ActionDecoder
from pvz_rl.policy.active_batch import encode_active
from pvz_rl.policy.effect_head import EffectHead
from pvz_rl.policy.entity_attention import SpatialTransformer
from pvz_rl.policy.forecast_input import ForecastInput, effect_classes
from pvz_rl.policy.sequential_q import (
    POLICY_SIGNATURE,
    action_branch_mask,
    action_parts,
    card_ready,
    greedy_choice,
    observation_tile_masks,
    occupied_cells,
    select_q_actions,
)
from pvz_rl.policy.tile_decoder import TileDecoder


def execution_scope_for(phase):
    """Forward modules a curriculum phase is allowed to run."""
    from pvz_rl.envs.tile_scenarios import is_placement_skill

    if phase == "prediction":
        return "forecast"
    if phase == "peashooter" or is_placement_skill(phase):
        return "placement"
    if phase in ("easy", "standard", "shared"):
        return "full"
    raise ValueError(f"No execution scope for phase {phase!r}")


class TwoStageTransformerPolicy(nn.Module):
    protocol = POLICY_SIGNATURE

    def __init__(self, cfg):
        super().__init__()
        from pvz_rl.envs.encoding import ObservationEncoder

        self.cfg = cfg
        self.execution_scope = "full"
        self.protocol = cfg["policy"]["kind"]
        self.absolute_outcomes = absolute_outcomes(cfg)
        self.effect_width = EFFECT_WIDTH
        self.effect_sizes = ABSOLUTE_SIZES
        self.layout = ObservationEncoder(cfg, Rules())
        self.fit_precision = "bf16_master"
        spec = cfg["policy"]
        self.board_encoder = SpatialTransformer(self.layout, spec, "board")
        self.tile_decoder = TileDecoder(self.layout, spec)
        self.action_decoder = ActionDecoder(self.layout, spec)
        board_width = spec["board"]["width"]
        action_width = spec["action"]["width"]
        tile_width = spec["tile"]["width"]
        self.forecast_input = ForecastInput(board_width)
        self.species_head = nn.Sequential(
            nn.Linear(action_width, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.species_bias = nn.Parameter(torch.zeros(self.action_decoder.token_overhead))
        self.tile_head = nn.Sequential(nn.Linear(tile_width, 128), nn.ReLU(), nn.Linear(128, 1))
        self.effect_head = EffectHead(
            spec["effect"]["width"],
            board_width,
            self.effect_width,
            spec["effect"]["heads"],
            spec["execution"]["query_chunk"],
        )
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

    encoder_names = ("species", "tile", "action")

    @property
    def encoders(self):
        return self.board_encoder, self.tile_decoder, self.action_decoder

    _FITTING_STAGES = (
        "species",
        "tile",
        "tile_outcomes",
        "placement_outcomes",
        "prediction_only",
        "action_heads",
        "easy_action",
    )

    @staticmethod
    def _eligible(name, stage):
        """Prediction owns the encoder, effect head, and forecast embeddings.

        Later stages leave those three fixed. Placement and Peashooter train the
        tile decoder and tile head. Easy, standard, and shared add the action
        decoder and action head.
        """
        board = name.startswith("board_encoder.")
        action_decoder = name.startswith("action_decoder.")
        tile_decoder = name.startswith("tile_decoder.")
        action_head = name.startswith("species_head.") or name == "species_bias"
        tile_head = name.startswith("tile_head.")
        effect_head = name.startswith("effect_head.")
        forecast_input = name.startswith("forecast_input.")
        if stage == "prediction_only":
            return board or effect_head or forecast_input
        if stage == "species":
            return action_decoder or action_head
        if stage in ("tile", "tile_outcomes", "placement_outcomes"):
            return tile_decoder or tile_head
        if stage in ("easy_action", "action_heads"):
            return action_decoder or tile_decoder or action_head or tile_head
        return False

    def _require_stage(self, stage):
        if stage not in self._FITTING_STAGES:
            raise ValueError("Parameter stage must be species, tile or outcome fitting")

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
        self._require_stage(stage)
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
        self._require_stage(stage)
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
    def accumulate_master_gradients(self, stage=None):
        eligible = None
        if stage is not None:
            self._require_stage(stage)
            eligible = {
                id(parameter)
                for name, parameter in zip(self._master_names, super().parameters(), strict=True)
                if self._eligible(name, stage)
            }
        destinations, sources = [], []
        for master, parameter in zip(self._master_parameters, super().parameters(), strict=True):
            if parameter.grad is None:
                continue
            if eligible is not None and id(parameter) not in eligible:
                continue
            if master.grad is None:
                master.grad = parameter.grad.float()
            else:
                destinations.append(master.grad)
                sources.append(parameter.grad.float())
        if destinations:
            torch._foreach_add_(destinations, sources)

    @torch.no_grad()
    def update_teacher_runtime(self, teacher, stage, decay):
        """Lerp trained runtime weights and copy frozen ones onto the teacher."""
        if stage is None:
            torch._foreach_lerp_(
                list(teacher.runtime_parameters()), list(self.runtime_parameters()), 1 - decay
            )
            return
        self._require_stage(stage)
        trained_teacher, trained_online = [], []
        frozen_teacher, frozen_online = [], []
        for name, teacher_parameter, online_parameter in zip(
            self._master_names,
            teacher.runtime_parameters(),
            self.runtime_parameters(),
            strict=True,
        ):
            if self._eligible(name, stage):
                trained_teacher.append(teacher_parameter)
                trained_online.append(online_parameter)
            else:
                frozen_teacher.append(teacher_parameter)
                frozen_online.append(online_parameter)
        if trained_teacher:
            torch._foreach_lerp_(trained_teacher, trained_online, 1 - decay)
        if frozen_teacher:
            torch._foreach_copy_(frozen_teacher, frozen_online)

    @property
    def runtime_dtype(self):
        return torch.bfloat16

    @contextmanager
    def fitting_precision(self, precision):
        if precision != "bf16_master":
            raise ValueError("Only bf16_master fitting is supported")
        yield

    def _encode(self, encoder, observations, branches=None, plan=None):
        if plan is None:
            return encoder(observations, branches)
        return encode_active(encoder, observations, plan, branches)

    def encode_board(self, observations, *, plan=None):
        """Encoder memory, then the pre-action forecast, then decoder memory.

        Decoder memory is the global token, five forecast tokens, and the
        entities. The effect head reads only the encoder memory.
        """
        observations = collate_observations(observations, self.device)
        encoded = self._encode(self.board_encoder, observations, plan=plan)
        encoded_valid = torch.cat(
            (
                torch.ones((len(observations), 1), device=self.device, dtype=torch.bool),
                observations.entity_mask,
            ),
            1,
        )
        logits = self.effect_head(encoded, encoded_valid)
        tokens, token_valid = self.forecast_input.tokens(effect_classes(logits))
        memory = torch.cat((encoded[:, :1], tokens, encoded[:, 1:]), 1)
        valid = torch.cat(
            (
                torch.ones((len(observations), 1), device=self.device, dtype=torch.bool),
                token_valid[:, None].expand(-1, 5),
                observations.entity_mask,
            ),
            1,
        )
        self._effect_logits = logits
        return memory, valid

    @contextmanager
    def board_context(self, observations, *, plan=None):
        # Explicit call-scoped memory, never an across-decision policy cache.
        saved = getattr(self, "_board_context", None)
        self._board_context = (observations, self.encode_board(observations, plan=plan))
        try:
            yield self._board_context[1]
        finally:
            self._board_context = saved

    def _memory(self, observations, plan=None):
        cached = getattr(self, "_board_context", None)
        if cached is not None and cached[0] is observations:
            return cached[1]
        return self.encode_board(observations, plan=plan)

    def species_values(self, observations, *, plan=None, memory=None):
        observations = collate_observations(observations, self.device)
        memory = self._memory(observations, plan) if memory is None else memory
        return self.action_values(memory)

    def _require_full_scope(self, module):
        if self.execution_scope != "full":
            raise RuntimeError(f"The {module} runs only for easy, standard, and shared")

    def _require_tile_scope(self, module):
        if self.execution_scope == "forecast":
            raise RuntimeError(f"The {module} stays uncalled during prediction")

    def action_vectors(self, memory, *, rows=None):
        """Ten action-decoder vectors. A full-batch call is reused within this forward."""
        self._require_full_scope("action decoder")
        tokens, valid = memory
        if rows is not None:
            tokens, valid = tokens[rows], valid[rows]
        vectors = self.action_decoder(tokens, valid)
        if rows is None:
            self._decision_action_vectors = vectors
            self._decision_action_memory = tokens
        return vectors

    def action_values(self, memory, *, rows=None):
        self._require_full_scope("action head")
        scores = self.species_head(self.action_vectors(memory, rows=rows)).squeeze(-1)
        return scores + self.species_bias

    def _action_vectors(self, memory):
        cached = getattr(self, "_decision_action_vectors", None)
        if cached is not None and getattr(self, "_decision_action_memory", None) is memory[0]:
            return cached
        return self.action_vectors(memory)

    def wait_features(self, memory):
        """Slot 0 of decoder memory, the global token."""
        return memory[0][:, 0]

    def tile_values(self, features):
        self._require_tile_scope("tile head")
        return self.tile_head(features).squeeze(-1)

    def tile_features(self, observations, branches, *, rows=None, plan=None, memory=None):
        self._require_tile_scope("tile decoder")
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

    def conditional_tiles(self, observations, branches, *, rows=None, plan=None, memory=None):
        observations = collate_observations(observations, self.device)
        branches = torch.as_tensor(branches, device=self.device, dtype=torch.long)
        rows = (
            torch.arange(len(branches), device=self.device)
            if rows is None
            else torch.as_tensor(rows, device=self.device, dtype=torch.long)
        )
        result = next(self.parameters()).new_zeros(len(branches), A.tiles)
        requested = torch.nonzero(branches > 0, as_tuple=False).flatten()
        if not len(requested):
            return result
        features = self.tile_features(
            observations, branches[requested], rows=rows[requested], plan=plan, memory=memory
        )
        result = result.index_copy(0, requested, self.tile_values(features))
        return result

    def fitting_tiles(self, observations, branches, *, rows):
        observations = collate_observations(observations, self.device)
        memory = self._memory(observations)
        features = self.tile_features(observations, branches, rows=rows, memory=memory)
        return self.tile_values(features)

    def collection_tiles(self, observations, branches, *, plan=None, memory=None):
        return self.conditional_tiles(observations, branches, plan=plan, memory=memory)

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

    def _selection_mask(self, observations, branch_usable):
        """Hard branch mask from planted cells, sun, cooldown, and a passed card ban."""
        layout = self.layout
        occupied = occupied_cells(
            observations.entities,
            observations.entity_mask,
            rows=layout.rows,
            cols=layout.cols,
            units_per_tile=layout.rules.game["units_per_tile"],
        )
        fields = layout.global_fields
        passed = None
        dig_permitted = torch.ones(len(observations), dtype=torch.bool, device=observations.device)
        if branch_usable is None:
            costs = observations.globals.new_tensor(
                [layout.rules.plants[name]["cost"] for name in self.cfg["environment"]["plants"]]
            )
            cards = card_ready(
                observations.globals,
                sun_index=fields["sun"],
                cooldown_index=[fields[f"cooldown_{i}"] for i in range(A.plant_types)],
                cost_scale=layout.cost_scale,
                costs=costs,
            )
        else:
            passed = torch.as_tensor(branch_usable, device=observations.device, dtype=torch.bool)
            if passed.shape != (len(observations), 10):
                raise ValueError("Branch availability must have shape [batch, 10]")
            cards = passed[:, 1:-1]
            dig_permitted = passed[:, -1]
        return occupied, action_branch_mask(occupied, cards, dig_permitted), passed

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
        if self.execution_scope == "forecast":
            return self._forecast_decision(observations, memory)
        if self.execution_scope == "placement":
            if supplied_branches is None:
                raise RuntimeError("Placement decisions require the exercise branch")
            species = torch.zeros(
                (len(observations), 10),
                device=self.device,
                dtype=next(self.parameters()).dtype,
            )
            self._decision_action_vectors = None
            self._decision_action_memory = None
        elif self.execution_scope == "full":
            species = self.species_values(observations, memory=memory)
        else:
            raise RuntimeError(f"Unknown execution scope {self.execution_scope!r}")
        if action_masks is None:
            action_masks = observation_tile_masks(observations)
        if action_masks.shape != (len(observations), A.size):
            raise ValueError("Action masks must match the board batch")
        occupied, branch_mask, passed_usable = self._selection_mask(observations, branch_usable)
        if supplied_branches is None:
            branches = greedy_choice(species, branch_mask)
            supplied_valid = True
        else:
            branches = torch.as_tensor(supplied_branches, device=self.device, dtype=torch.long)
            if branches.shape != (len(observations),):
                raise ValueError(
                    "Supplied branches must be an int64 vector matching the board batch"
                )
            supplied_valid = bool(((branches >= 0) & (branches < 10)).all())
            branches = branches.clamp(0, 9)
        if active_plan is not None and active is None:
            active = torch.as_tensor(active_plan.compact_indices >= 0, device=self.device)
        if active is not None:
            branches = torch.where(active, branches, 0)
        playable = branch_mask.gather(1, branches[:, None]).flatten()
        tile_rows = (branches > 0) & playable
        requests = torch.where(tile_rows, branches, torch.zeros_like(branches))
        tiles = self.collection_tiles(observations, requests, plan=active_plan, memory=memory)
        actions, _, _, details = select_q_actions(
            species,
            tiles,
            action_masks,
            occupied,
            branch_mask,
            deterministic=deterministic,
            tile_exploration_epsilon=getattr(self, "tile_exploration_epsilon", 0.0),
            active=active,
            supplied_branches=branches,
        )
        details["occupied"] = occupied
        details["branch_usable"] = passed_usable if passed_usable is not None else branch_mask
        details["valid"] &= supplied_valid
        details["tile_inference_skipped"] = (branches > 0) & ~tile_rows
        details["observations"] = observations
        details["board_memory"] = memory
        details["action_vectors"] = self._decision_action_vectors
        details["effect_logits"] = self._effect_logits
        return actions, details

    def _forecast_decision(self, observations, memory):
        """Wait, with the pre-action logits and no action or tile forward."""
        count = len(observations)
        blank = next(self.parameters()).new_zeros(count)
        actions = torch.zeros(count, device=self.device, dtype=torch.long)
        self._decision_action_vectors = None
        self._decision_action_memory = None
        return actions, dict(
            greedy_actions=actions,
            coins=torch.zeros(count, 2, dtype=torch.bool, device=self.device),
            branch_q=blank.new_zeros(count, 10),
            tile_q=blank.new_zeros(count, A.tiles),
            branch_value=blank,
            tile_value=blank,
            selected_value=blank,
            placement_gap=blank,
            valid=torch.ones((), dtype=torch.bool, device=self.device),
            occupied=torch.zeros(count, A.tiles, dtype=torch.bool, device=self.device),
            branch_usable=None,
            tile_inference_skipped=torch.zeros(count, dtype=torch.bool, device=self.device),
            observations=observations,
            board_memory=memory,
            action_vectors=None,
            effect_logits=self._effect_logits,
        )

    def effect_action_display(self, actions=None, memory=None):
        """Pre-action field and probe steps from the logits of this encode."""
        del actions, memory
        return effect_forecast(self._effect_logits)
