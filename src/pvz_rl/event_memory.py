"""Bounded causal public history, independent of all learned parameters.

Recent tokens retain every decision. Evicted event tokens enter a FIFO; quiet
tokens and old events become run summaries (latest public state, span and count).
No simulator entity IDs, countdowns, rewards or future schedules are consulted.
"""

from dataclasses import dataclass

import torch

from .encoding import ObservationEncoder


@dataclass
class MemoryContext:
    tokens: torch.Tensor
    valid: torch.Tensor
    counts: torch.Tensor
    starts: torch.Tensor

    def select(self, indices):
        return MemoryContext(*(getattr(self, k)[indices] for k in self.__dataclass_fields__))

    def clone(self):
        return MemoryContext(*(getattr(self, k).clone() for k in self.__dataclass_fields__))


class EventMemory:
    def __init__(self, cfg, rules, n, device):
        self.cfg, self.n = cfg, n
        self.layout = ObservationEncoder(cfg, rules)
        spec = cfg["policy"]["memory"]
        self.local, self.events, self.summaries = (
            spec[k] for k in ("local_tokens", "event_tokens", "summary_tokens")
        )
        self.capacity = self.local + self.events + self.summaries
        self.stride = spec["summary_stride"]
        self.width = self.layout.size + 3  # previous executed action, reset, public tick
        self.tokens = torch.zeros(n, self.capacity, self.width, device=device)
        self.valid = torch.zeros(n, self.capacity, dtype=torch.bool, device=device)
        self.counts = torch.zeros(n, self.capacity, device=device)
        self.starts = torch.zeros(n, self.capacity, device=device)
        self.ids = torch.zeros(n, self.capacity, dtype=torch.long, device=device)
        self.event_flags = torch.zeros(n, self.local, dtype=torch.bool, device=device)
        self.last_obs = torch.zeros(n, self.layout.size, device=device)
        self.last_mask = torch.zeros(n, 406, dtype=torch.bool, device=device)
        self.have_previous = torch.zeros(n, dtype=torch.bool, device=device)
        # Ignore continuous clock/movement for history admission, never for decisions.
        relevant = torch.ones(self.layout.size, dtype=torch.bool, device=device)
        zs, gs = self.layout.slices["zombies"], self.layout.slices["globals"]
        nearest_zombie_offsets = torch.arange(
            zs.start + self.layout.zombie_fields["nearest"],
            zs.stop,
            self.layout.zombie_width,
            device=device,
        )
        self.nearest_pole_offsets = nearest_zombie_offsets + 1
        relevant[nearest_zombie_offsets] = False
        relevant[self.nearest_pole_offsets] = False
        relevant[gs.start + self.layout.global_fields["elapsed"]] = False
        self.relevant = relevant
        self.admitted = self.observed = 0

    def context(self):
        return MemoryContext(self.tokens, self.valid, self.counts, self.starts)

    def _push(self, start, end, token, token_id, active, count, first):
        for field, incoming in (
            (self.tokens, token),
            (self.ids, token_id),
            (self.valid, active),
            (self.counts, count),
            (self.starts, first),
        ):
            old = field[:, start:end]
            shifted = torch.cat((old[:, 1:], incoming[:, None]), dim=1)
            selector = active.reshape(self.n, *([1] * (field.ndim - 1)))
            field[:, start:end] = torch.where(selector, shifted, old)

    def _compress(self, token, token_id, active, count, first):
        start, end = self.local + self.events, self.capacity
        full = active & (~self.valid[:, -1] | (self.counts[:, -1] >= self.stride))
        self._push(start, end, token, token_id, full, count, first)
        merge = active & ~full
        latest = merge & (token[:, -1] >= self.tokens[:, -1, -1])
        self.tokens[:, -1] = torch.where(latest[:, None], token, self.tokens[:, -1])
        self.ids[:, -1] = torch.where(latest, token_id, self.ids[:, -1])
        self.counts[:, -1] += torch.where(merge, count, 0)
        self.starts[:, -1] = torch.where(
            merge, torch.minimum(first, self.starts[:, -1]), self.starts[:, -1]
        )

    @torch.no_grad()
    def observe(self, obs, masks, previous_actions, resets, ticks, token_ids=None):
        resets = resets.bool()
        self.valid[resets] = False
        self.counts[resets] = 0
        self.event_flags[resets] = False
        self.have_previous[resets] = False
        changed = (
            (obs[:, self.relevant] != self.last_obs[:, self.relevant]).any(-1)
            | (masks != self.last_mask).any(-1)
            | (previous_actions > 0)
            | ~self.have_previous
        )
        # Distance changes caused by ordinary movement stay quiet, but a region
        # entering or leaving the active-pole set is a meaningful public event.
        pole_presence_changed = self.have_previous & (
            (obs[:, self.nearest_pole_offsets] != self.layout.empty_distance)
            != (self.last_obs[:, self.nearest_pole_offsets] != self.layout.empty_distance)
        ).any(-1)
        changed |= pole_presence_changed
        token = torch.cat(
            (
                obs,
                previous_actions[:, None].float(),
                resets[:, None].float(),
                ticks[:, None].float(),
            ),
            -1,
        )
        if token_ids is None:
            token_ids = torch.zeros(self.n, dtype=torch.long, device=obs.device)
        evict = self.valid[:, 0]
        event = evict & self.event_flags[:, 0]
        # Copy before shifting any storage that contains the source token.
        old, old_id = self.tokens[:, 0].clone(), self.ids[:, 0].clone()
        old_count, old_first = self.counts[:, 0].clone(), self.starts[:, 0].clone()
        e = self.local
        self._compress(
            self.tokens[:, e].clone(),
            self.ids[:, e].clone(),
            event & self.valid[:, e],
            self.counts[:, e].clone(),
            self.starts[:, e].clone(),
        )
        self._push(e, e + self.events, old, old_id, event, old_count, old_first)
        self._compress(old, old_id, evict & ~event, old_count, old_first)
        self._push(
            0, self.local, token, token_ids, torch.ones_like(changed), torch.ones_like(ticks), ticks
        )
        self.event_flags = torch.cat((self.event_flags[:, 1:], changed[:, None]), dim=1)
        self.last_obs.copy_(obs)
        self.last_mask.copy_(masks)
        self.have_previous.fill_(True)
        self.observed += self.n
        # Keep device counters; diagnostic transfers occur only at rollout boundaries.
        self.admitted = self.admitted + changed.sum()
        return self.context()

    def snapshot(self):
        return {
            k: getattr(self, k).clone()
            for k in (
                "tokens",
                "valid",
                "counts",
                "starts",
                "ids",
                "event_flags",
                "last_obs",
                "last_mask",
                "have_previous",
            )
        }

    def restore(self, state):
        for key, value in state.items():
            getattr(self, key).copy_(value)

    def fork(self):
        import copy

        other = copy.copy(self)
        for key, value in self.snapshot().items():
            setattr(other, key, value)
        return other
