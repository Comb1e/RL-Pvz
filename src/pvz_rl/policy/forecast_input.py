"""Detached effect-head decisions carried into the next board encoding."""

import numpy as np
import torch
from torch import nn

from pvz_rl.learning.forecast_labels import DEPTH_CLASSES, TIME_CLASSES, effect_blocks

FORECAST_LANES = 5
FORECAST_ITEMS = 3


def effect_classes(logits):
    """Detached field, breach, and probe class ids. Shape [..., 5, 3]."""
    classes = torch.stack([block.argmax(-1) for block in effect_blocks(logits)], -1)
    if classes.shape[-2:] != (FORECAST_LANES, FORECAST_ITEMS):
        raise ValueError("Effect classes must be five lanes of field, breach, and probe")
    return classes.detach()


def prior_from_rows(rows, device):
    """Stored prior that this decision saw. Missing history stays masked."""
    classes = torch.as_tensor(np.asarray(rows["prior_classes"]), device=device, dtype=torch.long)
    valid = torch.as_tensor(np.asarray(rows["prior_valid"]), device=device, dtype=torch.bool)
    if classes.shape != (len(rows), FORECAST_LANES, FORECAST_ITEMS) or valid.shape != (len(rows),):
        raise ValueError("Stored forecast prior must be [rows, 5, 3] with one valid bit")
    return classes, valid


def played_logits(policy, actions, memory, active):
    """Forecast the played rows without replacing the full-batch action cache."""
    index = active.nonzero(as_tuple=False).flatten()
    if not len(index):
        return index, None
    vectors = policy._action_vectors(memory).index_select(0, index)
    subset = (memory[0].index_select(0, index), memory[1].index_select(0, index))
    return index, policy.forecast_logits(vectors, actions.index_select(0, index), subset)


class ForecastCarry:
    """Per-env class ids for the decision that has not run yet."""

    def __init__(self, count, device):
        self.classes = torch.zeros(
            count, FORECAST_LANES, FORECAST_ITEMS, dtype=torch.long, device=device
        )
        self.valid = torch.zeros(count, dtype=torch.bool, device=device)

    def __len__(self):
        return self.valid.shape[0]

    def as_prior(self):
        return self.classes, self.valid

    def clear_indices(self, indices):
        index = torch.as_tensor(list(indices), device=self.valid.device, dtype=torch.long)
        if not len(index):
            return
        self.valid[index] = False
        self.classes[index] = 0

    def clear_mask(self, mask):
        mask = torch.as_tensor(mask, device=self.valid.device, dtype=torch.bool).reshape(-1)
        if mask.shape != self.valid.shape:
            raise ValueError("Forecast reset mask must match the carry")
        self.valid.masked_fill_(mask, False)
        self.classes.masked_fill_(mask[:, None, None], 0)

    def record(self, policy, actions, memory, active):
        index, logits = played_logits(policy, actions, memory, active)
        if len(index):
            self.classes[index] = effect_classes(logits)
            self.valid[index] = True
        return logits


class ForecastInput(nn.Module):
    """Five lane tokens: lane identity plus the three detached class decisions."""

    def __init__(self, width):
        super().__init__()
        self.lanes = nn.Embedding(FORECAST_LANES, width)
        self.field = nn.Embedding(DEPTH_CLASSES, width)
        self.breach = nn.Embedding(TIME_CLASSES, width)
        self.probe = nn.Embedding(DEPTH_CLASSES, width)

    def tokens(self, prior, batch, device):
        """Return ``[batch, 5, width]`` tokens and a per-board valid bit.

        An absent prior is masked and zero. Class 0 is a real decision, so a
        valid prior of zeros is not the same token.
        """
        if prior is None:
            classes = torch.zeros(
                batch, FORECAST_LANES, FORECAST_ITEMS, dtype=torch.long, device=device
            )
            valid = torch.zeros(batch, dtype=torch.bool, device=device)
        else:
            classes, valid = prior
            classes = torch.as_tensor(classes, device=device, dtype=torch.long).detach()
            valid = torch.as_tensor(valid, device=device, dtype=torch.bool)
            if classes.shape != (batch, FORECAST_LANES, FORECAST_ITEMS) or valid.shape != (batch,):
                raise ValueError("Forecast prior must be [batch, 5, 3] classes and [batch] valid")
        lane = torch.arange(FORECAST_LANES, device=device)
        embedded = (
            self.lanes(lane)[None]
            + self.field(classes[..., 0])
            + self.breach(classes[..., 1])
            + self.probe(classes[..., 2])
        )
        return embedded.masked_fill(~valid[:, None, None], 0), valid
