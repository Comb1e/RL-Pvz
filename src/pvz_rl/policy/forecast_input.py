"""Detached pre-action class decisions embedded for the action and tile decoders."""

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


class ForecastInput(nn.Module):
    """Five lane tokens: lane identity plus the three detached class decisions."""

    def __init__(self, width):
        super().__init__()
        self.lanes = nn.Embedding(FORECAST_LANES, width)
        self.field = nn.Embedding(DEPTH_CLASSES, width)
        self.breach = nn.Embedding(TIME_CLASSES, width)
        self.probe = nn.Embedding(DEPTH_CLASSES, width)

    def tokens(self, classes):
        """Return ``[batch, 5, width]`` tokens and a true valid bit for every row.

        Class 0 is a real decision. The ids are detached, so decoder losses do
        not train the effect head that produced them.
        """
        classes = torch.as_tensor(classes, device=self.lanes.weight.device, dtype=torch.long)
        classes = classes.detach()
        if classes.ndim != 3 or classes.shape[1:] != (FORECAST_LANES, FORECAST_ITEMS):
            raise ValueError("Forecast classes must be [batch, 5, 3]")
        lane = torch.arange(FORECAST_LANES, device=classes.device)
        embedded = (
            self.lanes(lane)[None]
            + self.field(classes[..., 0])
            + self.breach(classes[..., 1])
            + self.probe(classes[..., 2])
        )
        valid = torch.ones(classes.shape[0], dtype=torch.bool, device=classes.device)
        return embedded, valid
