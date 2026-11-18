"""Shared dense readouts and transformer feed-forward networks."""

from torch import nn


def build_mlp(input_width, hidden_width, output_width, hidden_layers, *, activation=nn.GELU):
    """Use one activation per hidden layer and leave the output unrestricted."""
    dimensions = (input_width, hidden_width, output_width, hidden_layers)
    if any(type(value) is not int or value < 1 for value in dimensions):
        raise ValueError("MLP widths and hidden-layer count must be positive integers")
    layers = []
    width = input_width
    for _ in range(hidden_layers):
        layers.extend((nn.Linear(width, hidden_width), activation()))
        width = hidden_width
    layers.append(nn.Linear(width, output_width))
    return nn.Sequential(*layers)
