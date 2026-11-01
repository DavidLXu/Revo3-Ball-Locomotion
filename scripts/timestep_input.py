"""Preserve the original input projection while learning one timestep column."""
import torch
from torch import nn
from torch.nn import functional as F


class SplitTimestepLinear(nn.Linear):
    """Keep checkpoint/Adam shapes; evaluate the old 124 inputs separately.

    A zero column in a 125-wide GEMV can still change rounding of the original
    dot product. Contiguous 124-wide operands preserve its previous arithmetic.
    The extra term remains differentiable even when its weights are zero.
    """

    @classmethod
    def from_linear(cls, layer):
        assert type(layer) is nn.Linear and layer.in_features == 125
        replacement = cls.__new__(cls)
        nn.Module.__init__(replacement)
        replacement.in_features = layer.in_features
        replacement.out_features = layer.out_features
        replacement.weight = layer.weight
        replacement.bias = layer.bias
        replacement.train(layer.training)
        return replacement

    def forward(self, inputs):
        base = F.linear(inputs[..., :124].contiguous(),
                        self.weight[:, :124].contiguous(), self.bias)
        return base + inputs[..., 124:] * self.weight[:, 124]


def install_timestep_projection(model):
    """Retain Parameter identity, optimizer references and RNG state."""
    layer = model.mlp[0]
    if isinstance(layer, SplitTimestepLinear):
        return
    model.mlp[0] = SplitTimestepLinear.from_linear(layer)
