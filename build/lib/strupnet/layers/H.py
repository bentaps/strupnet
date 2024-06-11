import torch
from torch import nn
from ..utils import get_pq, get_x
from ..nn.scalar import ScalarNet


class Layer(nn.Module):
    """Henon-like map layer."""

    def __init__(self, dim, width, activation=None, **kwargs):
        super().__init__()
        self.dim = dim
        self.hamiltonian = ScalarNet(dim, width, activation=activation)

    def forward(self, x, h, reverse=False):
        p, q = get_pq(x)
        if reverse:
            h = -h
            p, q = -q + h * self.hamiltonian.grad(p), p
        else:
            p, q = q, -p + h * self.hamiltonian.grad(q)
        return get_x(p, q)