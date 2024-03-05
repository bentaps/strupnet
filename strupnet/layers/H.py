import torch
from torch import nn
from ..nn.scalar import ScalarNet

class Layer(nn.Module):
    """Henon-like map layer."""

    def __init__(self, dim, width, activation=None, **kwargs):
        super().__init__()
        self.dim = dim
        self.hamiltonian = ScalarNet(dim, width, activation=activation)
        self.init_params()

    def forward(self, p, q, h, reverse=False):
        if reverse:
            h = -h
            return  - q + h * self.hamiltonian.grad(p), p
        else:        
            return q, -p + h *self.hamiltonian.grad(q)

    def init_params(self):
        self.hamiltonian.init_params()
