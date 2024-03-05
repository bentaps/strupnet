from torch import nn
from ..nn.scalar import ScalarNet

class Layer(nn.Module):
    """Gradient symplectic module."""

    def __init__(self, dim, width, mode, activation, **kwargs):
        super().__init__()
        self.dim = dim
        self.width = width
        self.mode = mode
        self.hamiltonian = ScalarNet(dim, width, activation=activation)
        self.init_params()

    def forward(self, p, q, h, **kwargs):
        if self.mode == "odd":
            return p + self.hamiltonian.grad(q) * h, q
        elif self.mode == "even":
            return p, self.hamiltonian.grad(p) * h + q
        else:
            raise ValueError

    def init_params(self):
        self.hamiltonian.init_params()