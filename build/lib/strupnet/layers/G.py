from torch import nn
from ..utils import get_pq, get_x
from ..nn.scalar import ScalarNet

class Layer(nn.Module):
    """Gradient symplectic module."""

    def __init__(self, dim, width, mode, activation, **kwargs):
        super().__init__()
        self.dim = dim
        self.width = width
        self.mode = mode
        self.hamiltonian = ScalarNet(dim, width, activation=activation)

    def forward(self, x, h, **kwargs):
        p, q = get_pq(x)
        if self.mode == "odd":
            p = p + self.hamiltonian.grad(q) * h
        elif self.mode == "even":
            q = q + self.hamiltonian.grad(p) * h
        else:
            raise ValueError("attribute mode must be either 'odd' or 'even'")
        return get_x(p, q)