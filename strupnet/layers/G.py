import numpy as np
import sympy as sp
from torch import nn
from ..utils import get_pq, get_x, get_parameters
from ..nn.scalar import ScalarNet

class Layer(nn.Module):
    """Gradient symplectic module."""

    def __init__(self, dim, width, mode, activation, **kwargs):
        super().__init__()
        self.dim = dim
        self.width = width
        self.activation = nn.Tanh()
        
        self.params = nn.ParameterDict()
        self.params['K'] = get_parameters(dim, width)
        self.params['a'] = get_parameters(self.width)
        self.params['b'] = get_parameters(self.width)

        self.mode = mode
    
    def forward(self, x, h, **kwargs):
        p, q = get_pq(x)
        if self.mode == "odd":
            p = p - h * (self.activation(q @ self.params['K'] + self.params['b']) * self.params['a']) @ self.params['K'].t()
        elif self.mode == "even":
            q = q + h * (self.activation(p @ self.params['K'] + self.params['b']) * self.params['a']) @ self.params['K'].t()
        else:
            raise ValueError("attribute mode must be either 'odd' or 'even'")
        return get_x(p, q)

    def hamiltonian(self, x):
        """use the shear Hamiltonian of p or q to create a hamiltonian of p and q depinding on mode"""
        p, q = x[: self.dim], x[self.dim :]
        integral_tanh = lambda x:  np.array([sp.log(sp.cosh(x[i]) ** 2) for i in range(len(x))])
        K = self.params['K'].detach().numpy()
        b = self.params['b'].detach().numpy()
        a = self.params['a'].detach().numpy()
        if self.mode == "odd":
            return integral_tanh(q @ K + b) @ a
        elif self.mode == "even":
            return integral_tanh(p @ K + b) @ a