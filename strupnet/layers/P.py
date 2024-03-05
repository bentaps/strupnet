import torch
from torch import nn
from ..utils import get_parameters


class Layer(nn.Module):
    def __init__(self, dim, min_degree=2, max_degree=4, **kwargs):
        super().__init__()
        self.dim = dim
        self.max_degree = max_degree
        self.min_degree = min_degree
        self.init_params()

    def forward(self, p, q, h, **kwargs):
        monomial = (p @ (self.params["A"]) + q @ (self.params["B"])).unsqueeze(-1)
        dpolynomial = sum(
            i * self.params["a"][i - self.min_degree] * torch.pow(monomial, i - 1)
            for i in range(self.min_degree, self.max_degree + 1)
        )  # compute polynomial: sum a[i] * x.T @ A^i
        p = p - h * dpolynomial * self.params["B"]
        q = q + h * dpolynomial * self.params["A"]
        return p, q

    def init_params(self):
        params = nn.ParameterDict()
        params["a"] = get_parameters(self.max_degree - self.min_degree + 1)
        params["A"] = get_parameters(self.dim)
        params["B"] = get_parameters(self.dim)
        self.params = params

    def get_sub_hamiltonian(self, p, q):
        """Returns the sub-hamiltonian of the module"""
        # if isinstance(p, torch.Tensor) and isinstance(q, torch.Tensor):
        monomial = sum(
            p[i] * self.params["A"][i] + q[i] * self.params["B"][i]
            for i in range(self.dim)
        )
        polynomial = sum(
            self.params["a"][i - self.min_degree] * monomial**i
            for i in range(self.min_degree, self.max_degree + 1)
        )
        return polynomial
