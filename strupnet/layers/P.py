import torch
from torch import nn
from ..utils import get_parameters
from ..utils import canonical_symplectic_transformation, symplectic_matrix_transformation_2d


class Layer(nn.Module):
    def __init__(self, dim, min_degree=None, max_degree=4, keepdim=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.max_degree = max_degree
        self.min_degree = min_degree or 2

        self.params = nn.ParameterDict()
        self.params["a"] = get_parameters(self.max_degree - self.min_degree + 1)
        self.params["w"] = get_parameters(dim if keepdim else 2 * dim)


    def forward(self, x, h, i=None, **kwargs):
        monomial = (x @ self.params["w"]).unsqueeze(-1)
        polynomial_derivative = sum(
            i * self.params["a"][i - self.min_degree] * torch.pow(monomial, i - 1)
            for i in range(self.min_degree, self.max_degree + 1)
        )
        if i is None:
            symp_weight = canonical_symplectic_transformation(self.params["w"])
        elif isinstance(i, int): # pick the i and i+1 components of w for the volume preserving symplectic flows. 
            symp_weight = symplectic_matrix_transformation_2d(self.params["w"], i)
        else:
            raise ValueError("i must be an integer or None")
        x = x + h * polynomial_derivative * symp_weight
        return x

    def hamiltonian(self, x):
        """Returns the sub-hamiltonian of the layer"""
        # if isinstance(p, torch.Tensor) and isinstance(q, torch.Tensor):
        monomial = sum(x[i] * self.params["w"][i] for i in range(2 * self.dim) )
        polynomial = sum(
            self.params["a"][i - self.min_degree] * monomial**i
            for i in range(self.min_degree, self.max_degree + 1)
        )
        return polynomial
