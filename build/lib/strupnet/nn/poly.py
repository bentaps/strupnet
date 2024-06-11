
from ..utils import get_parameters
from .activation import get_activation, extract_last_digits
from ..numeric import AVFRK2
import torch
import numpy as np
import sympy as sp

class PolyNet(torch.nn.Module):
    """Implements a fully connected neural network with one output dimension. Comes with
    an exact gradient and discrete gradient method."""

    def __init__(self, dim, width, max_degree, min_degree=2):
        super().__init__()
        self.dim = dim
        self.width = width
        self.max_degree = max_degree
        self.act = get_activation(
            f"poly{max_degree}", width=width, min_degree=min_degree
        )
        self.weight_matrix = get_parameters(self.dim, self.width)

    def grad(self, x):
        """Returns the exact gradient at x."""
        return self.act(x @ self.weight_matrix, derivative=1) @ self.weight_matrix.T

    def hessian(self, x):
        """Returns the exact hessian at x."""
        exact_1d_gradients = self.act(x @ self.weight_matrix, derivative=2)
        w_prod_matrices = torch.einsum(
            "ij,kj->jik", self.weight_matrix, self.weight_matrix
        )
        hessian = torch.einsum("ij,jkl->ikl", exact_1d_gradients, w_prod_matrices)
        return hessian

    def discrete_grad(self, x0, x1):
        """Returns the AVF discrete gradient of a single layer scalar NN evaluated at x0 and x1."""
        return AVFRK2(lambda x, t: self.grad(x), x0=x0, x1=x1, degree=self.max_degree)

    def forward(self, x):
        """Returns the single layer scalar NN evaluated at x."""
        return torch.sum(self.act(x @ self.weight_matrix), dim=-1)

    def numpy_forward(self, x):  # TODO refactor this
        """same as forward but with numpy implementation instead of torch
        tensors. Used for when x is symbolic variables."""
        W = self.weight_matrix.detach().numpy()
        output_array = np.sum(self.act.numpy_forward(x @ W), axis=-1)
        return output_array[0]

    def numpy_grad(self, x):  # TODO refactor this
        """same as forward but with numpy implementation instead of torch
        tensors. Used for when x is symbolic variables."""
        W = self.weight_matrix.detach().numpy()
        return self.act.numpy_forward(x @ W, derivative=1) @ W.T

    def numpy_hessian(self, x):  # TODO refactor this
        """Returns the exact hessian at x."""
        W = self.weight_matrix.detach().numpy()
        exact_1d_gradients = self.act.numpy_forward(x @ W, derivative=2)
        w_prod_matrices = np.einsum("ij,kj->jik", W, W)
        hessian = np.einsum("ij,jkl->ikl", exact_1d_gradients, w_prod_matrices)
        return hessian
