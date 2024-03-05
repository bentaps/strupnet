from ..utils import get_parameters
from .activation import get_activation
import torch

TOLERANCE = 1e-8  # if norm(x1, x0) < TOLERANCE, then the discrete gradient is defined as the exact gradient


class ScalarNet(torch.nn.Module):
    """Implements a fully connected neural network with one output dimension. Comes with
    an exact gradient and discrete gradient method."""

    def __init__(self, dim, width, activation=None):
        super().__init__()
        self.dim = dim
        self.width = width
        self.act = get_activation(activation, dim=dim)
        self.params = self.init_params()

    def grad(self, x):
        """Returns the exact gradient at x."""
        input = x @ self.params["W"] + self.params["b"]
        exact_gradient = self.act(input, derivative=1)
        exact_gradient *= self.params["a"]
        return exact_gradient@self.params["W"].T

    def hessian(self, x):
        """Returns the exact hessian at x."""
        input = x @ self.params["W"] + self.params["b"]
        exact_1d_gradients = self.act(input, derivative=2)
        exact_1d_gradients *= self.params["a"]
        w_prod_matrices = torch.einsum("ij,kj->jik", self.params["W"], self.params["W"])
        hessian = torch.einsum("ij,jkl->ikl", exact_1d_gradients, w_prod_matrices)
        return hessian

    def discrete_grad(self, x0, x1):
        """Returns the AVF discrete gradient of a single layer scalar NN evaluated at x0 and x1."""
        if torch.norm(x1 - x0) < TOLERANCE:
            return self.grad(x0)
        x1_input = x1 @ self.params["W"] + self.params["b"]
        x0_input = x0 @ self.params["W"] + self.params["b"]
        denom = (x1 - x0) @ self.params["W"]
        avf_discrete_gradient = (self.act(x1_input) - self.act(x0_input))/denom 
        avf_discrete_gradient *= self.params["a"]
        return avf_discrete_gradient @ self.params["W"].T

    def discrete_hessian(self, x0, x1):
        """Returns the AVF discrete gradient Jacobian (derivative of x1) of a single layer scalar NN evaluated at x0 and x1: D_2 DGH(x0, x1)]"""
        denom = (x1 - x0) @ self.params["W"]
        if abs(torch.norm(denom)) < TOLERANCE:
            return self.hessian((x0 + x1) / 2)
        x1_input = x1 @ self.params["W"] + self.params["b"]
        x0_input = x0 @ self.params["W"] + self.params["b"]
        one_dim_derivs = self.act(x1_input, derivative=1) / denom
        one_dim_derivs -= (self.act(x1_input) - self.act(x0_input)) / denom**2
        one_dim_derivs *= self.params["a"]
        w_prod_matrices = torch.einsum("ij,kj->jik", self.params["W"], self.params["W"])
        discrete_hessian = torch.einsum("ij,jkl->ikl", one_dim_derivs, w_prod_matrices)
        return discrete_hessian

    def forward(self, x):
        """Returns the single layer scalar NN evaluated at x."""
        return self.act(x @ self.params["W"] + self.params["b"]) @ self.params["a"]

    def init_params(self):
        params = torch.nn.ParameterDict()
        params["W"] = get_parameters(self.dim, self.width)
        params["a"] = get_parameters(self.width)
        params["b"] = get_parameters(self.width, scale=0.0)
        return params


