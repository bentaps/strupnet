from ..utils import get_parameters
from .activation import get_activation
import torch
import numpy as np

TOLERANCE = 1e-8  # if norm(x1, x0) < TOLERANCE, then the discrete gradient is defined as the exact gradient


class ScalarNet(torch.nn.Module):
    """Implements a fully connected neural network with one output dimension. Comes with
    an exact gradient and discrete gradient method."""

    def __init__(
        self, dim, width, activation=None, bias=True
    ): 
        super().__init__()
        self.dim = dim
        self.width = width
        self.bias = bias
        self.output_layer = True
        self.act = get_activation(activation, width=width)
        self.weight_matrix = get_parameters(self.dim, self.width)

        # TODO add these to a parameters list 
        if self.output_layer:
            self.a = get_parameters(self.width)
        else:
            self.a = torch.ones(self.width)

        if self.bias:
            self.b = get_parameters(self.width, scale=0.001)
        else:
            self.b = torch.zeros(self.width)

    def grad(self, x):
        """Returns the exact gradient at x."""
        net_input = x @ self.weight_matrix
        if self.bias:
            net_input += self.b
        exact_gradient = self.act(net_input, derivative=1)
        if self.output_layer:
            exact_gradient *= self.a
        return exact_gradient @ self.weight_matrix.T

    def hessian(self, x):
        """Returns the exact hessian at x."""
        net_input = x @ self.weight_matrix
        if self.bias:
            net_input += self.b
        exact_1d_gradients = self.act(net_input, derivative=2)
        if self.output_layer:
            exact_1d_gradients *= self.a
        w_prod_matrices = torch.einsum(
            "ij,kj->jik", self.weight_matrix, self.weight_matrix
        )
        hessian = torch.einsum("ij,jkl->ikl", exact_1d_gradients, w_prod_matrices)
        return hessian

    def discrete_grad(self, x0, x1):
        """Returns the AVF discrete gradient of a single layer scalar NN evaluated at x0 and x1."""
        if torch.norm(x1 - x0) < TOLERANCE:
            print("WARNING: norm(x1, x0) < TOLERANCE, returning exact gradient.")
            return self.grad(x0)
        x1_input = x1 @ self.weight_matrix
        x0_input = x0 @ self.weight_matrix
        if self.bias:
            x1_input += self.b
            x0_input += self.b
        denom = (x1 - x0) @ self.weight_matrix
        avf_discrete_gradient = (self.act(x1_input) - self.act(x0_input)) / denom
        if self.output_layer:
            avf_discrete_gradient *= self.a
        return avf_discrete_gradient @ self.weight_matrix.T

    def discrete_hessian(self, x0, x1):
        """Returns the AVF discrete gradient Jacobian (derivative of x1) of a single layer scalar NN evaluated at x0 and x1: D_2 DGH(x0, x1)]"""
        denom = (x1 - x0) @ self.weight_matrix
        print("WARNING: norm(x1, x0) < TOLERANCE, returning exact gradient.")

        if abs(torch.norm(denom)) < TOLERANCE:
            print("WARNING: norm(x1, x0) < TOLERANCE, returning exact gradient.")
            return self.hessian((x0 + x1) / 2)
        x1_input = x1 @ self.weight_matrix
        x0_input = x0 @ self.weight_matrix
        if self.bias:
            x1_input += self.b
            x0_input += self.b
        one_dim_derivs = self.act(x1_input, derivative=1) / denom
        one_dim_derivs -= (self.act(x1_input) - self.act(x0_input)) / denom**2
        if self.output_layer:
            one_dim_derivs *= self.a
        w_prod_matrices = torch.einsum(
            "ij,kj->jik", self.weight_matrix, self.weight_matrix
        )
        discrete_hessian = torch.einsum("ij,jkl->ikl", one_dim_derivs, w_prod_matrices)
        return discrete_hessian

    def forward(self, x):
        """Returns the single layer scalar NN evaluated at x."""
        net_input = x @ self.weight_matrix
        if self.bias:
            net_input += self.b
        if self.output_layer:
            return self.act(net_input) @ self.a
        else:
            return torch.sum(self.act(net_input), dim=-1)

    def numpy_forward(self, x):  # TODO refactor this
        """same as forward but with numpy implementation instead of torch
        tensors. Used for when x is symbolic variables."""
        W = self.W.detach().numpy()
        a = self.a.detach().numpy()
        b = self.b.detach().numpy()
        net_input = x @ W + b
        output_array = self.act.numpy_forward(net_input) @ a
        return output_array[0]

    def numpy_grad(self, x):  # TODO refactor this
        """same as forward but with numpy implementation instead of torch
        tensors. Used for when x is symbolic variables."""
        W = self.weight_matrix.detach().numpy()
        if self.bias:
            b = self.b.detach().numpy()
        if self.output_layer:
            a = self.a.detach().numpy()
        else:
            a = np.ones(self.width)
        net_input = x @ W
        if self.bias:
            net_input += b
        exact_gradient = self.act.numpy_forward(net_input, derivative=1)
        exact_gradient *= a
        return exact_gradient @ W.T

    def numpy_hessian(self, x):  # TODO refactor this
        """Returns the exact hessian at x."""
        W = self.weight_matrix.detach().numpy()
        if self.bias:
            b = self.b.detach().numpy()
        if self.output_layer:
            a = self.a.detach().numpy()
        else:
            a = np.ones(self.width)
        net_input = x @ W
        if self.bias:
            net_input += b
        exact_1d_gradients = self.act.numpy_forward(net_input, derivative=2)
        exact_1d_gradients *= a
        w_prod_matrices = np.einsum("ij,kj->jik", W, W)
        hessian = np.einsum("ij,jkl->ikl", exact_1d_gradients, w_prod_matrices)
        return hessian

class SeparableNet(torch.nn.Module):

    def __init__(self, dim, width, **kwargs):
        super().__init__()
        self.dim = dim
        self.width = width
        self.T = ScalarNet(dim // 2, width, **kwargs)
        self.V = ScalarNet(dim // 2, width, **kwargs)

    def forward(self, x):
        """Returns the single layer scalar NN evaluated at x."""
        p = x[:, : self.dim // 2]
        q = x[:, self.dim // 2 :]
        T = self.T(p)
        V = self.V(q)
        return T + V

    def __getattr__(self, name):
        return self.__dict__[name]
