import torch
from torch import nn
from ..utils import get_parameters
from ..nn.scalar import ScalarNet

class Layer(nn.Module):
    """Generalised ridge function symplectic module"""

    def __init__(self, dim, width, mode="odd", activation=None, **kwargs):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.width = width
        self.hamiltonian = ScalarNet(dim, width, activation=activation)
        self.params = self.init_params()

    def forward(self, p, q, h, **kwargs):
        A, B = self.get_basis()
        z = torch.matmul(p, A) + torch.matmul(q, B)
        hamiltonian_grad = self.hamiltonian.grad(z)
        p = p - h * torch.matmul(hamiltonian_grad, B.T)
        q = q + h * torch.matmul(hamiltonian_grad, A.T)
        return p, q

    def init_params(self):
        self.hamiltonian.init_params()
        params = nn.ParameterDict()
        if self.dim == 1:
            params["A"] = get_parameters(1, 1)
            params["B"] = get_parameters(1, 1)
            return params
        n_sym = self.dim * (self.dim + 1) // 2
        params["s1"] = get_parameters(n_sym, 1)
        params["s2"] = get_parameters(n_sym, 1)
        params["s3"] = get_parameters(n_sym, 1)
        return params

    def get_basis(self):
        if self.dim == 1:
            return self.params["A"], self.params["B"]
        S1 = self.vector_to_symmetric_matrix(self.params["s1"], self.dim)
        S2 = self.vector_to_symmetric_matrix(self.params["s2"], self.dim)
        S3 = self.vector_to_symmetric_matrix(self.params["s3"], self.dim)
        if self.mode == "odd":
            A = torch.eye(self.dim) + S1 @ S2
            B = S1 + S1 @ S2 @ S3 + S3
        elif self.mode == "even":
            A = S1 + S1 @ S2 @ S3 + S3
            B = torch.eye(self.dim) + S1 @ S2
        return A, B

    def vector_to_symmetric_matrix(self, vector, n):
        matrix = torch.zeros((n, n))
        matrix[torch.triu_indices(n, n)] = vector
        matrix = matrix + matrix.T
        return matrix