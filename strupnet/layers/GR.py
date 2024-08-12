import torch
from torch import nn
from ..utils import get_parameters
from ..nn.scalar import ScalarNet
from ..utils import get_pq, get_x


class Layer(nn.Module):
    """Generalised ridge function symplectic module"""

    def __init__(self, dim, width, mode="odd", basis="general", activation=None, bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.width = width
        self.hamiltonian = ScalarNet(dim, width, activation=activation)
        self.basis = basis
        self.params = nn.ParameterDict()
        if self.dim == 1:
            self.params["A"] = get_parameters(1, 1)
            self.params["B"] = get_parameters(1, 1)
        if self.basis == "general":
            n_sym = self.dim * (self.dim + 1) // 2
            self.params["s1"] = get_parameters(n_sym, 1)
            self.params["s2"] = get_parameters(n_sym, 1)
            self.params["s3"] = get_parameters(n_sym, 1)
        elif self.basis == "diagonal":
            self.params["A"] = get_parameters(dim)
            self.params["B"] = get_parameters(dim)
        else: 
            raise ValueError(f"Invalid basis: {basis}. Must be 'general' or 'diagonal'")
        self.bias = bias
        if self.bias:
            self.params["bias"] = get_parameters(dim, scale=0.0)

    def forward(self, x, h, **kwargs):
        A, B = self.get_basis()
        p, q = get_pq(x)
        z = torch.matmul(p, A) + torch.matmul(q, B)
        z = z + self.params["bias"] if self.bias else z
        hamiltonian_grad = self.hamiltonian.grad(z)
        p = p - h * torch.matmul(hamiltonian_grad, B.T)
        q = q + h * torch.matmul(hamiltonian_grad, A.T)
        return get_x(p, q)

    def get_basis(self):
        if self.dim == 1:
            return self.params["A"], self.params["B"]
        if self.basis == "general":
            S1 = self.vector_to_symmetric_matrix(self.params["s1"], self.dim)
            S2 = self.vector_to_symmetric_matrix(self.params["s2"], self.dim)
            S3 = self.vector_to_symmetric_matrix(self.params["s3"], self.dim)
            A = torch.eye(self.dim) + S1 @ S2
            B = S1 + S1 @ S2 @ S3 + S3
            if self.mode == "odd":
                return A, B
            elif self.mode == "even":
                return B, A
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Must be 'odd' or 'even'")
        elif self.basis == "diagonal":
            # return diagonal matrices A and B
            A = torch.diag(self.params["A"])
            B = torch.diag(self.params["B"])
            return A, B
            
            

    def vector_to_symmetric_matrix(self, vector, n):
        matrix = torch.zeros((n, n))
        matrix[torch.triu_indices(n, n)] = vector
        matrix = matrix + matrix.T
        return matrix