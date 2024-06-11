import torch
from torch import nn
from ..utils import get_parameters, canonical_symplectic_matrix
from ..utils import get_pq, get_x


class Layer(nn.Module):
    """Radial basis function Hamiltonian map"""

    def __init__(self, dim, **kwargs):
        super().__init__()
        self.dim = dim
        self.J = canonical_symplectic_matrix(dim)
        self.params = self.init_params()

    def forward(self, p, q, h, **kwargs):
        x = torch.cat([p, q], dim=1)
        y = ((x - self.params["r"]) * (x - self.params["r"])).sum(dim=1)
        rbf = torch.exp(-self.params["epsilon"] ** 2 * y).unsqueeze(-1)
        phase = -2 * h * self.params["alpha"] * self.params["epsilon"] ** 2 * rbf
        batch_matrices = phase.unsqueeze(-1) * self.J
        # print(batch_matrices.shape)
        x = torch.bmm(torch.matrix_exp(batch_matrices), (x - self.params["r"]).unsqueeze(-1)).squeeze()
        # print("x shape", x.shape)
        p, q = x[..., : self.dim], x[..., self.dim :]        
        return p, q

    def init_params(self):
        params = nn.ParameterDict()
        params["r"] = get_parameters(2 * self.dim)
        params["epsilon"] = get_parameters(1)
        params["alpha"] = get_parameters(1)
        return params
