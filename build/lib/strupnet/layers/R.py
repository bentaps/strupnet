from torch import nn
from ..utils import get_parameters
from ..nn.activation import Tanh
from ..nn.scalar import ScalarNet
from ..utils import get_pq, get_x, canonical_symplectic_matrix, symplectic_matrix_transformation_2d


class Layer(nn.Module):
    """Shear module with regular ridge function."""

    def __init__(self, dim, width, keepdim=False, **kwargs):
        super().__init__()
        self.params = nn.ParameterDict()
        self.params["w"] = get_parameters(dim if keepdim else 2 * dim)
        self.symplectic_matrix = canonical_symplectic_matrix(dim)
        self.hamiltonian = ScalarNet(1, width, activation="tanh")

    def forward(self, x, h, i=None, **kwargs):
        z = (x @ self.params["w"]).unsqueeze(-1)
        gradH = self.hamiltonian(z).unsqueeze(-1)
        if i is None:
            symp_weight = self.params["w"] @ self.symplectic_matrix
        elif isinstance(i, int): # pick the i and i+1 components of w for the volume preserving symplectic flows. 
            symp_weight = symplectic_matrix_transformation_2d(self.params["w"], i)
        else:
            raise ValueError("i must be an integer or None")
        x = x + h * gradH * symp_weight
        return x

        # gradH = self.mlp(x @ self.params["w"])
        # print(f"{gradH.shape=}")
        # print(f"{x.shape=}")
        # A = torch.matmul(self.symplectic_matrix, self.params["w"])
        # print(f"{A.shape=}")
        # x = x + h * gradH * torch.matmul(self.symplectic_matrix, self.params["w"])
        # return x
