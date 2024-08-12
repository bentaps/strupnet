from torch import nn
from ..utils import get_parameters
from ..nn.activation import get_activation
from ..nn.scalar import ScalarNet
from ..utils import canonical_symplectic_matrix, symplectic_matrix_transformation_2d


class Layer(nn.Module):
    """Shear module with regular ridge function."""

    def __init__(self, dim, width, keepdim=False, **kwargs):
        """ linear (bool): If True, adds a quadratic term to the hamiltonian to exactly represent linear dynamics.
            keepdim (bool): If True, output will be of size dim, otherwise will have size 2*dim. Default: False."""

        super().__init__()
        self.params = nn.ParameterDict()
        self.params["w"] = get_parameters(dim if keepdim else 2 * dim)
        self.symplectic_matrix = canonical_symplectic_matrix(dim)
        self.hamiltonian = ScalarNet(1, width, activation="tanh")
        self.init_kwargs = kwargs
        
    def forward(self, x, h, i=None, **kwargs):
        """ i (int): Used for VolNet only. Index specifying 2D components for subflows: (i, i+1)."""
        z = (x @ self.params["w"]).unsqueeze(-1)        
        gradH = self.hamiltonian.grad(z)
        if i is None:
            symp_weight = self.params["w"] @ self.symplectic_matrix
        elif isinstance(i, int): # pick the i and i+1 components of w for the volume preserving symplectic flows. 
            symp_weight = symplectic_matrix_transformation_2d(self.params["w"], i)
        else:
            raise ValueError("i must be an integer or None")
        x = x + h * gradH * symp_weight
        return x
