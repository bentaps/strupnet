import torch

from .nn.net import Net
from .utils import get_parameters, canonical_symplectic_matrix
import scipy.optimize as optimize

torch.set_default_dtype(torch.float64)


class HamiltonianNet(torch.nn.Module):
    """Hamiltonian Neural Net with midpoint rule and exact gradient"""

    def __init__(
        self,
        dim,
        width=32,
        force_width=None,
        learn_resistance=False,
        learn_force=False,
        activation=None,
        **kwargs,
    ):
        super().__init__()
        print(learn_force, learn_resistance)
        self.dim = dim
        self.method = "HNN"
        self.hamiltonian = Net(input_dim=2 * dim, output_dim=1, width=width, activation=activation)
        self.force_width = force_width if force_width else width
        self.J = canonical_symplectic_matrix(dim)
        self.learn_resistance = learn_resistance
        self.learn_force = learn_force
        self.init_params()

    def forward(self, x0, x1, t0, t1):
        dt = t1 - t0
        t_mid = 0.5 * (t0 + t1)
        hamiltonian_grad_midpoint = self.hamiltonian.grad(0.5 * (x0 + x1))
        # Phi is the discretised vector field according to midpoint rule s.t. x1 = x0 + dt * Phi
        phi = hamiltonian_grad_midpoint @ self.J

        if self.learn_resistance:
            R_diag = self.params["sqrt_r_diag"] ** 2
            phi[..., : self.dim] -= hamiltonian_grad_midpoint[..., : self.dim] * R_diag

        if self.learn_force:
            phi[..., : self.dim] += self.force(t_mid)

        return x0 + dt * phi

    def flow_map(self, x, t, dt, symplectic=True):
        """Returns the midpoint solution of the learned Hamiltonian ODE"""
        if symplectic:
            obj = lambda x1: x + dt * self.grad_ham(0.5 * (x + x1)) @ self.J - x1
        else:
            raise NotImplementedError("Only implemented for symplectic flow maps.")
        with torch.no_grad():
            result = optimize.root(obj, x)
        x1 = result.x
        return x1

    @property
    def resistance_matrix(self):
        return torch.diag(self.params["sqrt_r_diag"] ** 2)

    def init_params(self):
        if self.learn_resistance:
            self.params = torch.nn.ParameterDict()
            self.params["sqrt_r_diag"] = get_parameters(self.dim)
        if self.learn_force:
            self.force = torch.nn.Sequential(
                torch.nn.Linear(1, self.force_width),
                torch.nn.Tanh(),
                torch.nn.Linear(self.force_width, self.dim),
            )
        self.hamiltonian.init_params()