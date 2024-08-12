import torch
import importlib
import sympy as sp
from .bea import truncated_bea_map


torch.set_default_dtype(torch.float64)

ALLOWED_METHODS = ["R", "P", "G", "GR", "LA", "H"]


class SympNet(torch.nn.Module):
    def __init__(
        self,
        dim,
        layers=8,
        width=8,
        symmetric=False,
        method=None,
        min_degree=2,
        max_degree=4,
        sublayers=5,
        activation=None,
        volume_step=False,
        **kwargs,
    ):
        """SympNet.
        Args:
            dim (int): Degrees of freedom of the system.
            layers (int): Number of layers.
            width (int): Width of the hidden layers.
            method (str): Method to use for the layers. Allowed values are "R", "P", "G", "GR", "LA" or "H". Default: R.
            symmetric (bool): Make the method time-symmetric. Default: False.
            min_degree (int): For method="P" only. Minimum degree of the polynomial. Default: 2.
            max_degree (int): For method="P" only. Maximum degree of the polynomial. Default: 4.
            sublayers (int): For method="LA" only. Number of sublayers for the LA method. Default: 5.
        """
        super().__init__()
        self.symmetric = symmetric
        self.dim = dim
        self.layers = layers
        method = method if method is not None else "R"
        assert method in ALLOWED_METHODS, f"Method must be one of {ALLOWED_METHODS}"
        self.method = method
        self.layer_kwargs = {
            "dim": dim,
            "layers": layers,
            "width": width,
            "min_degree": min_degree if method == "P" else None,
            "max_degree": max_degree if method == "P" else None,
            "sublayers": sublayers if method == "LA" else None,
            "activation": activation,
            "volume_step": volume_step,
            **kwargs,
        }
        self.layers_list = self.init_layers()

    def forward(self, x, dt, symmetric=None, reverse=False):
        """Forward pass of the SNN.
        Args:
            x (torch.Tensor): Input tensor of shape [..., 2 * dim].
            dt (float): Time step.
            symmetric (bool): Make the method time-symmetric. Default: None.
            reverse (bool): Reverse the SNN layers. Default: False.
        """
        symmetrise = symmetric if symmetric is not None else self.symmetric
        if symmetrise:
            x = self.forward(x, dt / 2, symmetric=False)
            x[..., self.dim :] *= -1.0
            x = self.forward(x, -dt / 2, symmetric=False, reverse=True)
            x[..., self.dim :] *= -1.0
            return x
        else:
            assert (
                x.size(-1) == 2 * self.dim
            ), f"Input must be of shape [..., 2 * dim], got shape {x.shape}, when dim is {self.dim}"
            # TODO handle variable timesteps. Code currently only accepts a scalar uniform time step dt.
            h = dt * torch.ones_like(x[..., -1:])
            layers = reversed(self.layers_list) if reverse else self.layers_list
            for layer in layers:
                x = layer(x, h, reverse=reverse)
            return x

    def init_layers(self):
        layers = torch.nn.ModuleList()
        for i in range(self.layers):
            self.layer_kwargs["mode"] = "odd" if i % 2 == 0 else "even"
            if i == self.layers - 1 and self.method == "LA":
                # LA needs special treatment for last layer
                self.layer_kwargs["mode"] = "end"

            # Import Layer class from layers folder based on self.method
            layer = importlib.import_module(
                f".{self.method}", package="strupnet.layers"
            ).Layer
            layers.append(layer(**self.layer_kwargs))
        return layers

    def hamiltonian(self, x, h=None, order=0):
        if not hasattr(self, "_hamiltonian_function"):
            error_msg = "Must provide timestep h for first call to hamiltonian"
            assert h is not None, error_msg
            H_sym = self.corrected_hamiltonian(order=order, h=h)
            H_fun = sp.lambdify(sp.symbols(f"x:{2*self.dim}"), H_sym.as_expr())
            self._hamiltonian_function = lambda x: H_fun(*x.T)
        x = x.detach().numpy() if isinstance(x, torch.Tensor) else x
        return self._hamiltonian_function(x)

    def corrected_hamiltonian(self, order=0, h=None, truncate=True, tol=None):
        """returns the corrected hamiltonian for the sympnet up to the specified
        order. Substitutes the timestep h if provided, otherwise returns the
        expression with symbolic h."""
        x_sym = sp.symbols(f"x:{2 * self.dim}")
        h_sym = sp.symbols(f"h")
        sub_hamiltonians = self._inv_mod_hamiltonians(x_sym)
        corrected_H_with_sym_h = truncated_bea_map(
            sub_hamiltonians,
            dim=self.dim,
            h=h_sym,
            order=order,
            truncate=truncate,
            x_vars=x_sym,
            tol=tol,
        )
        if h is not None:
            return corrected_H_with_sym_h.subs(h_sym, h)
        else:
            return corrected_H_with_sym_h

    def _inv_mod_hamiltonians(self, x):
        """Returns the list of sub-hamiltonians for the inverse modified Hamiltonian"""
        if not hasattr(self.layers_list[0], "hamiltonian"):
            raise NotImplementedError(
                f"Hamiltonian not implemented for method {self.method}"
            )
        hamiltonian_list = []
        for layer in self.layers_list:
            hamiltonian_list.append(layer.hamiltonian(x))
        return hamiltonian_list
