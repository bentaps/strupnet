import torch
import importlib

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
        }
        self.layers_list = self.init_layers()

    def forward(self, x, dt, symmetric=None, reverse=False):
        symmetrise = symmetric if symmetric is not None else self.symmetric
        if symmetrise:
            x = self.forward(x, dt / 2, symmetric=False)
            x[..., : self.dim] *= -1.0
            x = self.forward(x, -dt / 2, symmetric=False, reverse=True)
            x[..., : self.dim] *= -1.0
            return x
        else:
            p, q = x[..., : self.dim], x[..., self.dim :]
            # TODO handle variable timesteps. Code currently only accepts a scalar uniform time step dt.
            h = dt * torch.ones_like(x[..., -1:])
            layers = reversed(self.layers_list) if reverse else self.layers_list
            for i, layer in enumerate(layers):
                p, q = layer(p, q, h, reverse=reverse)
            return torch.cat([p, q], dim=-1)

    def init_layers(self):
        layers = torch.nn.ModuleList()
        for i in range(self.layers):
            self.layer_kwargs["mode"] = "odd" if i % 2 == 0 else "even"
            if i == self.layers - 1 and self.method == "LA":
                # LA needs special treatment for last layer
                self.layer_kwargs["mode"] = "end"

            # import Layer class from layers folder based on self.method
            layer = importlib.import_module(
                f".{self.method}", package="strupnet.layers"
            ).Layer
            layers.append(layer(**self.layer_kwargs))
        return layers

    def init_params(self):
        for module in self.layers_list:
            if hasattr(module, "init_params"):
                module.init_params()
