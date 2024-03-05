from torch import nn
from ..utils import get_parameters
from ..nn.activation import Tanh


class LinearLayer(nn.Module):
    """Linear symplectic layer."""

    def __init__(self, dim, sublayers, **kwargs):
        super().__init__()
        self.dim = dim
        self.sublayers = sublayers
        self.init_params()

    def forward(self, p, q, h):
        for i in range(self.sublayers):
            S = self.params[f"S{i + 1}"]
            if i % 2 == 0:
                p = p + q @ (S + S.t()) * h
            else:
                q = p @ (S + S.t()) * h + q
        return p + self.params["bp"] * h, q + self.params["bq"] * h

    def init_params(self):
        params = nn.ParameterDict()
        for i in range(self.sublayers):
            params[f"S{i + 1}"] = get_parameters(self.dim, self.dim)
        params["bp"] = get_parameters(self.dim, scale=0.0)
        params["bq"] = get_parameters(self.dim, scale=0.0)
        self.params = params


class ActivationLayer(nn.Module):
    """Activation symplectic layer."""

    def __init__(self, dim, mode, **kwargs):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.activation = Tanh()
        self.init_params()

    def forward(self, p, q, h):
        if self.mode == "odd":
            return p + self.activation(q) * self.params["a"] * h, q
        elif self.mode == "even":
            return p, self.activation(p) * self.params["a"] * h + q
        else:
            raise ValueError

    def init_params(self):
        params = nn.ParameterDict()
        params["a"] = get_parameters(self.dim)
        self.params = params


class Layer(nn.Module):
    """LA layer """

    def __init__(self, dim, sublayers, mode="odd", **kwargs):
        super().__init__()
        self.dim = dim
        self.sublayers = sublayers
        self.mode = mode
        self.linear_layer = LinearLayer(self.dim, self.sublayers)
        if mode != "end":
            self.activation_layer = ActivationLayer(self.dim, self.mode)

    def forward(self, p, q, h, reverse=False):
        if self.mode == "end":
            p, q = self.linear_layer(p, q, h)
        elif reverse:
            p, q = self.activation_layer(p, q, h)
            p, q = self.linear_layer(p, q, h)
        else: 
            p, q = self.linear_layer(p, q, h)
            p, q = self.activation_layer(p, q, h)
        return p, q 
