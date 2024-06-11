from torch import nn
from ..utils import get_parameters
from ..nn.activation import Tanh
from ..utils import get_pq, get_x


class LinearLayer(nn.Module):
    """Linear symplectic layer."""

    def __init__(self, dim, sublayers, **kwargs):
        super().__init__()
        self.dim = dim
        self.sublayers = sublayers

        params = nn.ParameterDict()
        for i in range(self.sublayers):
            params[f"S{i + 1}"] = get_parameters(self.dim, self.dim)
        params["bp"] = get_parameters(self.dim, scale=0.0)
        params["bq"] = get_parameters(self.dim, scale=0.0)
        self.params = params


    def forward(self, x, h):
        p, q = get_pq(x)
        for i in range(self.sublayers):
            S = self.params[f"S{i + 1}"]
            if i % 2 == 0:
                p = p + q @ (S + S.t()) * h
            else:
                q = q + p @ (S + S.t()) * h
        p = p + self.params["bp"] * h
        q = q + self.params["bq"] * h
        return get_x(p, q)


class ActivationLayer(nn.Module):
    """Activation symplectic layer."""

    def __init__(self, dim, mode, **kwargs):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.activation = Tanh()
        self.params = nn.ParameterDict()
        self.params["a"] = get_parameters(self.dim)

    def forward(self, x, h):
        p, q = get_pq(x)
        if self.mode == "odd":
            p = p + self.activation(q) * self.params["a"] * h
        elif self.mode == "even":
            q = q + self.activation(p) * self.params["a"] * h
        else:
            raise ValueError
        return get_x(p, q)

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

    def forward(self, x, h, reverse=False):
        if self.mode == "end":
            x = self.linear_layer(x, h)
        elif reverse:
            x = self.activation_layer(x, h)
            x = self.linear_layer(x, h)
        else: 
            x = self.linear_layer(x, h)
            x = self.activation_layer(x, h)
        return x 
