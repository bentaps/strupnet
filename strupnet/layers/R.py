from torch import nn
from ..utils import get_parameters
from ..nn.activation import Tanh

class Layer(nn.Module):
    """Shear module with regular ridge function."""

    def __init__(self, dim, width, **kwargs):
        super().__init__()
        self.dim = dim
        self.init_params()
        self.mlp = nn.Sequential(
            nn.Linear(1, width),
            Tanh(),
            nn.Linear(width, 1, bias=False),
        )

    def forward(self, p, q, h, **kwargs):
        z = (p @ self.params["A"] + q @ self.params["B"]).unsqueeze(-1)
        gradH = self.mlp(z)
        return p - h * gradH * self.params["B"], q + h * gradH * self.params["A"]

    def init_params(self):
        params = nn.ParameterDict()
        params["A"] = get_parameters(self.dim)
        params["B"] = get_parameters(self.dim)
        params["theta"] = get_parameters(1)
        self.params = params