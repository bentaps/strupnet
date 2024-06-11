import torch
from ..utils import get_parameters
import re
import numpy as np


def extract_last_digits(s: str):
    """Returns trailing digits from a string. E.g., 'mono10' -> 10."""
    matches = re.findall(r"\d+", s)
    return int(matches[-1]) if matches else None


def get_activation(activation, width=None, min_degree=1):
    """Converts valid string to callable activation function. Defaults to tanh."""
    if activation is None:
        return Tanh()
    if activation.lower() == "tanh":
        return Tanh()
    elif activation.lower() == "sigmoid":
        return Sigmoid()
    elif activation.lower().startswith("mono"):
        degree = extract_last_digits(activation)
        assert degree is not None, "degree must be provided for monomial activation"
        return Monomial(degree=degree)
    elif activation.lower().startswith("poly"):
        degree = extract_last_digits(activation)
        assert degree is not None, "degree must be provided for polynomial activation"
        assert width is not None, "width must be provided for polynomial activation"
        return Polynomial(width=width, min_degree=min_degree, max_degree=degree)
    else:
        raise ValueError(
            """activation must be 'tanh', 'sigmoid', 'monoN' or 'polyN', where
            the integer N is the desired degree. E.g., mono4, or poly6"""
        )


class Sigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, derivative=0):
        if derivative == 0:
            return torch.sigmoid(x)
        elif derivative == 1:
            return torch.sigmoid(x) * (1 - torch.sigmoid(x))
        elif derivative == 2:
            return (
                torch.sigmoid(x) * (1 - torch.sigmoid(x)) * (1 - 2 * torch.sigmoid(x))
            )
        else:
            raise ValueError("derivative must be 0, 1 or 2")


class Tanh(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, derivative=0):
        if derivative == 0:
            return torch.tanh(x)
        elif derivative == 1:
            return 1 - torch.tanh(x) ** 2
        elif derivative == 2:
            return -2 * torch.tanh(x) * (1 - torch.tanh(x) ** 2)
        else:
            raise ValueError("derivative must be 0, 1 or 2")


class Monomial(torch.nn.Module):
    def __init__(self, degree):
        """Monomial activation function layer that accepts an input tensor 'x' of dimension
        (nbatch, dim) and returns a tensor 'P' of same dimension. Each element in the
        returned tensor 'P[:, j] is of the form sum_{j=0}^{degree} a_ij x[:]^j
        """
        super().__init__()
        self.degree = degree

    def forward(self, x, derivative=0):
        if derivative == 0:
            return torch.pow(x, self.degree)
        elif derivative == 1:
            return self.degree * torch.pow(x, self.degree - 1)
        elif derivative == 2:
            return self.degree * (self.degree - 1) * torch.pow(x, self.degree - 2)


class Polynomial(torch.nn.Module):
    def __init__(self, max_degree, width, min_degree=1):
        """Trainable polynomial activation function layer that accepts an input tensor 'x' of dimension
        (nbatch, width) and returns a tensor 'P' of same dimension. Each batch in the
        returned tensor is as follows
            P[:, i] = sum_{j=min_degree}^{max_degree} a[i, j] * x[:, i]^j
        where a are trainable parameters.
        """
        super().__init__()
        self.max_degree = max_degree
        self.min_degree = min_degree
        self.width = width
        self.degrees = torch.arange(min_degree, max_degree + 1)
        self.params = self.init_params()

    def forward(self, x, derivative=0):
        if derivative == 0:
            y = x[..., None] ** self.degrees
        elif derivative == 1:
            degrees1 = torch.clamp(self.degrees - 1, min=0)
            y = self.degrees[None] * torch.pow(x[..., None], degrees1)
        elif derivative == 2:
            degrees1 = torch.clamp(self.degrees - 1, min=0)
            degrees2 = torch.clamp(self.degrees - 2, min=0)
            y = self.degrees[None] * degrees1[None] * torch.pow(x[..., None], degrees2)
        return torch.sum(self.params["c"][None, ...] * y, dim=-1)

    def numpy_forward(self, x, derivative=0):
        """Same as the above forward but with numpy implementation"""
        coeffs = self.params["c"].detach().numpy()
        degrees = self.degrees.detach().numpy()
        if derivative == 0:
            y = x[..., None] ** degrees
        elif derivative == 1:
            degrees1 = np.clip(a=degrees - 1, a_min=0, a_max=None)
            y = degrees[None] * (x[..., None] ** degrees1)
        elif derivative == 2:
            degrees1 = np.clip(a=degrees - 1, a_min=0, a_max=None)
            degrees2 = np.clip(a=degrees - 2, a_min=0, a_max=None)
            y = degrees[None] * degrees1[None] * (x[..., None] ** degrees2)
        return np.sum(coeffs[None, ...] * y, axis=-1)

    def init_params(self):
        params = torch.nn.ParameterDict()
        params["c"] = get_parameters(self.width, len(self.degrees))
        return params
