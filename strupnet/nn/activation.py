import torch
from ..utils import get_parameters
import re

VALID_ACTIVATIONS = ['tanh', 'sigmoid', 'mono\d+']

def extract_last_digits(s: str):
    matches = re.findall(r'\d+', s)
    return int(matches[-1]) if matches else None

def get_activation(activation, dim=None):
    """Converts valid string to callable activation function. Defaults to tanh."""
    if activation == "tanh" or activation is None:
        return Tanh()
    elif activation == "sigmoid":
        return Sigmoid()
    elif activation.lower().startswith("mono"):
        degree = extract_last_digits(activation)
        assert degree is not None, "degree must be provided for monomial activation, e.g., mono2"
        return Monomial(degree=degree)
    else:
        raise ValueError(f"activation must be one of [{VALID_ACTIVATIONS}]")

class Sigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, derivative=0):
        if derivative == 0:
            return torch.sigmoid(x)
        elif derivative == 1:
            return torch.sigmoid(x) * (1 - torch.sigmoid(x))
        elif derivative == 2:
            return torch.sigmoid(x) * (1 - torch.sigmoid(x)) * (1 - 2 * torch.sigmoid(x))
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
        """ Monomial activation function layer that accepts an input tensor 'x' of dimension 
        (nbatch, dim) and returns a tensor 'P' of same dimension. Each element in the 
        returned tensor 'P[:, j] is of the form sum_{j=0}^{degree} a_ij x[:]^j
        """
        super().__init__()
        self.degree = degree

    def forward(self, x, derivative=0):   
        if derivative==0:
            return torch.pow(x, self.degree)
        elif derivative==1:
            return self.degree * torch.pow(x, self.degree - 1)
        elif derivative==2:
            return self.degree * (self.degree - 1) * torch.pow(x, self.degree - 2)                 

