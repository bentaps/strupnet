import torch
from strupnet.nn.activation import get_activation

class Net(torch.nn.Module):
    """Implements a fully connected deep neural network."""

    def __init__(self, input_dim, output_dim, width, hidden_layers=1, activation=None, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.hidden_layers = hidden_layers
        self.activation = get_activation(activation)
        self.init_params()

    def init_params(self):
        """Initializes the neural network."""
        layers = []
        layers.append(torch.nn.Linear(self.input_dim, self.width))
        layers.append(self.activation)
        for _ in range(self.hidden_layers):
            layers.append(torch.nn.Linear(self.width, self.width))
            layers.append(self.activation)
        layers.append(torch.nn.Linear(self.width, self.output_dim))
        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        """Returns the neural network evaluated at x."""
        return self.net(x)

    def grad(self, x):
        """Returns the gradient of the neural network evaluated at x."""
        return torch.autograd.grad(
            self.net(x).sum(), x, create_graph=True
        )[0]
        

