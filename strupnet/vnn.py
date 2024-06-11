import torch
import importlib


class VolNet(torch.nn.Module):
    def __init__(self, dim, **snn_kwargs):
        """VolNet.

        Args:
            dim (int): Dimension of the ODE.
            layers (int): Number of layers.
            width (int): Width of the hidden layers.
            snn_kwargs: key_word arguments for the SympNet (see SympNet doc string for those options).
                Only supports method='R' or 'P'.
        """
        super().__init__()
        assert dim>2, "VolNet should only be used for dim>2. For dim=2 use SympNet!"
        if not hasattr(snn_kwargs, "method"):
            snn_kwargs["method"] = "R"
        if snn_kwargs["method"] not in ["R", "P"]:
            raise ValueError("VolNet only supports method 'R'")

        self.layer_list = torch.nn.ModuleList(
            [Symp2DStep(dim=dim, **snn_kwargs) for _ in range(dim - 1)]
        )

    def forward(self, x, dt):
        # Initialize an empty tensor to store the output
        # Loop through each layer and update the corresponding slices of x_out
        for i, layer in enumerate(self.layer_list):
            x = layer(x=x, dt=dt, i=i)
        return x
    

class Symp2DStep(torch.nn.Module):
    # use a seperate class than SympNet (for now) to allow for odd dimension

    def __init__(
        self,
        dim,
        layers=8,
        width=8,
        method=None,
        activation=None,
        min_degree=2,
        max_degree=8,
    ):
        super().__init__()
        self.dim = dim
        self.layers = layers
        method = method if method is not None else "R"
        assert method in ['R', 'P'], f"Method must be 'R' or 'P' not {method}"
        self.method = method
        self.layer_kwargs = {
            "dim": dim,
            "layers": layers,
            "width": width,
            "activation": activation,
            "min_degree": min_degree,
            "max_degree": max_degree,
        }
        self.layers_list = self.init_layers()

    def forward(self, x, dt, i):
        """
        i (int): Used for VolNet only. Index specifying 2D components for subflows: (i, i+1). 
                Other dims are frozen Default: None.
        """
        h = dt * torch.ones_like(x[..., -1:])
        for layer in self.layers_list:
            x = layer(x, h, i=i)
        return x
    
    def init_layers(self):
        layers_list = torch.nn.ModuleList()
        for _ in range(self.layers):
            layer = importlib.import_module(
                f".{self.method}", package="strupnet.layers"
            ).Layer
            layers_list.append(layer(**self.layer_kwargs, keepdim=True))
        return layers_list