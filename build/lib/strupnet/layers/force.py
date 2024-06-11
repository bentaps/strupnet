import torch
from torch import nn
from ..utils import get_parameters

# TODO implement inverse mass matrix! i.e., dp/dt = -R*M(q)*p + f(t)
# TODO implement q-dependence in f and/or R! i.e., dp/dt = -R(q)*M(q)*p + f(t, q)

class Layer(nn.Module):
    """ Computes the solution to
            dp/dt = -R*p + f(t), 
            dq/dt = 0,
        where R is constant and diagonal. """
    def __init__(self, dim, width, learn_resistance=True, learn_force=True):
        super().__init__()
        self.dim = dim
        self.width = width
        self.learn_resistance = learn_resistance
        self.learn_force = learn_force
        self.init_params()

    def forward(self, x, t, h):
        p, q = x[..., :self.dim], x[..., self.dim:]
        
        if not self.learn_resistance and not self.learn_force: # learns nothing
            return x # identity map
        
        if self.learn_resistance:
            R_diag = self.params["sqrt_r_diag"]**2
            R_diag_exp_h = torch.exp(-R_diag * h)  # Element-wise exponential
        else:
            R_diag = 1.0
            R_diag_exp_h = 1.0

        if self.learn_force:
            f = self.force(t)
        elif not self.learn_force:
            f = torch.zeros_like(p)
            
        
        b = f / R_diag  # R is diagonal, use element-wise division
        
        p = (p - b) * R_diag_exp_h + b  # Element-wise operations
        
        x = torch.cat([p, q], dim=-1)
        return x
     
    @property
    def resistance_matrix(self):
        return torch.diag(self.params["sqrt_r_diag"]**2)
    
    def init_params(self):
        if self.learn_force:
            self.force = nn.Sequential(
                nn.Linear(1, self.width),
                nn.Tanh(),
                nn.Linear(self.width, self.dim),
            )
        if self.learn_resistance:
            self.params = nn.ParameterDict()    
            self.params["sqrt_r_diag"] = get_parameters(self.dim)