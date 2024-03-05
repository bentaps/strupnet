# STRUPNET: structure-preserving neural networks

This package implements structure-preserving neural networks for learning dynamics of differential systems from data. 

## Installing 
Install it using pip: ```pip install strupnet```

## SympNet: Symplectic neural networks

### Basic example
```python 
import torch
from strupnet import SympNet

dim=2 # degrees of freedom for the Hamiltonian system. x = (p, q) \in R^{2*dim}
sympnet = SympNet(dim=dim, layers=12, width=8)

timestep = torch.tensor([0.1]) # time-step 
x0 = torch.randn(2 * dim) # phase space coordinate x0 = (p0, q0) 

x1 = sympnet(x0, timestep) # defines a random but symplectic transformation from x0 to x1
```
The rest of your code is identical to you how you would train any module that inherits from `torch.nn.Module`. 

<!-- # Contributing:

To add your own ```SympNet``` method/layer, do the following: 
- Create a new branch.
- Add a file to the ```sympnet/layers``` folder. Call it, for example, ```sympnet/layers/NEW_LAYER.py``` where NEW_LAYER is an abbreviation to the methods name (ideally no longer than a couple of letters). 
- In ```sympnet/layers/NEW_LAYER.py``` define a ```Layer``` class that inherits from ```torch.nn.Module```. 
- Define the forward method to accept an input of the form ```p, q, h``` and return the tuple ```p, q``` where ```p``` and ```q``` are of type ```torch.Tensor``` and shape ```(dim, )``` or ```(nbatch, dim)``` and ```h``` of shape ```(1, )``` or ```(nbatch, 1)```. 
- Add ```"NEW_LAYER"``` to the ```ALLOWED_METHODS``` list in ```sympnet.py```.
- Check that it passes the unit tests by running ```python -m pytest``` (Note that the tests will automatically test your new layer if it is added to ```ALLOWED_METHODS```). This tests for things like valid implementation and whether it is symplectic or not. 
- Create a pull request to the main branch. 

Otherwise, any contribution is appreciated! -->