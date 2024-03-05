# STRUPNET: structure-preserving neural networks

This package implements structure-preserving neural networks for learning dynamics of differential systems from data. 

## Installing 
Install it using pip: ```pip install strupnet```

## Symplectic neural networks

### Basic example
```python 
import torch
from strupnet import SympNet

dim=2 # degrees of freedom for the Hamiltonian system. x = (p, q) \in R^{2*dim}

# Define a symplectic neural network with random parameters:
symp_net = SympNet(dim=dim, layers=12, width=8)

x0 = torch.randn(2 * dim) # phase space coordinate x = (p, q) 
h = torch.tensor([0.1]) # time-step 

x1 = symp_net(x0, h) # defines a random but symplectic transformation from x to X
```

### Training a SympNet
SympNet inherits from ```torch.nn.Module``` and can therefore be trained like a pytorch module.  Here is a minimal working example of training a SympNet using quadratic ridge polynomials (which is best for quadratic Hamiltonians)

#### Generating data
We will generate data of the form $` \{x(ih)\}_{i=0}^{n+1}=\{p(ih), q(ih)\}_{i=0}^{n+1}`$, where $`x(t)`$ is the solution to the Hamiltonian ODE $`\dot{x} = J\nabla H `$, with the simple Harmonic oscillator Hamiltonian $` H = \frac{1}{2} (p^2 + q^2) `$. The data is arranged in the form $` x_0 = \{x(ih)\}_{i=0}^{n} `$, $` x_1 = \{x((i+1)h)\}_{i=0}^{n} `$ and same for $` t `$. 
```python 
import torch 

# Generate training and testing data using simple harmonic oscillator solution
def simple_harmonic_oscillator_solution(t_start, t_end, timestep):
    time_grid = torch.linspace(t_start, t_end, int((t_end-t_start)/timestep)+1)
    p_sol = torch.cos(time_grid)
    q_sol = torch.sin(time_grid)
    pq_sol = torch.stack([p_sol, q_sol], dim=-1)
    return pq_sol, time_grid.unsqueeze(dim=1)

timestep=0.05

x_train, t_train = simple_harmonic_oscillator_solution(t_start=0, t_end=1, timestep=timestep)
x_test, t_test = simple_harmonic_oscillator_solution(t_start=1, t_end=4, timestep=timestep)

x0_train, x1_train, t0_train, t1_train = x_train[:-1, :], x_train[1:, :], t_train[:-1, :], t_train[1:, :]
x0_test, x1_test, t0_test, t1_test = x_test[:-1, :], x_test[1:, :], t_test[:-1, :], t_test[1:, :]
```
#### Training
We can train a SympNet like any PyTorch module on the loss function defined as follows. Letting $\Phi_h^{\theta}(x)$ denote the SympNet, where $\theta$ denotes its set of trainable parameters, then we want to find $\theta$ that minimises 

$\qquad loss=\sum_{i=0}^{n}\|\Phi_h^{\theta}(x(ih))-x\left((i+1)h\right)\|^2$

```python
from sympnet.sympnet import SympNet

# Initialize Symplectic Neural Network
symp_net = SympNet(dim=1, layers=2, max_degree=2, method="P")

# Train it like any other PyTorch model
optimizer = torch.optim.Adam(symp_net.parameters(), lr=0.01)
mse = torch.nn.MSELoss()
for epoch in range(1000):
    optimizer.zero_grad()    
    x1_pred = symp_net(x=x0_train, dt=t1_train - t0_train)
    loss = mse(x1_train, x1_pred)
    loss.backward()
    optimizer.step()
print("final loss value: ", loss.item())

x1_test_pred = symp_net(x=x0_test, dt=t1_test - t0_test)
print("test set error", torch.norm(x1_test_pred - x1_test).item())
```
Outputs:
```
Final loss value:  2.1763008371575767e-33
test set error 5.992433957888383e-16
```


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