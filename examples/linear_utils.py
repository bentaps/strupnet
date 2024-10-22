import io
from math import e
from os import times
from re import M
import matplotlib.pyplot as plt
import imageio
import torch
from tqdm import tqdm
from scipy.linalg import expm
import scipy 
import numpy as np 

def canonical_symplectic_matrix(n):
    J = torch.zeros((2 * n, 2 * n))
    J[:n, n:] = -torch.eye(n)
    J[n:, :n] = torch.eye(n)
    return J

def circulant_matrix(n):
    x = torch.linspace(-1, 1, n)
    v = torch.cos(np.pi * x)
    A = np.zeros((n, n))  # Initialize an n x n matrix
    for i in range(n):
        A[i] = np.roll(v, i)  # Shift v to the right by i
    return A

def laplacian_matrix(n):
    # Discrete Laplacian matrix
    L = torch.eye(n)
    L -= torch.roll(L, 1, dims=1)
    L -= torch.roll(L, -1, dims=1)
    L *= -1

    # Mass matrix
    M = torch.eye(n)

    # Symmetric matrix
    A = torch.zeros((2 * n, 2 * n))
    A[:n, :n] = M
    A[n:, n:] = -L
    return A



def dense_random_symmetric_matrix(n):
    A = torch.zeros(2 * n, 2 * n)
    A[:, :] += torch.rand(2*n, 2*n)

    A = (A + A.t()) / 2
    A *= 0.5
    A += torch.eye(2*n) 
    return A

def separable_random_symmetric_matrix(n):
    A = torch.eye(2 * n)
    I = torch.eye(n)
    V = torch.rand(n, n)*0.5
    T = torch.rand(n, n)*0.5
    V = (V + V.t()) / 2
    T = (T + T.t()) / 2
    A[n:, n:] += V
    A[:n, :n] += T
    A[:n, n:] += I
    A[n:, :n] += I
    return A


def hamiltonian_matrix_exponential(h, A):
    n = A.shape[0] // 2
    J = canonical_symplectic_matrix(n)
    # use scipy.linalg.expm (Pade approximation), instead of torch.matrix_exp (Taylor approximation)
    # return torch.tensor(expm(h * J @ A))
    return torch.matrix_exp(h * J @ A)

def spectrum_adjoint_action(X):
    eigs = np.linalg.eigvals(X)
    ad_eigs = np.array([eigs[i] - eigs[j] for i in range(len(eigs)) for j in range(len(eigs))])
    ad_eigs.sort()
    return ad_eigs

def generate_linear_hamiltonian_trajectory(p0, q0, h, nsteps, symmetric_matrix, device=None, method="matrix_exp"):
    x = torch.cat([p0, q0])
    dim = x.shape[0]
    data = torch.zeros(nsteps + 1, dim)
    
    A = symmetric_matrix 
    assert A.allclose(A.t()), "A is not symmetric"

    if method == "matrix_exp":
        matrix_exp = hamiltonian_matrix_exponential(h, A)
        if device:
            matrix_exp = matrix_exp.to(device=device)
            data = data.to(device=device)
        for i in range(nsteps + 1):
            data[i] = x
            x = matrix_exp @ x
        return data
    elif method == "odeint":
        # compute the solution to the linear Hamiltonian system using scipy.odeint 
        from scipy.integrate import odeint
        def linear_hamiltonian(x, t):
            x = torch.tensor(x)
            return (J @ A @ x).numpy()
        J = canonical_symplectic_matrix(dim // 2)
        t = torch.arange(0, nsteps * h + h, h).numpy()
        x = x.numpy()
        x = odeint(linear_hamiltonian, x, t, atol=1e-14, rtol=1e-14)
        return torch.tensor(x)
    else:
        raise ValueError("method must be 'matrix_exp' or 'scipy'")
        

def generate_linear_hamiltonian_data(
    dim, ndata, timestep, symmetric_matrix, lims=[-1, 1], nsteps=1, device=None, method="matrix_exp"
):
    data = torch.zeros(ndata, 2, 2 * dim)
    if device:
        data = data.to(device=device)
    for i in range(ndata):
        # random initial condition
        p0 = lims[0] + (lims[1] - lims[0]) * torch.rand(dim)
        q0 = lims[0] + (lims[1] - lims[0]) * torch.rand(dim)
        data[i, :, :] = generate_linear_hamiltonian_trajectory(
            p0, q0, timestep, symmetric_matrix=symmetric_matrix, nsteps=nsteps, device=device, method=method
        )
    x0, x1 = data[:, :-1, :].reshape(-1, 2 * dim), data[:, 1:, :].reshape(-1, 2 * dim)
    return x0, x1


def create_gif(solution, exact_solution=None, title="gif", duration=0.05):
    """
    Create a GIF from a numpy array of shape (dim, nt).

    :param solution: numpy array of shape (dim, nt)
    :param filename: name of the output GIF file
    :param duration: duration of each frame in the GIF
    """
    filename = title + ".gif"
    fig, ax = plt.subplots()
    images = []

    # Create a plot for each time step and save as an image
    for i in range(solution.shape[0]):
        ax.clear()
        ax.plot(solution[i, :], "o", label="predicted")
        if exact_solution is not None:
            ax.plot(exact_solution[i, :], "kx", label="exact", alpha=0.5)
        ax.set_ylim([-1, 1])  # Fix the y-axis scale for consistency
        ax.set_title(f"{title}\ntime step: {i+1}")
        ax.legend(loc="upper right")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        images.append(imageio.imread(buf))
    imageio.mimsave(filename, images, duration=duration, loop=0)
    print(f"GIF saved as {filename}")

def train(
    net,
    x0,
    x1,
    timestep,
    lr=0.01,
    nepochs=4000,
    tol=1e-18,  # Lower tolerance
    use_best_train_loss=True,
):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    training_curve = torch.zeros(nepochs)
    progress_bar = tqdm(range(nepochs))
    best_train_loss = torch.tensor(float("inf"))
    best_state_dict = None  # Ensure proper initialization
    for epoch in progress_bar:
        optimizer.zero_grad()  # Clear previous gradients
        x1_pred = net(x=x0, dt=timestep)
        loss = mse(x1, x1_pred)
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the model parameters
        
        if use_best_train_loss:
            if loss.item() < best_train_loss:
                best_state_dict = net.state_dict()
                best_train_loss = loss.item()

        # Print gradients
        norm_grads = sum(param.grad.norm().item() for param in net.parameters())

        progress_bar.set_postfix(
            {"train_loss": loss.item(), "best_train_loss": best_train_loss, "norm_grads": norm_grads}
        )

        training_curve[epoch] = loss.item()
        
        if loss.item() < tol or norm_grads < tol:  # Early stopping if loss or grads reach tol
            break
        
    
    print("Final loss value: ", loss.item())
    
    # Only load the best state if it was saved
    if best_state_dict is not None:
        net.load_state_dict(best_state_dict)
    
    return training_curve, best_train_loss
