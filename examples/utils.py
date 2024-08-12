import io
import matplotlib.pyplot as plt
import imageio
import torch
import numpy as np


def canonical_symplectic_matrix(n):
    J = torch.zeros((2 * n, 2 * n))
    J[:n, n:] = -torch.eye(n)
    J[n:, :n] = torch.eye(n)
    return J

def hamiltonian_exp_map(n, h):
    # Symplectic matrix
    J = canonical_symplectic_matrix(n)
    
    # Discrete Laplacian matrix
    L = torch.eye(n)
    L -= torch.roll(L, 1, dims=1)
    L -= torch.roll(L, -1, dims=1)
    L *= -1

    # Mass matrix
    M = torch.eye(n)

    # Hamiltonian matrix
    A = torch.zeros((2*n, 2*n))
    A[:n, :n] = M 
    A[n:, n:] = -L
    
    return torch.matrix_exp(h*J@A) 

def generate_linear_hamiltonian_trajectory(p0, q0, h, nsteps):
    x = torch.cat([p0, q0])
    dim = x.shape[0]
    data = torch.zeros(nsteps+1, dim)
    flow_map = hamiltonian_exp_map(dim//2, h)
    for i in range(nsteps+1):
        data[i] = x
        x = flow_map @ x
    return data

def generate_linear_hamiltonian_data(dim, ndata, timestep, lims=[-1, 1], nsteps=1):
    data = torch.zeros(ndata, 2, 2*dim)
    for i in range(ndata):
        # random initial condition
        p0 = lims[0] + (lims[1] - lims[0])*torch.rand(dim)
        q0 = lims[0] + (lims[1] - lims[0])*torch.rand(dim)
        data[i, :, :] = generate_linear_hamiltonian_trajectory(p0, q0, timestep, nsteps=1)
    x0, x1 = data[:, :-1, :].reshape(-1, 2*dim), data[:,1:, :].reshape(-1, 2*dim)
    return x0, x1


def create_gif(solution, exact_solution=None, title='gif', duration=0.05):
    """
    Create a GIF from a numpy array of shape (dim, nt).

    :param solution: numpy array of shape (dim, nt)
    :param filename: name of the output GIF file
    :param duration: duration of each frame in the GIF
    """
    filename = title + '.gif'
    fig, ax = plt.subplots()
    images = []

    # Create a plot for each time step and save as an image
    for i in range(solution.shape[0]):
        ax.clear()  # Clear previous frame
        ax.plot(solution[i, :], 'o', label='predicted')
        if exact_solution is not None:
            ax.plot(exact_solution[i, :], 'ko', label='exact', alpha=0.5)
        ax.set_ylim([-1, 1])  # Fix the y-axis scale for consistency
        ax.set_title(f'{title}\ntime step: {i+1}')
        ax.legend(loc='upper right')
        # Save the plot as a PNG image in memory (not on disk)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images.append(imageio.imread(buf))


    # Create a GIF from the images
    imageio.mimsave(filename, images, duration=duration, loop=0)

    print(f'GIF saved as {filename}')


def train(net, x0, x1, lr=0.01, nepochs=4000, tol=1e-13, timestep=0.1):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    training_curve = torch.zeros(nepochs)
    for epoch in range(nepochs):
        optimizer.zero_grad()    
        x1_pred = net(x=x0, dt=timestep)
        loss = mse(x1, x1_pred)
        loss.backward()
        optimizer.step()
        if loss.item() < tol:
            break
        if epoch % 100 == 0:
            print("Epoch: ", epoch, " Loss: ", loss.item())
        training_curve[epoch] = loss.item()
    return training_curve
    print("Final loss value: ", loss.item())