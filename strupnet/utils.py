import torch


def get_parameters(*dims, scale=0.01):
    """Returns an nn.Parameter tensor of the given dimensions from N(0, scale)."""
    return torch.nn.Parameter(torch.randn(*dims) * scale).requires_grad_(True)


def canonical_symplectic_matrix(dim):
    """Returns the canonical symplectic matrix of dimension 2*dim."""
    J = torch.zeros((2 * dim, 2 * dim))
    J[:dim, dim:] = -torch.eye(dim)
    J[dim:, :dim] = torch.eye(dim)
    return J