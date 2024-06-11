import torch


def get_pq(x):
    """Returns p, q from x of shape (nbatch, 2*dim)"""
    dim = x.shape[-1] // 2
    return x[..., :dim], x[..., dim:]


def get_x(p, q):
    """Returns x from p, q of shape (nbatch, dim)"""
    return torch.cat([p, q], dim=-1)


def get_parameters(*dims, scale=0.01):
    """Returns an nn.Parameter tensor of the given dimensions from N(0, scale)."""
    return torch.nn.Parameter(torch.randn(*dims) * scale).requires_grad_(True)


def canonical_symplectic_matrix(dim):
    """Returns the canonical symplectic matrix of dimension 2*dim."""
    J = torch.zeros((2 * dim, 2 * dim))
    J[:dim, dim:] = -torch.eye(dim)
    J[dim:, :dim] = torch.eye(dim)
    return J


def symplectic_matrix_transformation_2d(w, i):
    """Used for the 2D symplectic volume-preserving substeps. Takes an n-dim vector w and an index i, and returns [0, ..., w[i+1], -w[i], ..., 0]"""
    out = torch.zeros_like(w)
    out[i] = w[i + 1]
    out[i + 1] = -w[i]
    return out
