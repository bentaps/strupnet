import pytest
import torch

from ..snn import SympNet, ALLOWED_METHODS

sympnet_kwargs = dict(
    layers=8,
    width=8,
    min_degree=2,
    max_degree=4,
    sublayers=5,
)


def canonical_symplectic_matrix(dim):
    J = torch.zeros(2 * dim, 2 * dim)
    J[:dim, dim:] = torch.eye(dim)
    J[dim:, :dim] = -torch.eye(dim)
    return J


@pytest.mark.parametrize("method", ALLOWED_METHODS)
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("dim", [1, 2])
def test_sympnet(method, symmetric, dim, kwargs=sympnet_kwargs):
    # Create batch of random data
    n_batch = 10
    x0 = torch.randn(n_batch, 2 * dim)
    dt = torch.tensor([0.1])

    # Create SympNet instance
    net = SympNet(method=method, symmetric=symmetric, dim=dim, **kwargs)
    output = net(x0, dt)
    J = canonical_symplectic_matrix(dim)

    # Check for correct output shape
    assert output.shape == (n_batch, 2 * dim), "Output shape is incorrect!"
    for i in range(n_batch): # Check for symplecticity 
        D = torch.autograd.functional.jacobian(lambda x0: net(x0, dt), x0[i, :])
        assert torch.allclose(D.T @ J @ D, J), "Method not sympelctic!"

    if symmetric: # Check for time-symmetry
        for i in range(n_batch):
            x0_tmp = x0[i, :]
            forward = net(x0_tmp, dt)
            x0_tmp[dim:] *= -1 
            backward = net(x0_tmp, -dt)
            backward[dim:] *= -1 
            tol =  1e-8 
            assert torch.allclose(forward, backward, atol=tol), f"Time symmetry not enforced when symmetric=True! error = {forward - backward}"
