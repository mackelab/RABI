import pytest
import torch
from torch import nn

from rbi.utils.autograd_tools import batch_jacobian, batch_hessian

# Some special fixtures for benchmarks

@pytest.fixture
def mlp_2_2():
    return nn.Sequential(nn.Linear(2, 100), nn.ReLU(), nn.Linear(100, 2))


@pytest.fixture
def mlp_2_100():
    return nn.Sequential(nn.Linear(2, 100), nn.ReLU(), nn.Linear(100, 100))


@pytest.fixture
def mlp_100_2():
    return nn.Sequential(nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 2))


@pytest.fixture
def deep_nn():
    return nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.Sigmoid(),
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 10),
        nn.Identity(),
        nn.Linear(10, 2),
    )


def test_batch_jacobian_mlp(mlp):
    # No batch
    mlp, input_dim, output_dim = mlp
    x = torch.randn(input_dim)

    j = batch_jacobian(mlp, x)

    assert j.shape[-1] == input_dim, "Input dim is 2, thus also fist jacobian dim must be 2"
    assert j.shape[-2] == output_dim, "Output dim is 2, thus also second jacobian dim must be 2"
    assert len(j.shape) == 2, "No batch dim"
    assert j.sum() != 0, "Should be nonzero."

    # Check if gradients work
    x = torch.randn(input_dim, requires_grad=True)
    j = batch_jacobian(mlp, x, create_graph=True)
    j.sum().backward()

    only_relu = all([isinstance(l,nn.ReLU) or isinstance(l, nn.Linear) for l in mlp])

    if only_relu:
        assert (
            x.grad is not None and x.grad.sum() == 0
        ), "Gradients worked... . Gradient of gradient of ReLU should be zero almost everwhere. "
    else:
        assert (
            x.grad is not None and x.grad.sum().abs() > 0
        ), "Gradients worked... . Gradient of gradient of ReLU should be zero almost everwhere. "

    x = torch.randn(10, input_dim)
    j = batch_jacobian(mlp, x)

    assert j.shape[-1] == input_dim, "Input dim is 2, thus also fist jacobian dim must be 2"
    assert j.shape[-2] == output_dim, "Output dim is 2, thus also second jacobian dim must be 2"
    assert j.shape[0] == 10, "Batch dimensions wrong"
    assert j.sum() != 0, "Should be nonzero."

    x = torch.randn(10, 1, 10, input_dim)
    j = batch_jacobian(mlp, x)

    assert j.shape[-1] == input_dim, "Input dim is 2, thus also fist jacobian dim must be 2"
    assert j.shape[-2] == output_dim, "Output dim is 2, thus also second jacobian dim must be 2"
    assert j.shape[:-2] == x.shape[:-1], "Batch dimensions wrong"
    assert j.sum() != 0, "Should be nonzero."


def test_batch_hessian_mlp(mlp):
    # No batch
    mlp, input_dim, output_dim = mlp
    x = torch.randn(input_dim)

    _func = lambda x: mlp(x).sum(-1, keepdim=True)

    h = batch_hessian(_func, x)

    only_relu = all([isinstance(l,nn.ReLU) or isinstance(l, nn.Linear) for l in mlp])

    assert h.shape[-2] == input_dim, "Input dim is 2, thus also Hessian dim must be 2"
    assert h.shape[-1] == input_dim, "Input dim is 2, thus also second jacobian dim must be 2"
    assert len(h.shape) == 2, "No batch dim"
    if only_relu:
        assert h.sum() == 0, "Should be zero of RELU networks."
    else:
        assert h.sum().abs() > 0, "Should be zero of RELU networks."



    if only_relu:
        assert h.sum() == 0, "Should be zero of RELU networks."
    else:
        assert h.sum().abs() > 0, "Should be zero of RELU networks."

    x = torch.randn(10, input_dim)
    h = batch_hessian(_func, x)

    assert h.shape[-2] == input_dim, "Input dim is 2, thus also Hessian dim must be 2"
    assert h.shape[-1] == input_dim, "Input dim is 2, thus also second jacobian dim must be 2"
    assert h.shape[0] == 10, "Batch dimensions wrong"
    if only_relu:
        assert h.sum() == 0, "Should be zero of RELU networks."
    else:
        assert h.sum().abs() > 0, "Should be zero of RELU networks."

    x = torch.randn(10, 1, 10, input_dim)
    h = batch_hessian(_func, x)

    assert h.shape[-2] == input_dim, "Input dim is 2, thus also fist jacobian dim must be 2"
    assert h.shape[-1] == input_dim, "Output dim is 2, thus also second jacobian dim must be 2"
    assert h.shape[:-2] == x.shape[:-1], "Batch dimensions wrong"
    if only_relu:
        assert h.sum() == 0, "Should be zero of RELU networks."
    else:
        assert h.sum().abs() > 0, "Should be zero of RELU networks."


def test_correctness_linear_hessian(linear):

    x = torch.randn(100, linear.dim)
    h = batch_hessian(lambda x: linear(x).sum(-1, keepdim=True), x)

    assert h.shape[-2] == linear.dim and h.shape[-1] == linear.dim, "Shapes are right"
    assert torch.isclose(h.sum(), torch.zeros(1), atol=1e5), "Jacobain of Wx is zero"


def test_correctness_quadratic_hessian(quadratic):
    true_hessian = quadratic.hessian
    x = torch.randn(100, quadratic.dim)
    j = batch_hessian(quadratic, x)

    assert torch.isclose(j, true_hessian(x), atol=1e-5).all(), "Hessian do not match"


def test_benchmark_100_batch_jacobian_mpl_2_2(benchmark, mlp_2_2):
    x = torch.randn(100, 2)
    benchmark(lambda: batch_jacobian(mlp_2_2, x))


def test_benchmark_100_batch_jacobian_deep_nn(benchmark, deep_nn):
    x = torch.randn(100, 2)
    benchmark(lambda: batch_jacobian(deep_nn, x))


def test_benchmark_10000_batch_jacobian_mpl_2_2(benchmark, mlp_2_2):
    x = torch.randn(10000, 2)
    benchmark(lambda: batch_jacobian(mlp_2_2, x))


def test_benchmark_10000_batch_jacobian_deep_nn(benchmark, deep_nn):
    x = torch.randn(10000, 2)
    benchmark(lambda: batch_jacobian(deep_nn, x))


def test_benchmark_100_batch_hessian_mpl_2_2(benchmark, mlp_2_2):
    x = torch.randn(100, 2)
    benchmark(lambda: batch_hessian(lambda x: mlp_2_2(x).sum(-1, keepdim=True), x))


def test_benchmark_100_batch_hessian_deep_nn(benchmark, deep_nn):
    x = torch.randn(100, 2)
    benchmark(lambda: batch_hessian(lambda x: deep_nn(x).sum(-1, keepdim=True), x))


def test_benchmark_10000_batch_hessian_mpl_2_2(benchmark, mlp_2_2):
    x = torch.randn(10000, 2)
    benchmark(lambda: batch_hessian(lambda x: mlp_2_2(x).sum(-1, keepdim=True), x))


def test_benchmark_10000_batch_hessain_deep_nn(benchmark, deep_nn):
    x = torch.randn(10000, 2)
    benchmark(lambda: batch_hessian(lambda x: deep_nn(x).sum(-1, keepdim=True), x))
