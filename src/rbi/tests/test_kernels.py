
import pytest
import torch
from rbi.loss.kernels import RBFKernel, MultiDimRBFKernel, RationalQuadraticKernel, ConstantKernel, WhiteNoiseKernel, LinearKernel

@pytest.fixture(params=[RBFKernel, RationalQuadraticKernel, MultiDimRBFKernel])
def kernel_with_sigma_and_lengthscale(request):
    return request.param

@pytest.fixture(params=[ConstantKernel, WhiteNoiseKernel, LinearKernel])
def kernel_with_sigma(request):
    return request.param

@pytest.fixture(params=[0.1,0.5, 1., 2.])
def sigmas(request):
    return request.param 

@pytest.fixture(params=[0.1,0.5, 1., 2.])
def length_scales(request):
    return request.param 


def test_kernels_with_sigma_l(kernel_with_sigma_and_lengthscale, sigmas, length_scales):
    k = kernel_with_sigma_and_lengthscale(sigma = sigmas, l = length_scales)
    x = torch.randn(100, 5)
    K = k(x, x.unsqueeze(1))

    eigvals = torch.linalg.eigvalsh(K)
    assert eigvals.min() >= 0., "Not p.s.d kernel matrix"

