import pytest 

import torch
from rbi.utils.lipschitz_tools import lipschitz_neural_net, check_lipschitz_continuouity, collect_lipschitz

import numpy as np

@pytest.fixture(params=[(2.,2.), (1.,1.),(torch.inf,torch.inf)], ids=["2", "1", "inf"])
def ord(request):
    return request.param

def test_lipschitz_for_parametric_families(parametric_family, device, ord):
    dim = parametric_family.input_dim
    net = parametric_family.net.to(device)

    ord1, ord2 = ord

    x = torch.randn(10, dim, device=device)
    y = net(x)
    assert torch.isfinite(y).all()
    lipschitz_neural_net(net, 1., ord1=ord1, ord2=ord2)
    y = net(x)
    assert torch.isfinite(y).all()

    check_lipschitz_continuouity(net, 1.,ord1=ord1, ord2=ord2)

    L_coll = collect_lipschitz(net)
    assert torch.isclose(L_coll.prod(), torch.ones(1, device=L_coll.device))


    

