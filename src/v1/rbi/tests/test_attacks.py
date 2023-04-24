from numpy import isin
import pytest
import torch

from rbi.attacks import (
    GaussianNoiseAttack,
    TruncatedGaussianNoiseAttack,

)

from rbi.loss import ForwardKLLoss




def test_attacks(model_2d, attack, eps, device):

    model_2d = model_2d.to(device)
   
    input_dim = model_2d.input_dim

    a1 = attack(model_2d, eps=eps, loss_fn = ForwardKLLoss(mc_samples=1))

    X = torch.randn((5, input_dim), device=device)
    X_pert1 = a1.perturb(X)
    try:
        with torch.no_grad():
            t = model_2d(torch.randn_like(X))
        X_pert1_targeted = a1.perturb(X, t)
    except:
        t = None
        X_pert1_targeted = None

    assert (
        X.shape == X_pert1.shape
    ), "Mhh, the shapes do not match. Something is wrong!"

    try:
        assert (X != X_pert1).any(), "Mhh, the attack did not change the input at all..."
    except:
        s1 = model_2d(X).sample()
        if t is not None:
            s2 = t.sample()
        else:
            s2 = None
        
        cond = (X_pert1_targeted != X)
        if isinstance(cond, torch.Tensor):
            cond = cond.any()
        assert cond or (s1 == s2).all(), "Mhh, the attack did not change the input at all..."



    if not attack in [GaussianNoiseAttack, TruncatedGaussianNoiseAttack]:
        assert (
            torch.linalg.norm(X - X_pert1, ord=torch.inf, dim=-1).max() <= eps + 1e-2
        ), f"The eps constraints seems not to hold it is {torch.linalg.norm(X - X_pert1, ord=torch.inf, dim=-1).max() }"

