import advertorch.attacks as _attacks
from advertorch.attacks.iterative_projected_gradient import PGDAttack, perturb_iterative, MomentumIterativeAttack, L2MomentumIterativeAttack, LinfMomentumIterativeAttack, L2BasicIterativeAttack, LinfBasicIterativeAttack

from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp


import numpy as np
import inspect

from typing import Optional
from rbi.attacks.base import Attack as _Attack
from rbi.utils.distributions import sample_lp_uniformly

from torch import Tensor
import numpy as np

import torch


class OverWriteDefaults:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_min = kwargs.get("clip_min", -1000)
        self.clip_max = kwargs.get("clip_max", 1000)


# Fancy metaclassing to overwrite these methods...
# In short this method does replace the original Attack Metaclass with the modified one.
_globals = globals()
for _name, _candidate in _attacks.__dict__.items():
    if inspect.isclass(_candidate) and issubclass(_candidate, _attacks.base.Attack):

        _params = {"__doc__": _candidate.__doc__}
        mro = list(_candidate.__mro__)
        idx = mro.index(_attacks.Attack)

        mro.insert(idx, _Attack)

        _globals[_name] = type(_name, tuple([OverWriteDefaults] + mro), _params)  # type: ignore


def project_lp(x, ord, dim=-1):
    return x / torch.linalg.norm(x, ord=ord, dim=dim, keepdim=True)

def perturb_iterative_adam(xvar, yvar, predict, nb_iter, eps, eps_iter, loss_fn, minimize, ord, clip_min, clip_max, delta_init, l1_sparsity):

    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    delta.requires_grad_(True)

    optim = torch.optim.Adam([delta], lr=eps_iter)
    for ii in range(nb_iter):
        outputs = predict(xvar + delta)
        loss = loss_fn(outputs, yvar)
        if not minimize:
            loss = -loss
        loss.backward()
        optim.step()
        with torch.no_grad():
            delta.data = eps*project_lp(delta.data, ord= ord)

    x_adv = clamp(xvar + delta, clip_min, clip_max)
    return x_adv




# Some changes to pgd


def pgd_randint_change_perturb(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
    """Attacks the input using PGD

    Args:
        x (Tensor): Input
        y (Optional[Tensor], optional): Target. Defaults to None.

    Raises:
        NotImplementedError: Ord which are not implemented

    Returns:
        Tensor: Perturbed input
    """
    x, y = self._verify_and_process_inputs(x, y)


    # Always rand init
    noise = sample_lp_uniformly(
        x.shape[:-1].numel(), x.shape[-1], p=self.ord, eps=self.eps, device=x.device  # type: ignore
    )
    delta = torch.tensor(noise, requires_grad=True)

    if self.ord in [1, 2, torch.inf, np.inf]:
        rval = perturb_iterative(
            x,
            y,
            self.predict,
            nb_iter=self.nb_iter,
            eps=self.eps,
            eps_iter=self.eps_iter,
            loss_fn=self.loss_fn,
            minimize=self.targeted,
            ord=self.ord,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            delta_init=delta,
            l1_sparsity=self.l1_sparsity,
        )
    else:
        raise NotImplementedError("Maybe extend...")

    return rval.data

def moment_randint_change_perturb(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
    """Attacks the input using PGD with moments.

    Args:
        x (Tensor): Input
        y (Optional[Tensor], optional): Target. Defaults to None.

    Raises:
        NotImplementedError: Ord which are not implemented

    Returns:
        Tensor: Perturbed input
    """
    x, y = self._verify_and_process_inputs(x, y)

    print(self.ord)
    delta = sample_lp_uniformly(
            x.shape[:-1].numel(), x.shape[-1], p=self.ord, eps=self.eps, device=x.device  # type: ignore
        )
    g = torch.zeros_like(x)

    delta = torch.tensor(delta, requires_grad=True)

    for i in range(self.nb_iter):

        if delta.grad is not None:
            delta.grad.detach_()
            delta.grad.zero_()

        imgadv = x + delta
        outputs = self.predict(imgadv)
        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()

        g = self.decay_factor * g + normalize_by_pnorm(
            delta.grad.data, p=1)  # type: ignore
        # according to the paper it should be .sum(), but in their
        #   implementations (both cleverhans and the link from the paper)
        #   it is .mean(), but actually it shouldn't matter
        if self.ord == np.inf:
            delta.data += batch_multiply(self.eps_iter, torch.sign(g))
            delta.data = batch_clamp(self.eps, delta.data)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x
        elif self.ord == 2:
            delta.data += self.eps_iter * normalize_by_pnorm(g, p=2)
            delta.data *= clamp(
                (self.eps * normalize_by_pnorm(delta.data, p=2) /
                    delta.data),
                max=1.)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x
        else:
            error = "Only ord = inf and ord = 2 have been implemented"
            raise NotImplementedError(error)

    rval = x + delta.data
    return rval



pgd_randint_change_perturb.__doc__ = PGDAttack.perturb.__doc__
setattr(PGDAttack, "perturb", pgd_randint_change_perturb)
setattr(L2BasicIterativeAttack, "perturb", pgd_randint_change_perturb)
setattr(LinfBasicIterativeAttack, "perturb", pgd_randint_change_perturb)
setattr(L2MomentumIterativeAttack, "perturb", moment_randint_change_perturb)
setattr(MomentumIterativeAttack, "perturb", moment_randint_change_perturb)
setattr(LinfMomentumIterativeAttack, "perturb", moment_randint_change_perturb)


# Some other simple attacks

class PGDAdamAttack(PGDAttack):

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)


        # Always rand init
        noise = sample_lp_uniformly(
            x.shape[:-1].numel(), x.shape[-1], p=self.ord, eps=self.eps, device=x.device  # type: ignore
        )
        delta = noise.clone().requires_grad_(True)

        rval = perturb_iterative_adam(
            x,
            y,
            self.predict,
            nb_iter=self.nb_iter,
            eps=self.eps,
            eps_iter=self.eps_iter,
            loss_fn=self.loss_fn,
            minimize=self.targeted,
            ord=self.ord,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            delta_init=delta,
            l1_sparsity=self.l1_sparsity,
        )

        return rval.data

class L1PGDAdamAttack(PGDAdamAttack):
    def __init__(self, predict, loss_fn=None, eps=0.3, nb_iter=40, eps_iter=0.1, rand_init=True, clip_min=-1e3, clip_max=1e3, ord=1., l1_sparsity=None, targeted=False):
        super().__init__(predict, loss_fn, eps, nb_iter, eps_iter, rand_init, clip_min, clip_max, ord, l1_sparsity, targeted)

class L2PGDAdamAttack(PGDAdamAttack):
    def __init__(self, predict, loss_fn=None, eps=0.3, nb_iter=40, eps_iter=0.1, rand_init=True, clip_min=-1e3, clip_max=1e3, ord=2., l1_sparsity=None, targeted=False):
        super().__init__(predict, loss_fn, eps, nb_iter, eps_iter, rand_init, clip_min, clip_max, ord, l1_sparsity, targeted)

class LinfPGDAdamAttack(PGDAdamAttack):
    def __init__(self, predict, loss_fn=None, eps=0.3, nb_iter=40, eps_iter=0.1, rand_init=True, clip_min=-1e3, clip_max=1e3, ord=torch.inf, l1_sparsity=None, targeted=False):
        super().__init__(predict, loss_fn, eps, nb_iter, eps_iter, rand_init, clip_min, clip_max, ord, l1_sparsity, targeted)
