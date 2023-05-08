import torch
from torch import Tensor

from torch.nn.init import _no_grad_trunc_normal_
from rbi.utils.distributions import sample_lp_uniformly

from rbi.attacks.base import Attack

from typing import Callable, Union


class UniformNoiseAttack(Attack):
    """General class for uniform noise on any lp hyperssphere."""

    def __init__(
        self,
        predict: Callable,
        eps: float = 0.5,
        ord: Union[float, str] = 2.0,
        **kwargs
    ) -> None:
        super().__init__(predict, **kwargs)
        self.eps = eps
        self.ord = ord

        # if self.targeted:
        #     raise ValueError("This attack cannot be targeted")

    def perturb(self, x: Tensor, *args,**kwargs) -> Tensor:
        with torch.no_grad():
            noise = sample_lp_uniformly(
                x.shape[0], x.shape[-1], p=self.ord, eps=self.eps, device=x.device
            )
            return torch.clip(x + noise, min=self.clip_min, max=self.clip_max)


class L2UniformNoiseAttack(UniformNoiseAttack):
    """Uniform samples form L2 Spheres"""

    def __init__(self, predict, eps=0.5, **kwargs) -> None:
        super().__init__(predict, eps, ord=2.0, **kwargs)


class L1UniformNoiseAttack(UniformNoiseAttack):
    """Uniform samples form 12 Spheres"""

    def __init__(self, predict, eps=0.5, **kwargs) -> None:
        super().__init__(predict, eps, ord=1.0, **kwargs)


class LinfUniformNoiseAttack(UniformNoiseAttack):
    """Uniform samples form Linf Spheres"""

    def __init__(self, predict, eps=0.5, **kwargs) -> None:
        super().__init__(predict, eps, ord="inf", **kwargs)


class GaussianNoiseAttack(Attack):
    """Gaussian perturbation with zero mean"""

    def __init__(self, predict, eps=0.5, **kwargs) -> None:
        super().__init__(predict, **kwargs)
        self.eps = eps

        if self.targeted:
            raise ValueError("This attack cannot be targeted")

    def perturb(self, x, **kwargs):
        with torch.no_grad():
            x_pert = x + torch.randn_like(x, device=x.device) * self.eps
            return torch.clip(x_pert, min=self.clip_min, max=self.clip_max)


class TruncatedGaussianNoiseAttack(GaussianNoiseAttack):
    """Truncated Gaussian perturbations"""

    def __init__(self, predict, eps=0.5, scale=0.5, **kwargs) -> None:
        super().__init__(predict, scale, **kwargs)
        self.eps = eps

    def perturb(self, x, **kwargs):
        with torch.no_grad():
            x_pert = x + _no_grad_trunc_normal_(
                torch.zeros_like(x, device=x.device), 0.0, self.eps, -self.eps, self.eps
            ).to(x.device)
            return torch.clip(x_pert, min=self.clip_min, max=self.clip_max)
