from functools import reduce
from typing import Callable, Optional, Union
from rbibm.metric.base import RobustnessMetric
import torch
from torch import Tensor

from rbi.models.base import ParametricProbabilisticModel, PyroFlowModel
from rbi.utils.distributions import EmpiricalDistribution
from rbi.attacks.base import Attack
from rbi.loss.kernels import MultiKernel
from rbi.loss import (
    ReverseKLLoss,
    ForwardKLLoss,
    NegativeLogLikelihoodLoss,
    MMDsquared,
    MMDsquaredOptimalKernel,
)


class NLLRobMetric(RobustnessMetric):
    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        attack: Attack,
        targeted: bool = False,
        target_wrapper: Callable = lambda x: x,
        attack_attemps: int = 5,
        device: str = "cpu",
        mask: Optional[Tensor] = None,
        reduction: str = "mean_quantile95",
        batch_size: Optional[int] = None,
        descending: bool = True,
        **kwargs
    ):
        loss_fn = NegativeLogLikelihoodLoss(**kwargs)
        super().__init__(
            model,
            loss_fn=loss_fn,
            attack=attack,
            attack_attemps=attack_attemps,
            targeted=False,
            target_wrapper=target_wrapper,
            device=device,
            batch_size=batch_size,
            mask =mask,
            reduction=reduction,
            descending=descending,
        )

    def _get_goal(self, x, target):
        return target


class ReverseKLRobMetric(RobustnessMetric):
    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        attack: Attack,
        targeted: bool = False,
        target_wrapper: Callable = lambda x: x,
        attack_attemps: int = 5,
        device: str = "cpu",
        mask: Optional[Tensor] = None,
        reduction: str = "mean_quantile95",
        batch_size: Optional[int] = None,
        descending: bool = True,
        **kwargs
    ):
        loss_fn = ReverseKLLoss(**kwargs)
        super().__init__(
            model,
            loss_fn=loss_fn,
            attack=attack,
            attack_attemps=attack_attemps,
            targeted=targeted,
            target_wrapper=target_wrapper,
            device=device,
            mask=mask,
            batch_size=batch_size,
            reduction=reduction,
            descending=descending,
        )

    def _get_goal(self, x, target):
        if self.targeted :
            q =  target
        else:
            with torch.no_grad():
                q = self.model(x)

        return q
    
    def _ensure_loss_constraint(self, loss):
        # Very negative values can be produced do to overflow...
        # Small negative values can still be there as they can occur due to limited MC samples
        # -> Substitute with median.
        mask_neg = loss < - 10000
        loss[mask_neg] = loss[~mask_neg].median()
        return loss



class ForwardKLRobMetric(RobustnessMetric):
    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        attack: Attack,
        targeted: bool = False,
        target_wrapper: Callable = lambda x: x,
        attack_attemps: int = 5,
        device: str = "cpu",
        mask: Optional[Tensor] = None,
        reduction: str = "mean_quantile95",
        batch_size: Optional[int] = None,
        descending: bool = True,
        empirical_approx: bool = False,
        empirical_mc_samples = 20,
        **kwargs
    ):
        self.empirical_approx = empirical_approx
        self.empirical_mc_samples = empirical_mc_samples
        loss_fn = ForwardKLLoss(**kwargs)
        super().__init__(
            model,
            loss_fn=loss_fn,
            attack=attack,
            attack_attemps=attack_attemps,
            targeted=targeted,
            target_wrapper=target_wrapper,
            device=device,
            batch_size=batch_size,
            reduction=reduction,
            mask = mask,
            descending=descending,
        )

    def _get_goal(self, x, target):
        if self.targeted or self.attack.targeted:
            q = target
        else:
            with torch.no_grad():
                q = self.model(x)
                
        if self.empirical_approx:
            with torch.no_grad():
                q = EmpiricalDistribution(q.sample((self.empirical_mc_samples,)).transpose(0,1))

        return q
    
    def _ensure_loss_constraint(self, loss):
        # Very negative values can be produced do to overflow...
        # Small negative values can still be there as they can occur due to limited MC samples
        mask_neg = loss < - 10000
        loss[mask_neg] = loss[~mask_neg].median()
        return loss


class MMDsquaredRobMetric(RobustnessMetric):
    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        attack: Attack,
        kernel: Callable,
        targeted: bool = False,
        target_wrapper: Callable = lambda x: x,
        attack_attemps: int = 5,
        device: str = "cpu",
        mask: Optional[Tensor] = None,
        reduction: str = "mean_quantile95",
        batch_size: Optional[int] = None,
        descending: bool = True,
        **kwargs
    ):
        if not isinstance(kernel, MultiKernel):
            loss_fn = MMDsquared(kernel,  **kwargs)
        else:
            loss_fn = MMDsquaredOptimalKernel(kernel, **kwargs)

        super().__init__(
            model,
            loss_fn=loss_fn,
            attack=attack,
            attack_attemps=attack_attemps,
            targeted=targeted,
            target_wrapper=target_wrapper,
            device=device,
            batch_size=batch_size,
            reduction=reduction,
            mask=mask,
            descending=descending,
        )

    def _get_goal(self, x, target):
        if self.targeted:
            return target
        else:
            return self.model(x)
