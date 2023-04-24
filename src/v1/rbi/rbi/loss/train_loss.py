
from rbi.loss.base import TrainLoss

import torch
from torch import Tensor
from torch.distributions import Distribution


from typing import Optional, Tuple, List, Callable, Union
from sbi.utils.metrics import c2st

from rbi.models.base import ParametricProbabilisticModel, PyroFlowModel
from rbi.loss.kl_div import set_mc_budget, kl_divergence


class NLLLoss(TrainLoss):
    def _loss(self, output: Distribution, target: Tensor, input: Tensor) -> Tensor:
        if not isinstance(output, Distribution):
            # If parameters are returned than make a distribution out of them
            output = self.model.generator(output)  # type: ignore

        if not isinstance(target, Tensor):
            raise ValueError(
                r"This loss function requires as input events $y\sim p(y|x)$, which must be given as tensors."
            )
        target = target
        sample_shape = output.batch_shape + output.event_shape

        return -output.log_prob(target.reshape(*sample_shape))


class NegativeElboLoss(TrainLoss):
    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        potential_fn: Optional[Callable] = None,
        loglikelihood_fn: Optional[Callable] = None,
        prior: Optional[Distribution] = None,
        method: str = "prior_contrastive",
        reduction: str = "mean",
        mc_samples: int = 8,
    ) -> None:
        super().__init__(model, reduction)

        if (potential_fn is None) and (loglikelihood_fn is None and prior is None):
            raise ValueError(
                "We require for this loss that either an potential function is given or a likelihood or a prior!"
            )
        elif potential_fn is not None and loglikelihood_fn is None:
            method = "joint_contrastive"
        elif (
            potential_fn is None and loglikelihood_fn is not None and prior is not None
        ):
            method = "prior_contrastive"

        self.mc_samples = mc_samples
        self.potential_fn = potential_fn
        self.loglikelihood_fn = loglikelihood_fn
        self.prior = prior
        self.method = method

    def _loss(self, output: Distribution, target: Tensor, input: Tensor) -> Tensor:
        if self.method == "joint_contrastive":
            return self._loss_joint_contrastive(output, target, input)
        elif self.method == "prior_contrastive":
            return self._loss_prior_contrastive(output, target, input)
        else:
            raise ValueError("Unknown method...")

    def _loss_joint_contrastive(
        self, output: Distribution, target: Tensor, input: Tensor
    ):

        if output.has_rsample:
            samples_q = output.rsample((self.mc_samples,))  # type: ignore
        else:
            samples_q = output.sample((self.mc_samples,))  # type: ignore
        potential = self.potential_fn(input, samples_q)  # type: ignore

        mask = torch.isfinite(potential)
        potential[~mask] = -100
        potential = potential.mean(0)

        logq = output.log_prob(samples_q)
        return logq - potential

    def _loss_prior_contrastive(
        self, output: Distribution, target: Tensor, input: Tensor
    ):
        if output.has_rsample:
            samples_q = output.rsample((self.mc_samples,))  # type: ignore
        else:
            samples_q = output.sample((self.mc_samples,))  # type: ignore

        logll = self.loglikelihood_fn(samples_q).log_prob(input).mean(0)  # type: ignore
        set_mc_budget(self.mc_samples)
        kl_term = kl_divergence(output, self.prior)

        return -logll + kl_term
