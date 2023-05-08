from abc import abstractmethod
from typing import Any, Callable, Optional, Tuple, Union
from pkg_resources import Distribution

import torch
from rbi.attacks.custom_attacks import Attack
from rbi.loss.base import EvalLoss
from rbi.models.base import ParametricProbabilisticModel, PyroFlowModel
from torch import Tensor
from rbibm.tasks.base import Simulator

from rbibm.utils.batched_processing import eval_function_batched_sequential


class NoAttack(Attack):
    def perturb(self, x):
        return x


def generate_adversarial_target_by_permutation(model, x):
    perm = torch.randperm(x.shape[0], device=x.device)
    x_perm = x[perm]
    q_perm = model(x_perm)

    return q_perm


class Metric:
    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        reduction: Optional[str] = None,
        batch_size: Optional[int] = None,
        descending: bool = False,
        device: str = "cpu",
    ) -> None:
        """Base class for any metric

        Args:
            model (Union[ParametricProbabilisticModel, PyroFlowModel]): Model to evaluate
            reduction (Optional[str], optional): Reduction methods. Defaults to None.
            batch_size (Optional[int], optional): Batch size for evaluation. Defaults to None.
            descending (bool, optional): If the loss is ascending or descending. Defaults to False.
            device (str, optional): Device to compute on. Defaults to "cpu".
        """
        self.model = model.to(device)
        self._reduction = reduction
        self.batch_size = batch_size
        self.descending = descending
        self.device = device

    @abstractmethod
    def _eval(self, x, other=None) -> Tensor:
        pass

    @property
    def reduction(self):
        """The method for reducing evaluations."""
        return self._reduction

    @reduction.setter
    def reduction(self, x: Optional[str]) -> None:
        self._set_reduction(x)

    def _set_reduction(self, x: Optional[str]) -> None:
        if x is None:
            self._reduction = None
        elif isinstance(x, str):
            vars = x.split("_")
            main_out = vars[0] in ["mean", "median", "sum", "max", "min"]
            if len(vars) > 1:
                supp_out = vars[1] in ["std", "q95", "summary"]
            else:
                supp_out = True

            if main_out and supp_out:
                self._reduction = x
            else:
                raise ValueError("Unknown reduction type")

        else:
            raise ValueError("Unknown reduction type")

    def _reduce(self, m: Tensor) -> Union[Tensor, Tuple]:

        if self.reduction is None:
            return m

        vars = self.reduction.split("_")
        main_out = vars[0]
        if len(vars) > 1:
            supp_out = vars[1]
        else:
            supp_out = None

        mask = torch.isfinite(m)
        m = m[mask]

        if main_out == "mean":
            out = m.mean()
        elif main_out == "median":
            out = m.median()
        elif main_out == "sum":
            out = m.sum()
        elif main_out == "max":
            out = m.max()
        elif main_out == "min":
            out = m.min()
        else:
            out = m

        if supp_out is not None:
            if supp_out == "std":
                supp = {"std": m.std()}
            elif supp_out == "q95":
                supp = {"q02_5": m.quantile(0.025), "q97_5": m.quantile(0.975)}
            elif supp_out == "summary":
                supp = {
                    "q00_5": m.quantile(0.005),
                    "q02_5": m.quantile(0.025),
                    "q05": m.quantile(0.05),
                    "q15": m.quantile(0.15),
                    "q20": m.quantile(0.20),
                    "q30": m.quantile(0.30),
                    "q50": m.quantile(0.50),
                    "q70": m.quantile(0.70),
                    "q80": m.quantile(0.80),
                    "q85": m.quantile(0.85),
                    "q95": m.quantile(0.95),
                    "q97_5": m.quantile(0.975),
                    "q99_5": m.quantile(0.995),
                    "min": m.min(),
                    "max": m.max(),
                    "std": m.std(),
                }
            else:
                supp = None
        else:
            supp = None

        if supp is None:
            return out
        else:
            return out, supp

    def eval(self, x: Tensor, other: Any = None):

        if x.ndim == 1:
            x = x.unsqueeze(0)

        batch_n = x.shape[0]

        if self.batch_size is not None and self.batch_size < batch_n:
            out = eval_function_batched_sequential(
                self._eval,
                x,
                other,
                batch_size=self.batch_size,
                dim=0,
                device=self.device,
            )
            out = self._reduce(out)
        else:
            x = x.to(self.device)
            if other is not None and hasattr(other, "to"):
                other = other.to(self.device)
            out = self._eval(x, other)
            out = self._reduce(out)

        return out


class RobustnessMetric(Metric):
    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        loss_fn: EvalLoss,
        attack: Attack,
        attack_attemps: int = 5,
        targeted: bool = False,
        target_wrapper: Callable = lambda x: x,
        mask: Optional[Tensor] = None,
        descending: bool = True,
        batch_size: Optional[int] = None,
        reduction: Optional[str] = None,
        device: str = "cpu",
    ):
        """This evaluates the robustness of the model as max_d D(q(x + d) || q(x)) or min_d D(q(x + d) || t) for some target.

        Args:
            model (Union[ParametricProbabilisticModel, PyroFlowModel]): Model to evaluate
            loss_fn (EvalLoss): Loss function used for evaluation
            attack (Attack): Attack
            attack_attemps (int, optional): Attack attemps. Defaults to 5.
            targeted (bool, optional): Targets. Defaults to False.
            descending (bool, optional): If lower values are better or worse. Defaults to True.
            batch_size (Optional[int], optional): Batch size used in computation. Defaults to None.
            reduction (Optional[str], optional): Reduction in report. Defaults to None.
            device (str, optional): Device. Defaults to "cpu".
        """
        super().__init__(
            model,
            reduction=reduction,
            batch_size=batch_size,
            descending=descending,
            device=device,
        )
        self.loss_fn = loss_fn
        self.loss_fn.reduction = None  # type: ignore
        self.attack_attemps = attack_attemps

        # Attack stuff
        self.targeted = targeted
        self.target_wrapper = target_wrapper
        self.attack = attack

        # Mask
        self.mask = mask

        # Cache
        self._cached_x = None
        self._cached_xs_tilde = None
        self._cached_losses = None

    def _eval(self, x: Tensor, target: Optional[Any] = None) -> Tensor:

        target = self.target_wrapper(target)

        losses = None
        xs_tilde = None

        for i in range(self.attack_attemps):

            if self.mask is not None:    

                def _masked_model(x_mask):
                    x_help = x.clone().detach()
                    x_help[...,self.mask] = x_mask
                    return self.model(x_help)
                
                self.attack.predict = _masked_model

            if target is None or not self.attack.targeted:
                if self.mask is None:
                    x_tilde = self.attack.perturb(x)
                else:
                    x_tilde_mask = self.attack.perturb(x[...,self.mask])
                    x_tilde = x.clone()
                    x_tilde[...,self.mask] = x_tilde_mask
            else:
                if target is None:
                    raise ValueError("No target is specified ...")
                if self.mask is None:
                    x_tilde = self.attack.perturb(x, target)
                else:
                    x_tilde_mask = self.attack.perturb(x[...,self.mask], target)
                    x_tilde = x.clone()
                    x_tilde[...,self.mask] = x_tilde_mask


            with torch.no_grad():
                q_tilde = self.model(x_tilde)
                q = self._get_goal(x, target)

                if isinstance(q_tilde, torch.distributions.Distribution) or isinstance(
                    q, torch.Tensor
                ):
                    # For loglikelihood loss...
                    loss = self.loss_fn(q_tilde, q)
                elif isinstance(q, torch.distributions.Distribution) or isinstance(
                    q_tilde, torch.Tensor
                ):
                    loss = self.loss_fn(q, q_tilde)
                else:
                    loss = self.loss_fn(q, q_tilde)

                loss = self._ensure_loss_constraint(loss)

            if i == 0:
                xs_tilde = x_tilde.detach()
                losses = loss.detach()
            else:
                if self.descending:
                    if not self.targeted:
                        mask_better_attack = loss > losses
                    else:
                        mask_better_attack = loss < losses

                else:
                    if not self.targeted:
                        mask_better_attack = loss < losses
                    else:
                        mask_better_attack = loss > losses

                xs_tilde[mask_better_attack] = x_tilde[mask_better_attack].detach()  # type: ignore
                losses[mask_better_attack] = loss[mask_better_attack].detach()  # type: ignore

        # Caching results

        if self._cached_x is None:
            self._cached_x = x
        else:
            self._cached_x = torch.vstack([self._cached_x, x])

        if self._cached_xs_tilde is None:
            self._cached_xs_tilde = xs_tilde
        else:
            self._cached_xs_tilde = torch.vstack([self._cached_xs_tilde, xs_tilde])  # type: ignore

        if self._cached_losses is None:
            self._cached_losses = losses
        else:
            self._cached_losses = torch.vstack([self._cached_losses, losses])  # type: ignore

        return losses  # type: ignore

    def generate_adversarial_examples(
        self, x: Tensor, target: Optional[Any] = None, num_examples: int = 100
    ) -> Tuple[Tensor, Tensor]:
        """Function returns best adversarial examples found.

        Args:
            x (Tensor): Datapoints to evaluate
            target (Optional[Any], optional): Targets. Defaults to None.
            num_examples (int, optional): Number of adversarial examples to return. Defaults to 100.

        Returns:
            Tuple[Tensor, Tensor]: Index of x on which example was found as well as the adversarial example.
        """
        if (
            self._cached_x is None
            or not torch.isclose(self._cached_x.to(x.device), x).all()
        ):
            self.eval(x, target)

        if self.descending:
            order_val = not self.targeted
        else:
            order_val = self.targeted

        _, idx = torch.sort(self._cached_losses.flatten(), descending=order_val)  # type: ignore

        mask = torch.isfinite(self._cached_losses.flatten()[idx])
        idx = idx[mask]

        return idx[:num_examples], self._cached_xs_tilde[idx][:num_examples]  # type: ignore

    def _ensure_loss_constraint(self, loss):
        # Enforces some constriants i.e. to avoid numerical overflows.
        return loss

    @abstractmethod
    def _get_goal(self, x, target):
        # Return the target/goal of the attack
        pass


class PredictiveMetric(Metric):

    requires_thetas = False
    can_eval_single = True
    requires_potential = False
    can_eval_x_xtilde = True

    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        simulator: Simulator,
        n_samples: int = 100,
        reduction: Optional[str] = "mean",
        batch_size: Optional[int] = None,
        descending: bool = False,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """Base class for predictive metrics

        Args:
            model (Union[ParametricProbabilisticModel, PyroFlowModel]): Model
            simulator (Simulator): Simulator
            n_samples (int, optional): Samples. Defaults to 100.
            reduction (Optional[str], optional): Reduction. Defaults to "mean".
            batch_size (Optional[int], optional): Batch size. Defaults to None.
            descending (bool, optional): If lower is better. Defaults to False.
            device (str, optional): Device. Defaults to "cpu".
        """
        super().__init__(
            model,
            reduction=reduction,
            batch_size=batch_size,
            descending=descending,
            device=device,
        )
        self.simulator = simulator
        self.n_samples = n_samples

    def get_predictives(self, x_o):
        q = self.model(x_o)
        thetas = q.sample((self.n_samples,))
        x_pred = self.simulator(thetas)
        return x_pred

    def _eval(self, x, target=None):
        with torch.no_grad():
            x_pred = self.get_predictives(x)
            if target is None:
                m = self._compare_predictes_xos(x, x_pred)
            else:
                m = self._compare_predictes_xos(target, x_pred)

            return m

    @abstractmethod
    def _compare_predictes_xos(self, x_os, x_predictives):
        pass


class ApproximationMetric(Metric):
    requires_posterior = False
    requires_thetas = False
    requires_potential = False
    can_eval_single = True
    can_eval_x_xtilde = False

    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        prior: Optional[Distribution] = None,
        simulator: Optional[Simulator] = None,
        ground_truth: Optional[Any] = None,
        potential_fn: Optional[Callable] = None,
        reduction: Optional[str] = "mean",
        batch_size: Optional[int] = None,
        descending: bool = False,
        device: str = "cpu",
    ) -> None:
        """Base class for approximaiton metrics.

        Args:
            model (Union[ParametricProbabilisticModel, PyroFlowModel]): Model
            prior (Optional[Distribution], optional): Prior if needed. Defaults to None.
            simulator (Optional[Simulator], optional): Simulator if needed. Defaults to None.
            ground_truth (Optional[Any], optional): Ground truth if needed. Defaults to None.
            potential_fn (Optional[Callable], optional): Potential_fn if needed. Defaults to None.
            reduction (Optional[str], optional): Reduction. Defaults to "mean".
            batch_size (Optional[int], optional): Batch size. Defaults to None.
            descending (bool, optional): If better values are better. Defaults to False.
            device (str, optional): Device. Defaults to "cpu".
        """
        super().__init__(
            model,
            reduction=reduction,
            batch_size=batch_size,
            descending=descending,
            device=device,
        )
        self.prior = prior
        self.simulator = simulator
        self.ground_truth = ground_truth
        self.potential_fn = potential_fn
