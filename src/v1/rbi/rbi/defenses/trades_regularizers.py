
from grpc import Call
from rbi.defenses.base import AdditiveRegularizer

from rbi.attacks import L2PGDAttack, L1PGDAttack, LinfPGDAttack  # type: ignore
from rbi.attacks import (
    L1UniformNoiseAttack,
    L2UniformNoiseAttack,
    LinfUniformNoiseAttack,
    GaussianNoiseAttack,
)

from rbi.loss import ReverseKLLoss, ForwardKLLoss
from rbi.loss.base import EvalLoss, TrainLoss
from rbi.attacks.base import Attack

import torch
from torch.nn import Module

from typing import Callable, Optional

from torch import Tensor



class TradesAdversarialRegularizer(AdditiveRegularizer):
    def __init__(
        self,
        model: Module,
        loss_fn: TrainLoss,
        attack: Attack,
        reg_loss_fn: Callable,
        beta: float=1.0,
        reduce: str="mean",
        **kwargs
    ):
        """This is a TRADES like regularizer. It adds a regularizer of the form

            beta * max_x' D(q(theta|x) || q'(theta|x')) s.t. ||x'|| < eps

        Args:
            model (Module): Model
            loss_fn (TrainLoss): Loss
            attack (Attack): Attack
            reg_loss_fn (Callable, optional): Loss fn of the regularizer. 
            beta (float, optional): Regularization strenght. Defaults to 1.0.
            reduce (str, optional): Reduction of regularizer over xs. Defaults to "mean".
        """
        super().__init__(model, loss_fn, reduce)
        self.attack = attack
        self.reg_loss_fn = reg_loss_fn
        self.beta = beta
        self._regularizer = self._regularizer_attack

    def _set_algorithm(self, method:str):
        self._regularizer = self._regularizer_attack

    def _regularizer_attack(self, input: Tensor, output: Tensor, target: Tensor):
    
        if self.beta == 0.:
            return torch.zeros(1)
        else:
            x_pert = self.attack.perturb(input)
            return self.beta * self.reg_loss_fn(self.model(input), self.model(x_pert))


class L2PGDTrades(TradesAdversarialRegularizer):
    def __init__(
        self,
        model: Module,
        loss_fn: TrainLoss, 
        attack_loss_fn: EvalLoss = ForwardKLLoss(mc_samples=1),
        reg_loss_fn: Optional[EvalLoss] = None,
        eps: float=0.5,
        beta: float=1.,
        reduce: str="mean",
        **kwargs
    ):
        """Trades with L2PGD attack approximating the maximization.

             beta * max_x' D(q(theta|x) || q'(theta|x')) s.t. ||x'|| < eps

        Args:
            model (Module): Model
            loss_fn (TrainLoss): Train loss
            attack_loss_fn (EvalLoss, optional): Loss used by the attack. Defaults to ReverseKLLoss().
            reg_loss_fn (Optional[EvalLoss], optional): Loss used in evaluating the regularizer. Defaults to None.
            eps (float, optional): Tolerance of the attack. Defaults to 0.5.
            beta (float, optional): Regularization constant. Defaults to 1..
            reduce (str, optional): Reduction of the regularizer. Defaults to "mean".
        """
        if reg_loss_fn is None:
            if hasattr(attack_loss_fn, "mc_samples"):
                mc_samples = int(attack_loss_fn.mc_samples)  # type: ignore
                reg_loss_fn = attack_loss_fn.__class__(mc_samples=mc_samples)  # type: ignore
            else:
                reg_loss_fn = attack_loss_fn.__class__()
        attack = L2PGDAttack(model, attack_loss_fn, eps=eps, **kwargs)
        super().__init__(model, loss_fn, attack, reg_loss_fn, beta, reduce, **kwargs)

class L2PGDrKLTrades(TradesAdversarialRegularizer):
    def __init__(
        self,
        model: Module,
        loss_fn: TrainLoss, 
        attack_loss_fn: EvalLoss = ReverseKLLoss(mc_samples=1),
        reg_loss_fn: Optional[EvalLoss] = None,
        eps: float=0.5,
        beta: float=1.,
        reduce: str="mean",
        **kwargs
    ):
        """Trades with L2PGD attack approximating the maximization.

             beta * max_x' D(q(theta|x) || q'(theta|x')) s.t. ||x'|| < eps

        Args:
            model (Module): Model
            loss_fn (TrainLoss): Train loss
            attack_loss_fn (EvalLoss, optional): Loss used by the attack. Defaults to ReverseKLLoss().
            reg_loss_fn (Optional[EvalLoss], optional): Loss used in evaluating the regularizer. Defaults to None.
            eps (float, optional): Tolerance of the attack. Defaults to 0.5.
            beta (float, optional): Regularization constant. Defaults to 1..
            reduce (str, optional): Reduction of the regularizer. Defaults to "mean".
        """
        if reg_loss_fn is None:
            if hasattr(attack_loss_fn, "mc_samples"):
                mc_samples = int(attack_loss_fn.mc_samples)  # type: ignore
                reg_loss_fn = attack_loss_fn.__class__(mc_samples=mc_samples)  # type: ignore
            else:
                reg_loss_fn = attack_loss_fn.__class__()
        attack = L2PGDAttack(model, attack_loss_fn, eps=eps, **kwargs)
        super().__init__(model, loss_fn, attack, reg_loss_fn, beta, reduce, **kwargs)


class L1PGDTrades(TradesAdversarialRegularizer):
    def __init__(
        self,
        model: Module,
        loss_fn: TrainLoss, 
        attack_loss_fn: EvalLoss = ForwardKLLoss(mc_samples=1),
        reg_loss_fn: Optional[EvalLoss] = None,
        eps: float=0.5,
        beta: float=1.,
        reduce: str="mean",
        **kwargs
    ):
        """Trades with L1PGD attack approximating the maximization.

             beta * max_x' D(q(theta|x) || q'(theta|x')) s.t. ||x'|| < eps

        Args:
            model (Module): Model
            loss_fn (TrainLoss): Train loss
            attack_loss_fn (EvalLoss, optional): Loss used by the attack. Defaults to ReverseKLLoss().
            reg_loss_fn (Optional[EvalLoss], optional): Loss used in evaluating the regularizer. Defaults to None.
            eps (float, optional): Tolerance of the attack. Defaults to 0.5.
            beta (float, optional): Regularization constant. Defaults to 1..
            reduce (str, optional): Reduction of the regularizer. Defaults to "mean".
        """
        if reg_loss_fn is None:
            if hasattr(attack_loss_fn, "mc_samples"):
                mc_samples = int(attack_loss_fn.mc_samples)  # type: ignore
                reg_loss_fn = attack_loss_fn.__class__(mc_samples=mc_samples)  # type: ignore
            else:
                reg_loss_fn = attack_loss_fn.__class__()
        attack = L1PGDAttack(model, attack_loss_fn, eps=eps, **kwargs)
        super().__init__(model, loss_fn, attack, reg_loss_fn, beta, reduce, **kwargs)


class LinfPGDTrades(TradesAdversarialRegularizer):
    def __init__(
        self,
        model: Module,
        loss_fn: TrainLoss, 
        attack_loss_fn: EvalLoss = ForwardKLLoss(mc_samples=1),
        reg_loss_fn: Optional[EvalLoss] = None,
         eps: float=0.5,
        beta: float=1.,
        reduce: str="mean",
        **kwargs
    ):
        """Trades with LinfPGD attack approximating the maximization.

             beta * max_x' D(q(theta|x) || q'(theta|x')) s.t. ||x'|| < eps

        Args:
            model (Module): Model
            loss_fn (TrainLoss): Train loss
            attack_loss_fn (EvalLoss, optional): Loss used by the attack. Defaults to ReverseKLLoss().
            reg_loss_fn (Optional[EvalLoss], optional): Loss used in evaluating the regularizer. Defaults to None.
            eps (float, optional): Tolerance of the attack. Defaults to 0.5.
            beta (float, optional): Regularization constant. Defaults to 1..
            reduce (str, optional): Reduction of the regularizer. Defaults to "mean".
        """
        if reg_loss_fn is None:
            if hasattr(attack_loss_fn, "mc_samples"):
                mc_samples = int(attack_loss_fn.mc_samples)  # type: ignore
                reg_loss_fn = attack_loss_fn.__class__(mc_samples=mc_samples)  # type: ignore
            else:
                reg_loss_fn = attack_loss_fn.__class__()
        attack = LinfPGDAttack(model, attack_loss_fn, eps=eps, **kwargs)
        super().__init__(model, loss_fn, attack, reg_loss_fn, beta, reduce, **kwargs)


class RandomTrades(TradesAdversarialRegularizer):
    def __init__(
        self,
        model: Module,
        loss_fn: TrainLoss,
        attack: Attack,
        reg_loss_fn: EvalLoss,
        beta: float=1.,
        mc_samples: int=1,
        reduce: str="mean",
    ):
        """Trades like regularizer.

             beta * E_d [D(q(theta|x) || q'(theta|x + d))] s.t. ||d|| < eps

        Args:
            model (Module): Model
            loss_fn (TrainLoss): Train loss
            reg_loss_fn (Optional[EvalLoss], optional): Loss used in evaluating the regularizer. Defaults to None.
            attack (Attack): Noise base attack
            beta (float, optional): Regularization constant. Defaults to 1.
            mc_samples (int, optional): Number of samples to approximate the expectation.
            reduce (str, optional): Reduction of the regularizer. Defaults to "mean".
        """
        super().__init__(model, loss_fn, attack, reg_loss_fn, beta, reduce)
        self.mc_samples_reg = mc_samples

    def _regularizer_attack(self, input: Tensor, output: Tensor, target: Tensor) -> Tensor:
        input = input.repeat(self.mc_samples_reg, 1)
        x_pert = self.attack.perturb(input)
        return self.beta * self.reg_loss_fn(self.model(input), self.model(x_pert))


class L2NoiseTrades(RandomTrades):
    def __init__(
        self,
        model: Module,
        loss_fn: TrainLoss,
        eps:float=0.5,
        reg_loss_fn: Optional[EvalLoss]=None,
        beta:float=1,
        mc_samples=1,
        reduce="mean",
        **kwargs
    ):
        """Trades like regularizer.

             beta * E_d [D(q(theta|x) || q'(theta|x + d))] s.t. ||d|| < eps

        Args:
            model (Module): Model
            loss_fn (TrainLoss): Train loss
            eps (float, optional): Tolerance of noise.
            reg_loss_fn (Optional[EvalLoss], optional): Loss used in evaluating the regularizer. Defaults to None.
            beta (float, optional): Regularization constant. Defaults to 1.
            mc_samples (int, optional): Number of samples to approximate the expectation.
            reduce (str, optional): Reduction of the regularizer. Defaults to "mean".
        """

        attack = L2UniformNoiseAttack(model, eps=eps)
        if reg_loss_fn is None:
            reg_loss_fn = ForwardKLLoss(mc_samples=3*mc_samples)
        super().__init__(
            model,
            loss_fn,
            attack,
            reg_loss_fn,
            beta,
            mc_samples,
            reduce,
            **kwargs
        )


class L1NoiseTrades(RandomTrades):
    def __init__(
        self,
        model: Module,
        loss_fn: TrainLoss,
        eps:float=0.5,
        reg_loss_fn: Optional[EvalLoss]=None,
        beta:float=1,
        mc_samples=1,
        reduce="mean",
        **kwargs
    ):
        """Trades like regularizer.

             beta * E_d [D(q(theta|x) || q'(theta|x + d))] s.t. ||d|| < eps

        Args:
            model (Module): Model
            loss_fn (TrainLoss): Train loss
            eps (float, optional): Tolerance of noise.
            reg_loss_fn (Optional[EvalLoss], optional): Loss used in evaluating the regularizer. Defaults to None.
            beta (float, optional): Regularization constant. Defaults to 1.
            mc_samples (int, optional): Number of samples to approximate the expectation.
            reduce (str, optional): Reduction of the regularizer. Defaults to "mean".
        """

        attack = L1UniformNoiseAttack(model, eps=eps)
        if reg_loss_fn is None:
            reg_loss_fn = ForwardKLLoss(mc_samples=3*mc_samples)
        super().__init__(
            model,
            loss_fn,
            attack,
            reg_loss_fn,
            beta,
            mc_samples,
            reduce,
            **kwargs
        )

class LinfNoiseTrades(RandomTrades):
    def __init__(
        self,
        model: Module,
        loss_fn: TrainLoss,
        eps:float=0.5,
        reg_loss_fn: Optional[EvalLoss]=None,
        beta:float=1,
        mc_samples=1,
        reduce="mean",
        **kwargs
    ):
        """Trades like regularizer.

             beta * E_d [D(q(theta|x) || q'(theta|x + d))] s.t. ||d|| < eps

        Args:
            model (Module): Model
            loss_fn (TrainLoss): Train loss
            eps (float, optional): Tolerance of noise.
            reg_loss_fn (Optional[EvalLoss], optional): Loss used in evaluating the regularizer. Defaults to None.
            beta (float, optional): Regularization constant. Defaults to 1.
            mc_samples (int, optional): Number of samples to approximate the expectation.
            reduce (str, optional): Reduction of the regularizer. Defaults to "mean".
        """

        attack = LinfUniformNoiseAttack(model, eps=eps)
        if reg_loss_fn is None:
            reg_loss_fn = ForwardKLLoss(mc_samples=3*mc_samples)
        super().__init__(
            model,
            loss_fn,
            attack,
            reg_loss_fn,
            beta,
            mc_samples,
            reduce,
            **kwargs
        )


class GaussianNoiseTrades(RandomTrades):
    def __init__(
        self,
        model: Module,
        loss_fn: TrainLoss,
        eps:float=0.5,
        reg_loss_fn: Optional[EvalLoss]=None,
        beta:float=1,
        mc_samples=1,
        reduce="mean",
        **kwargs
    ):
        """Trades like regularizer.

             beta * E_d [D(q(theta|x) || q'(theta|x + d))] s.t. ||d|| < eps

        Args:
            model (Module): Model
            loss_fn (TrainLoss): Train loss
            eps (float, optional): Tolerance of noise.
            reg_loss_fn (Optional[EvalLoss], optional): Loss used in evaluating the regularizer. Defaults to None.
            beta (float, optional): Regularization constant. Defaults to 1.
            mc_samples (int, optional): Number of samples to approximate the expectation.
            reduce (str, optional): Reduction of the regularizer. Defaults to "mean".
        """

        attack = GaussianNoiseAttack(model, eps=eps)
        if reg_loss_fn is None:
            reg_loss_fn = ForwardKLLoss(mc_samples=3*mc_samples)
        super().__init__(
            model,
            loss_fn,
            attack,
            reg_loss_fn,
            beta,
            mc_samples,
            reduce,
            **kwargs
        )
