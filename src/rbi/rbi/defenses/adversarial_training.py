

from torch import Tensor
from torch.nn import Module

from typing import Callable, Any, Tuple

from rbi.loss import ReverseKLLoss,ForwardKLLoss, LogLikelihoodLoss
from rbi.attacks.base import Attack
from rbi.attacks import L2PGDAttack, L1PGDAttack, LinfPGDAttack  # type: ignore
from rbi.attacks import (
    L1UniformNoiseAttack,
    L2UniformNoiseAttack,
    LinfUniformNoiseAttack,
)


from rbi.defenses.base import DataAugmentationRegularizer

from rbi.loss.base import TrainLoss


class AdversarialTraining(DataAugmentationRegularizer):

    def __init__(self, model: Module, loss_fn: TrainLoss, attack: Attack, **kwargs):
        """This implements general adversarial training. The input x is modified by an adversarial attack. The loss is then evaluated on the attacked x.

        Args:
            model (Module): The model to apply this defense.
            loss_fn (TrainLoss): The loss_fn.
            attack (Attack): The adversarial attack used.
        """
        super().__init__(model, loss_fn, **kwargs)
        self.attack = attack

    def _transform(self, input: Tensor, target: Any) -> Tuple[Tensor, Any]:
        """Regularizer just changes x by attacking it...

        Args:
            input (Tensor): Input
            target (Any): Target

        Returns:
            Tuple[Tensor, Any]: Modified input, same target.
        """
        if not self.attack.targeted:
            x_tilde = self.attack.perturb(input)
        else:
            x_tilde = self.attack.perturb(input, target)

        return x_tilde.detach(), target


class L2PGDAdversarialTraining(AdversarialTraining):
    def __init__(
        self, model: Module, loss_fn: TrainLoss, attack_loss_fn: Callable=ForwardKLLoss(mc_samples=1), eps:float=0.5, **kwargs
    ):
        """This implement adversarial training using the L2PGD Attack.

        Args:
            model (Module): Model
            loss_fn (TrainLoss): Loss
            attack_loss_fn (Callable, optional): Loss used by attack. Defaults to ReverseKLLoss().
            eps (float, optional): Tolerance of attack. Defaults to 0.5.
            kwargs: Additional parameters of the attack.
        """
        attack = L2PGDAttack(model, attack_loss_fn, eps=eps, **kwargs)
        super().__init__(model, loss_fn, attack)

class L2PGDTargetedAdversarialTraining(AdversarialTraining):
    def __init__(
        self, model: Module, loss_fn: TrainLoss, attack_loss_fn: Callable=LogLikelihoodLoss(), eps:float=0.5, **kwargs
    ):
        """This implement adversarial training using the L2PGD Attack.

        Args:
            model (Module): Model
            loss_fn (TrainLoss): Loss
            attack_loss_fn (Callable, optional): Loss used by attack. Defaults to ReverseKLLoss().
            eps (float, optional): Tolerance of attack. Defaults to 0.5.
            kwargs: Additional parameters of the attack.
        """
        attack = L2PGDAttack(model, attack_loss_fn, eps=eps, targeted=True, **kwargs)
        super().__init__(model, loss_fn, attack)


class L1PGDAdversarialTraining(AdversarialTraining):
    def __init__(
        self, model: Module, loss_fn: TrainLoss, attack_loss_fn: Callable=ForwardKLLoss(mc_samples=1), eps:float=0.5, **kwargs
    ):
        """This implement adversarial training using the L1PGD Attack.

        Args:
            model (Module): Model
            loss_fn (TrainLoss): Loss
            attack_loss_fn (Callable, optional): Loss used by attack. Defaults to ReverseKLLoss().
            eps (float, optional): Tolerance of attack. Defaults to 0.5.
            kwargs: Additional parameters of the attack.
        """
        attack = L1PGDAttack(model, attack_loss_fn, eps=eps, **kwargs)
        super().__init__(model, loss_fn, attack)


class LinfPGDAdversarialTraining(AdversarialTraining):
    def __init__(
        self, model: Module, loss_fn: TrainLoss, attack_loss_fn: Callable=ForwardKLLoss(mc_samples=1), eps:float=0.5, **kwargs
    ):
        """This implement adversarial training using the LinfPGD Attack.

        Args:
            model (Module): Model
            loss_fn (TrainLoss): Loss
            attack_loss_fn (Callable, optional): Loss used by attack. Defaults to ReverseKLLoss().
            eps (float, optional): Tolerance of attack. Defaults to 0.5.
            kwargs: Additional parameters of the attack.
        """
        attack = LinfPGDAttack(model, attack_loss_fn, eps=eps, **kwargs)
        super().__init__(model, loss_fn, attack)


class L2UniformNoiseTraining(AdversarialTraining):
    def __init__(self, model: Module, loss_fn: TrainLoss, eps:float=0.5, **kwargs):
        """This implement noise augmentation with uniform noise distributed uniformly in the L2 ball.

        Args:
            model (Module): Model
            loss_fn (TrainLoss): Loss
            attack_loss_fn (Callable, optional): Loss used by attack. Defaults to ReverseKLLoss().
            eps (float, optional): Tolerance of attack. Defaults to 0.5.
            kwargs: Additional parameters of the attack.
        """
        attack = L2UniformNoiseAttack(model, eps=eps, **kwargs)
        super().__init__(model, loss_fn, attack)


class L1UniformNoiseTraining(AdversarialTraining):
    def __init__(self, model: Module, loss_fn: TrainLoss, eps: float=0.5, **kwargs):
        """This implement noise augmentation with uniform noise distributed uniformly in the L1 ball.

        Args:
            model (Module): Model
            loss_fn (TrainLoss): Loss
            attack_loss_fn (Callable, optional): Loss used by attack. Defaults to ReverseKLLoss().
            eps (float, optional): Tolerance of attack. Defaults to 0.5.
            kwargs: Additional parameters of the attack.
        """
        attack = L1UniformNoiseAttack(model, eps=eps, **kwargs)
        super().__init__(model, loss_fn, attack)


class LinfUniformNoiseTraining(AdversarialTraining):
    def __init__(self, model: Module, loss_fn: TrainLoss, eps: float=0.5, **kwargs):
        """This implement noise augmentation with uniform noise distributed uniformly in the L2 ball.

        Args:
            model (Module): Model
            loss_fn (TrainLoss): Loss
            attack_loss_fn (Callable, optional): Loss used by attack. Defaults to ReverseKLLoss().
            eps (float, optional): Tolerance of attack. Defaults to 0.5.
            kwargs: Additional parameters of the attack.
        """
        attack = LinfUniformNoiseAttack(model, eps=eps, **kwargs)
        super().__init__(model, loss_fn, attack)
