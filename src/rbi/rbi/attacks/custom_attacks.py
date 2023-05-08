
from numpy import isin
import torch
from torch import Tensor
from torch.distributions import Distribution

from typing import Callable, Any, Optional, Tuple

from rbi.utils.transforms import sampling_transform_jacobian
from rbi.utils.fisher_info import fisher_information_matrix

from rbi.utils.distributions import sample_lp_uniformly



from rbi.attacks.base import Attack


class RandomSearchAttack(Attack):
    """A simple gradient-free attack. Which chooses the best out of n random samples within the unit hypersphere centered at x."""

    def __init__(
        self,
        predict: Callable,
        loss_fn: Callable,
        eps: float = 0.3,
        targeted: bool = False,
        ord: float = 2.0,
        mc_samples: int = 100,
        **kwargs
    ) -> None:
        """This is a gradient-free attack, which chooses the best out of n samples randomly distributied within a hypershphere of radius 'eps' centered at x.

        Args:
            predict (Callable): Predicition function
            loss_fn (Callable): Loss compatible with output of prediction
            eps (float, optional): Tolerance, typically ||x'-x|| < eps. Defaults to 0.3.
            ord (float, optional): Order of the norm . Defaults to 2.0.
            mc_samples (int, optional): Number of samples to use in the random search. Defaults to 1000.
        """

        super().__init__(predict, loss_fn, targeted=targeted, **kwargs)
        self.loss_fn = loss_fn
        self.loss_fn.reduction = None
        self.eps = eps
        self.ord = ord
        self.mc_samples = mc_samples

    def perturb(self, x, target=None):
        with torch.no_grad():
            if self.targeted:
                if target is None:
                    raise ValueError(
                        "You set the attack to be targeted, but did not pass a target."
                    )

            old_shape = x.shape 
            x = x.reshape(-1, x.shape[-1])

            samples = sample_lp_uniformly(
                self.mc_samples, d=x.shape[-1], p=self.ord, eps=self.eps, device=x.device
            )
            x_proposed = x.unsqueeze(1) + samples.unsqueeze(0)
            x_proposed = x_proposed.transpose(0,1)

            if target is None:
                target = self.predict(x)
                target = target.expand((self.mc_samples,) + target.batch_shape)
            else:
                if isinstance(target, torch.Tensor):
                    target = target.reshape(-1, target.shape[-1])
                    target = target.repeat(self.mc_samples, 1,1)
                else:
                    target = target.expand((self.mc_samples,) + target.batch_shape)

            l = self.loss_fn(self.predict(x_proposed), target)

            if target is None:
                selected = l.argmax(0)
            else:
                selected = l.argmin(0)
            return torch.vstack(
                [x_proposed[ selected[i],i] for i in range(len(selected))]
            ).clip(self.clip_min, self.clip_max).reshape(old_shape)


class IterativeRandomSearchAttack(Attack):
    """A simple gradient-free iterative random search attack."""

    def __init__(
        self,
        predict: Callable,
        loss_fn: Callable,
        eps: float = 0.3,
        targeted: bool = False,
        eps_iter: float = 0.1,
        nb_iter: int = 40,
        ord: float = 2.0,
        contraction_factor: float = 1.3,
        precision_factor: float = 0.05,
        mc_samples: int = 10,
        **kwargs
    ):
        """A simple gradient-free iterative random search attack.

        Args:
            predict (Callable): Prediction function
            loss_fn (Callable): Loss function
            eps (float, optional): Tolerance level. Defaults to 0.3
            eps_iter (float, optional): Tolerance level per iteration i.e. step size. Defaults to 0.5.
            nb_iter (int, optional): Number of iterations . Defaults to 40.
            ord (float, optional): Norm order. Defaults to 2.0.
            contraction_factor (float, optional): If a better solution is found step size shrinks by this factor . Defaults to 1.3.
            precision_factor (float, optional): If the step size is smaller than this we reset it to max value. Defaults to 0.05.
            mc_samples (int, optional): Number of samples per iteration. Defaults to 100.
        """
        super().__init__(predict, loss_fn, targeted=targeted)
        self.loss_fn = loss_fn
        self.eps = eps
        self.max_eps_iter = min(eps_iter, eps)
        self.loss_fn.reduction = None
        self.nb_iter = nb_iter
        self.contraction_factor = contraction_factor
        self.precision_factor = precision_factor
        self.ord = ord
        self.mc_samples = mc_samples

    def project_lp(self, x: Tensor) -> Tensor:
        """Projects the Tensor into the unit-sphere with radius eps.

        Args:
            x (Tensor): Tensor which may lies outside the sphere.

        Returns:
            Tensor: Tensor guarantee to lie inside the sphere.
        """
        x = x.clone()
        norm = torch.linalg.norm(x, p=self.ord, dim=-1, keepdim=True)
        mask = norm > self.eps
        x[mask] = x[mask] / norm[mask] * self.eps
        return x

    def propse_new_x(
        self, x_old: Tensor, x: Tensor, y: Optional[Tensor], eps_iter: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Proposes an new perturbation

        Args:
            x_old (Tensor): x of previous iteration.
            x (Tensor): Original x.
            y (Tensor): Target.
            eps_iter (float): Similar to a step size.

        Returns:
            Tensor: Selected x as well as loss values
        """
        batch_n = x_old.shape[0]
        d = x_old.shape[-1]

        deltas = sample_lp_uniformly(
            self.mc_samples, d, p=self.ord, eps=1.0, device=x.device  # type: ignore
        ).unsqueeze(0)
        proposed_deltas = deltas * eps_iter.reshape(batch_n, 1, 1)  # type: ignore
        x_proposed = x_old.clone().unsqueeze(1) + proposed_deltas
        x_proposed = x_proposed.transpose(0,1)

        p = torch.inf if isinstance(self.ord, str) else self.ord
        eps_constraint = (
            torch.linalg.norm(x_proposed - x.unsqueeze(0), dim=-1, ord=p) <= self.eps
        )
        l = self.loss_fn(self.predict(x_proposed), y)
        if not self.targeted:
            l[~eps_constraint] = -torch.inf
        else:
            l[~eps_constraint] = torch.inf
        val, ind = l.max(0)

        x_selected = []
        for i in range(x_old.shape[0]):
            if val[i] != -torch.inf:
                x_selected.append(x_proposed[ind[i],i])
            else:
                x_selected.append(x_old[i])

        x_selected = torch.vstack(x_selected)

        return x_selected, val

    def updated_eps_iter(
        self, x_old: Tensor, x_new: Tensor, eps_iter: Tensor
    ) -> Tensor:
        """Updates the 'step-size'.

        Args:
            x_old (Tensor): X of previous iteration
            x_new (Tensor): X of current iteration
            eps_iter (float): Current step size.

        Returns:
            Tensor: Updates step size
        """
        mask = (x_old == x_new).all(-1)
        eps_iter[mask] = eps_iter[mask] / self.contraction_factor
        eps_iter[~mask] = self.max_eps_iter

        mask2 = eps_iter < self.precision_factor
        eps_iter[mask2] = self.max_eps_iter

        return eps_iter

    def perturb(self, x: Tensor, target: Optional[Tensor] = None) -> Tensor:
        with torch.no_grad():
            if self.targeted:
                if target is None:
                    raise ValueError(
                        "You set the attack to be targeted, but did not pass a target."
                    )

            old_shape = x.shape 
            x = x.reshape(-1, x.shape[-1])
            

            if target is None:
                target = self.predict(x)
                target_expanded = target.expand((self.mc_samples,) + target.batch_shape)
            else:
                if isinstance(target, torch.Tensor):
                    target = target.reshape(-1, target.shape[-1])
                    target_expanded = target.repeat(1,self.mc_samples,1)
                else:
                    target_expanded = target.expand((self.mc_samples,) + target.batch_shape)

            eps_iter = torch.ones(x.shape[0], device=x.device) * self.max_eps_iter

            old_x = x.clone()
            for i in range(self.nb_iter):
                loss_old = self.loss_fn(self.predict(old_x), target)
                new_x, loss_new = self.propse_new_x(old_x, x, target_expanded, eps_iter)
                if not self.targeted:
                    mask = loss_new > loss_old
                else:
                    mask = loss_new < loss_old
                old_x[mask] = new_x[mask]
                eps_iter = self.updated_eps_iter(old_x, new_x, eps_iter)
            return old_x.clip(self.clip_min, self.clip_max).reshape(old_shape)


class SpectralKLAttack(Attack):
    """This attack uses the eigenvector of the Fisher information matrix as adversarial attack.
    The eigenvectors points locally into the direction of largest increase of KL divergence.
    """

    def __init__(self, predict: Callable, eps: float = 0.5, **kwargs) -> None:
        """This attack uses the eigenvector of the Fisher information matrix with the largest eigenvalues as adversarial attack.

        Args:
            predict (Callable): Prediction function of neural network. Should return a distribution.
            eps (float, optional): L2 length of the pertubation. Defaults to 0.5.
        """
        super().__init__(predict, **kwargs)
        self.eps = eps

        if self.targeted:
            raise ValueError("This attack cannot be targeted")

    def _get_kl_matrix_eigendecomposition(
        self, input: Tensor, output: Distribution
    ) -> Tuple[Tensor, Tensor]:
        """Return the eigenvalue decomposition of the Fisher information matrix.

        Args:
            input (Tensor): Inputs
            output (Distribution): Distributional prediction

        Returns:
            Tuple(Tensor, Tensor): Eigenvalues and eigenvectors of the FIM
        """
        F_x = fisher_information_matrix(self.predict, input, output)
        eigh = torch.linalg.eigh(F_x)
        return eigh.eigenvalues, eigh.eigenvectors

    def perturb(self, x, eigendirection=-1, **kwargs):

        output = self.predict(x)
        _, eigvecs = self._get_kl_matrix_eigendecomposition(x, output)
        adversarial_direction = eigvecs[..., eigendirection] * self.eps
        return (x + adversarial_direction).clip(self.clip_min, self.clip_max)


class SpectralTransformAttack(Attack):
    def __init__(self, predict, eps=0.5, *args, **kwargs):
        super().__init__(predict, *args, **kwargs)
        self.eps = eps

    def _get_transform_eigendecompositon(self, input, output):

        if output.has_rsample:
            raise ValueError("Distribution must have rsample attribute...")

        matrix = sampling_transform_jacobian(self.predict, input)
        eigh = torch.linalg.eigh(matrix)
        return eigh.eigenvalues, eigh.eigenvectors

    def perturb(self, x, y=None, eigendirection=-1):
        if y is not None:
            raise ValueError("This attack cannot be targeted")
        output = self.predict(x)
        _, eigvecs = self._get_transform_eigendecompositon(x, output)
        adversarial_direction = eigvecs[..., eigendirection] * self.eps
        return (x + adversarial_direction).clip(self.clip_min, self.clip_max)
