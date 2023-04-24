
import torch
from torch import Tensor

from rbi.models.base import Generator  # type: ignore

from torch.distributions import (
    Bernoulli,
    Categorical,
    Independent,
    Normal,
    MultivariateNormal,
    Distribution
)

from rbi.utils.distributions import MixtureOfDiagNormals


class BernoulliGenerator(Generator):
    """Generates a Bernoulli distribution"""

    def generate_distribution(self, phi: Tensor) -> Bernoulli:
        """Probability parametrization of Bernoulli distribution.

        Args:
            phi (Tensor): Probability of success

        Returns:
            Distribution: Bernoulli
        """
        return Bernoulli(phi, validate_args=False)


class CategoricalGenerator(Generator):
    """Generates a categorical distribution"""

    def generate_distribution(self, phi: Tensor) -> Categorical:
        """Probability parametrization of Categorical.

        Args:
            phi (Tensor): Class probabilities

        Returns:
            Categorical: Categorical distirbution.
        """
        return Categorical(phi, validate_args=False)


class DiagonalGaussianGenerator(Generator):
    """Generates a multivariate Gaussian with diagonal covariance matrix."""

    def __init__(self, d: int, min_scale: float = 1e-8):
        """Given a vector of d/2 real and d/2 positive values, it returns a Gaussian.

        Args:
            d (int): Event dimension
            min_scale (float, optional): Minimum scale for numerical stability. Defaults to 1e-8.
        """
        super().__init__(d)
        self.min_scale = min_scale

    def generate_distribution(self, phi: Tensor) -> Independent:
        mean = phi[..., : self.d]
        scales = phi[..., self.d :] + self.min_scale
        return Independent(Normal(mean, scales, validate_args=False), 1)


class MultivariateNormalGenerator(Generator):
    """Generates a multivariate Gaussian with general covariance matrix."""

    def __init__(self, d: int, min_scale: float = 1e-4) -> None:
        """Given a vector of real values outputs a Multivariate normal with psd. covariance matrix.

        Args:
            d (int): Event dimension.
            min_scale (float, optional): Min diagonal variance for numerical stability. Defaults to 1e-5.
        """
        super().__init__(d)
        self.min_scale = min_scale

    def generate_distribution(self, phi: Tensor) -> MultivariateNormal:
        device = phi.device
        mean = phi[..., : self.d]
        std = phi[..., self.d :]
        L_diag = std[..., : self.d]
        L_off_diag = (
            std[..., self.d :] * 0.01
        )  # At start it is more stable to have low correlation...
        L = torch.zeros(*std.shape[:-1], self.d, self.d, device=device)
        tril_indices = torch.tril_indices(row=self.d, col=self.d, offset=-1)
        L[..., tril_indices[0], tril_indices[1]] = L_off_diag
        L = L + torch.diag_embed(2*torch.nn.functional.softplus(L_diag))
        precission = L @ torch.transpose(L, dim0=-2, dim1=-1) + (
            torch.eye(self.d, device=device) * self.min_scale
        ).unsqueeze(0)
        return MultivariateNormal(mean, precision_matrix=precission, validate_args=False)


class MixtureDiagNormalGenerator(Generator):
    """Generates a mixture of diagonal normal generators."""

    def __init__(self, d: int, K: int, min_scale: float = 1e-8) -> None:
        """Given a vector of real values it outputs a parmeterized MixtureOfDiagNormals

        Args:
            d (int): Event dimension
            K (int): Number of components.
            min_scale (float, optional): Min scale for stability. Defaults to 1e-8.
        """
        super().__init__(d)
        self.K = K
        self.min_scale = min_scale

    def generate_distribution(self, phi: Tensor) -> MixtureOfDiagNormals:
        batch_dim = phi.shape[:-1]
        logits = torch.log_softmax(phi[..., : self.K], -1).reshape(*batch_dim, self.K)
        means = phi[..., self.K : self.K + self.K * self.d].reshape(
            *batch_dim, self.K, self.d
        )
        scales = (
            torch.nn.functional.softplus(phi[..., self.K + self.K * self.d :]).reshape(
                *batch_dim, self.K, self.d
            )
            + self.min_scale
        )
        return MixtureOfDiagNormals(means, scales, logits)
