import math
from rbi.loss.base import EvalLoss

from typing import Callable, Iterable, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from rbi.loss.kl_div import kl_divergence, set_mc_budget

from sbi.utils.metrics import c2st


class ExpectedFeatureLoss(EvalLoss):
    def __init__(
        self,
        feature_distance: Callable,
        feature_map: Callable,
        reduction: str = "mean",
        mc_samples: int = 128,
    ) -> None:
        """This computers a specified distance between specified expected features of two distributions. Simple examples is distance between mean,variance...

        Args:
            feature_distance (Callable): Distance function between features
            feature_map (Callable): Feature map
            reduction (str, optional): _description_. Defaults to "mean".
            mc_samples (int, optional): _description_. Defaults to 128.
        """
        super().__init__(reduction)
        self.feature_map = feature_map
        self.mc_samples = mc_samples
        self.feature_loss = feature_distance

    def _loss(self, output: Distribution, target: Distribution) -> Tensor:

        if not isinstance(target, Distribution):
            raise ValueError(r"The target must be a distribution...")

        phi1 = self.feature_map(output, mc_samples=self.mc_samples)
        phi2 = self.feature_map(target, mc_samples=self.mc_samples)

        return self.feature_loss(phi1, phi2)


def compute_mean(rv, mc_samples=128, **kwargs):
    try:
        return rv.mean
    except:
        samples = rv.rsample((mc_samples,))
        return samples.mean(-1)


class MeanDifference(ExpectedFeatureLoss):
    def __init__(self, ord: float = 2.0, **kwargs):
        """Computes the distance between means of the distributions

        Args:
            ord_norm (float, optional): Which p-norm to use. Defaults to 2.0.
        """
        feature_loss = lambda x, y: torch.linalg.norm(x - y, ord=ord)
        super().__init__(feature_loss, compute_mean, **kwargs)


class NegativeLogLikelihoodLoss(EvalLoss):
    def __init__(self, reduction: str = "mean", **kwargs) -> None:
        """Compute the negative log likelihood of the target under the output distribution.

        Args:
            reduction (str, optional): Reduction strategy. Defaults to "mean".
        """
        super().__init__(reduction)

    def _loss(self, output: Distribution, target: Tensor) -> Tensor:

        if not isinstance(target, Tensor):
            raise ValueError(
                r"This loss function requires as input events $y\sim p(y|x)$, which must be given as tensors."
            )
        target = target.float()
        sample_shape = output.batch_shape + output.event_shape

        return -output.log_prob(target.reshape(*sample_shape))


class LogLikelihoodLoss(EvalLoss):
    def __init__(self, reduction: str = "mean", **kwargs) -> None:
        """Compute the log likelihood of the target under the output distribution.

        Args:
            reduction (str, optional): Reduction strategy. Defaults to "mean".
        """
        super().__init__(reduction)

    def _loss(self, output: Distribution, target: Tensor) -> Tensor:

        if not isinstance(target, Tensor):
            raise ValueError(
                r"This loss function requires as input events $y\sim p(y|x)$, which must be given as tensors."
            )
        target = target.float()
        sample_shape = output.batch_shape + output.event_shape

        return output.log_prob(target.reshape(*sample_shape))


class ForwardKLLoss(EvalLoss):
    def __init__(self, reduction: str = "mean", mc_samples: int = 10, **kwargs) -> None:
        """Compute the kl divergence D_KL(target, output).

        Args:
            reduction (str, optional): Reduction used. Defaults to "mean".
            mc_samples (int, optional): MC samples used if needed. Defaults to 10.
        """
        super().__init__(reduction)
        self.mc_samples = mc_samples

    def _loss(self, output: Distribution, target: Distribution) -> Tensor:

        if not isinstance(target, Distribution):
            raise ValueError(r"The target must be a distribution...")

        set_mc_budget(self.mc_samples)
        kl_div = kl_divergence(target, output)
        return kl_div


class ReverseKLLoss(EvalLoss):
    def __init__(self, mc_samples: int = 10, reduction: str = "mean", **kwargs) -> None:
        """Computes the kl divergence D_KL(output, target)

        Args:
            mc_samples (int, optional): Reduction used. Defaults to 10.
            reduction (str, optional): MC samples used if needed. Defaults to "mean".
        """
        super().__init__(reduction)
        self.mc_samples = mc_samples

    def _loss(self, output: Distribution, target: Distribution) -> Tensor:

        if not isinstance(target, Distribution):
            raise ValueError(r"The target must be a distribution...")

        set_mc_budget(self.mc_samples)
        kl_div = kl_divergence(output, target)
        if self.reduction == "mean":
            return torch.mean(kl_div)
        else:
            return kl_div


class SymKLLoss(EvalLoss):
    def __init__(self, mc_samples: int = 10, reduction: str = "mean", **kwargs) -> None:
        """Computes the symmetric kl divergence 0.5* [D_KL(output, target) + D_KL(target, output)]

        Args:
            mc_samples (int, optional): Reduction used. Defaults to 10.
            reduction (str, optional): MC samples used if needed. Defaults to "mean".
        """
        super().__init__(reduction)
        self.mc_samples = mc_samples

    def _loss(self, output: Distribution, target: Distribution) -> Tensor:

        if not isinstance(target, Distribution):
            raise ValueError(r"The target must be a distribution...")
        set_mc_budget(self.mc_samples)
        rkl_div = kl_divergence(output, target)
        fkl_div = kl_divergence(target, output)

        return 0.5 * rkl_div + 0.5 * fkl_div


class C2ST(EvalLoss):
    def __init__(
        self,
        mc_samples: int = 300,
        classifier: str = "rf",
        n_folds: int = 3,
        reduction: str = "mean",
        **kwargs
    ):
        super().__init__(reduction)
        self.classifier = classifier
        self.n_folds = n_folds
        self.mc_samples = mc_samples

    def _loss(self, output: Distribution, target: Distribution):

        samples_p = output.sample((self.mc_samples,)).float()  # type: ignore
        samples_q = target.sample((self.mc_samples,)).float()  # type: ignore

        samples_p = samples_p.reshape(
            self.mc_samples, -1, samples_p.shape[-1]
        ).transpose(0, 1)
        samples_q = samples_q.reshape(
            self.mc_samples, -1, samples_q.shape[-1]
        ).transpose(0, 1)

        c2st_val = []
        for x_p, x_q in zip(samples_p, samples_q):
            c2st_val.append(
                c2st(x_p, x_q, n_folds=self.n_folds, classifier=self.classifier)
            )
        c2st_val = torch.hstack(c2st_val)

        return c2st_val.nan_to_num(0.)


class C2STKnn(EvalLoss):
    def __init__(
        self,
        K: int = 10,
        ord: float = 2.0,
        mc_samples: int = 300,
        n_folds=3,
        reduction: str = "mean",
        **kwargs
    ) -> None:
        """Computes the C2ST with a KNN classifier.

        Args:
            K (int, optional): K for K-NN classifier. Defaults to 10.
            ord (float, optional): p-norm used for distances. Defaults to 2.0.
            mc_samples (int, optional): Number of samples used. Defaults to 1200.
            n_folds (int, optional): Cross-validation folds. Defaults to 3.
            reduction (str, optional): Reductions. Defaults to "mean".
        """
        super().__init__(reduction)
        self.K = K
        self.n_folds = n_folds
        self.mc_samples = mc_samples
        self.ord = ord

    def _loss(self, output: Distribution, target: Distribution):

        split_size = int(self.mc_samples / self.n_folds)
        samples_p = output.sample((self.mc_samples,)).float()  # type: ignore
        samples_q = target.sample((self.mc_samples,)).float()  # type: ignore
        samples_p = samples_p.reshape(self.mc_samples, -1, samples_p.shape[-1]).split(
            split_size
        )
        samples_q = samples_q.reshape(self.mc_samples, -1, samples_q.shape[-1]).split(
            split_size
        )

        if self.n_folds > 1:
            accs = []
            for i in range(self.n_folds):
                samples_train = torch.vstack([samples_p[i], samples_q[i]]).transpose(
                    0, 1
                )
                for j in range(self.n_folds):
                    if j != i:
                        samples_test = torch.vstack(
                            [samples_p[j], samples_q[j]]
                        ).transpose(0, 1)
                        accs.append(self.knn_predict(samples_train, samples_test))

            accs = torch.stack(accs).mean(0)
        else:
            samples = torch.vstack([samples_p[0], samples_q[0]]).transpose(0, 1)
            accs = self.knn_predict(samples, samples)

        return accs

    def knn_predict(self, x_train, x_test):
        train_len = x_train.shape[1]
        test_len = x_test.shape[1]

        distances = torch.cdist(x_test, x_train, p=self.ord)
        _, ind = torch.topk(distances, k=self.K, largest=False)
        class_probs = (ind >= (train_len // 2)).float().mean(-1)
        predictions = torch.round(class_probs)
        acc_q = predictions[:, test_len // 2 :].mean(-1)
        acc_p = 1 - predictions[:, : test_len // 2].mean(-1)

        return 0.5 * (acc_p + acc_q)


class C2STBayesOptimal(EvalLoss):
    def __init__(
        self, mc_samples: int = 300, reduction: str = "mean", **kwargs
    ) -> None:
        """Computes the C2ST accuracy for the Bayes optimal classifier under 0-1-Loss.

        Args:
            mc_samples (int, optional): Number of samples to estimate accuracy. Defaults to 2000.
            reduction (str, optional): Reduction. Defaults to "mean".
        """
        super().__init__(reduction)
        self.mc_samples = mc_samples

    def _loss(self, output: Distribution, target: Distribution):

        samples_p = output.sample((self.mc_samples,)).float()  # type: ignore
        samples_q = target.sample((self.mc_samples,)).float()  # type: ignore

        total_samples = torch.vstack([samples_p, samples_q])
        logp = output.log_prob(total_samples)
        logq = target.log_prob(total_samples)

        bayes_optimal_predictions = torch.stack([logp, logq]).argmax(0).float()
        acc_q = bayes_optimal_predictions[self.mc_samples :].mean(0)
        acc_p = 1 - bayes_optimal_predictions[: self.mc_samples].mean(0)

        acc = 0.5 * (acc_p + acc_q)

        return acc

