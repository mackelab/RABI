from sklearn.covariance import log_likelihood
import torch
from torch import Tensor
from typing import Optional, Union, Any, Callable
from rbibm.metric.base import ApproximationMetric
from scipy.stats import kstest, uniform

from rbi.loss import (
    ReverseKLLoss,
    ForwardKLLoss,
    C2STKnn,
    C2STBayesOptimal,
    NegativeLogLikelihoodLoss,
)


from rbi.loss.base import EvalLoss
from rbi.models.base import ParametricProbabilisticModel, PyroFlowModel
from rbi.loss.mmd import MMDsquaredOptimalKernel, MMDsquared
from rbi.loss.kernels import MultiKernel


class Metric2GroundTruth(ApproximationMetric):

    requires_posterior = True
    can_eval_x_xtilde = True

    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        ground_truth: Optional[Any],
        loss_fn: EvalLoss,
        reduction: Optional[str] = "mean",
        batch_size: Optional[int] = None,
        descending: bool = False,
        device: str = "cpu",
    ) -> None:
        """Base class for metrics that have a ground truth."""
        super().__init__(
            model,
            ground_truth=ground_truth,
            reduction=reduction,
            batch_size=batch_size,
            descending=descending,
            device=device,
        )
        self.loss_fn = loss_fn

    def _eval(self, xs: Tensor, target: Optional[Tensor] = None) -> Tensor:
        with torch.no_grad():
            if target is None:
                p = self.ground_truth.condition(xs)  # type: ignore
                q = self.model(xs)
                return self.loss_fn(q, p)
            else:
                p = self.ground_truth.condition(target)  # type: ignore
                q = self.model(xs)
                return self.loss_fn(q, p)


class MMDsquared2GroundTruthMetric(Metric2GroundTruth):
    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        ground_truth: Optional[Any],
        kernel,
        reduction: Optional[str] = "mean",
        batch_size: Optional[int] = None,
        descending: bool = False,
        device: str = "cpu",
        **kwargs
    ) -> None:

        if not isinstance(kernel, MultiKernel):
            loss_fn = MMDsquared(kernel, reduction=None, **kwargs)
        else:
            loss_fn = MMDsquaredOptimalKernel(kernel, reduction=None, **kwargs)

        super().__init__(
            model, ground_truth, loss_fn, reduction, batch_size, descending, device
        )


class ReverseKL2GroundTruthMetric(Metric2GroundTruth):
    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        ground_truth: Optional[Any],
        reduction: Optional[str] = "mean",
        batch_size: Optional[int] = None,
        device: str = "cpu",
        **kwargs
    ) -> None:
        """Reverse KL divergence to ground truth.

        Args:
            model (Union[ParametricProbabilisticModel, PyroFlowModel]): Model
            ground_truth (Optional[Any]): Ground truth
            reduction (Optional[str], optional): Reduction type. Defaults to "mean".
            batch_size (Optional[int], optional): Batch size. Defaults to None.
            descending (bool, optional): . Defaults to False.
            device (str, optional): _description_. Defaults to "cpu".
        """
        loss_fn = ReverseKLLoss(reduction=None, **kwargs)  # type: ignore
        super().__init__(
            model,
            ground_truth,
            loss_fn,
            reduction=reduction,
            batch_size=batch_size,
            device=device,
        )


class ForwardKL2GroundTruthMetric(Metric2GroundTruth):
    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        ground_truth: Optional[Any],
        reduction: Optional[str] = "mean",
        batch_size: Optional[int] = None,
        device: str = "cpu",
        **kwargs
    ) -> None:
        """Forward KL divergence two ground truth.

        Args:
            model (Union[ParametricProbabilisticModel, PyroFlowModel]): Model
            ground_truth (Optional[Any]): Groudn truth
            reduction (Optional[str], optional): Reduction. Defaults to "mean".
            batch_size (Optional[int], optional): Batch size. Defaults to None.
            device (str, optional): Device. Defaults to "cpu".
        """
        loss_fn = ForwardKLLoss(reduction=None, **kwargs)  # type: ignore
        super().__init__(
            model,
            ground_truth,
            loss_fn,
            reduction=reduction,
            batch_size=batch_size,
            device=device,
        )


class C2STKnn2GroundTruthMetric(Metric2GroundTruth):
    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        ground_truth: Optional[Any],
        reduction: Optional[str] = "mean",
        batch_size: Optional[int] = None,
        device: str = "cpu",
        **kwargs
    ) -> None:
        """C2ST computed by a KNN algorithm.

        Args:
            model (Union[ParametricProbabilisticModel, PyroFlowModel]): Model
            ground_truth (Optional[Any]): Ground truth
            reduction (Optional[str], optional): Reduction. Defaults to "mean".
            batch_size (Optional[int], optional): Batch size. Defaults to None.
            device (str, optional): Device. Defaults to "cpu".
        """
        loss_fn = C2STKnn(reduction=None, **kwargs)  # type: ignore
        super().__init__(
            model,
            ground_truth,
            loss_fn,
            reduction=reduction,
            batch_size=batch_size,
            device=device,
        )


class C2STBayesOptimal2GroundTruthMetric(Metric2GroundTruth):
    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        ground_truth: Optional[Any],
        reduction: Optional[str] = "mean",
        batch_size: Optional[int] = None,
        device: str = "cpu",
        **kwargs
    ) -> None:
        """C2ST by Bayes optimal classifier.

        Args:
            model (Union[ParametricProbabilisticModel, PyroFlowModel]): Model
            ground_truth (Optional[Any]): Ground truth
            reduction (Optional[str], optional): Reduction. Defaults to "mean".
            batch_size (Optional[int], optional): Batch size. Defaults to None.
            device (str, optional): Device. Defaults to "cpu".
        """
        loss_fn = C2STBayesOptimal(reduction=None, **kwargs)  # type: ignore
        super().__init__(
            model,
            ground_truth,
            loss_fn,
            reduction=reduction,
            batch_size=batch_size,
            device=device,
        )


class NegativeLogLikelihoodMetric(ApproximationMetric):
    requires_thetas = True

    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        reduction: Optional[str] = "mean",
        batch_size: Optional[int] = None,
        device: str = "cpu",
        **kwargs
    ) -> None:
        """Negative loglikelihood metric between x and theta.

        Args:
            model (Union[ParametricProbabilisticModel, PyroFlowModel]): Model
            reduction (Optional[str], optional): Reduction. Defaults to "mean".
            batch_size (Optional[int], optional): Batch size. Defaults to None.
            device (str, optional): Device. Defaults to "cpu".
        """
        self.loss_fn = NegativeLogLikelihoodLoss(reduction=None)  # type: ignore
        super().__init__(
            model, reduction=reduction, batch_size=batch_size, device=device
        )

    def _eval(self, xs: Tensor, target: Optional[Tensor]) -> Tensor:
        with torch.no_grad():
            q = self.model(xs)
            return self.loss_fn(q, target)


class ExpectedCoverageMetric(ApproximationMetric):

    requires_thetas = True
    can_eval_single = False

    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        mc_samples: int = 100,
        alphas: Tensor = torch.linspace(0, 1, 100),
        device: str = "cpu",
        batch_size: int = 1000,
        reduction: Optional[str] = "absAUC",
        **kwargs
    ):
        """Expected coverage metric

        Args:
            model (Union[ParametricProbabilisticModel, PyroFlowModel]): Model
            mc_samples (int, optional): Samples used in estimate. Defaults to 100.
            alphas (Tensor, optional): Alphas. Defaults to torch.linspace(0, 1, 25).
            device (str, optional): Device. Defaults to "cpu".
            batch_size (int, optional): Batch size. Defaults to 1000.
            reduction (Optional[str], optional): Reduction. Defaults to None.
        """
        super().__init__(
            model=model,
            device=device,
            batch_size=batch_size,
            reduction=reduction,
        )
        self.mc_samples = mc_samples
        self.alphas = alphas.to(device)

    def _set_reduction(self, x):
        if x in ["AUC", "absAUC", "AUC_full", "absAUC_full", None]:
            self._reduction = x
        else:
            raise ValueError("Unknown reduction type")

    def _reduce(self, m):
        if self._reduction == "absAUC" or "absAUC_full":
            while m.ndim > 1:
                m = m.mean(0)

            val = torch.trapz(torch.abs(m - self.alphas), self.alphas)

            if self._reduction == "absAUC":
                return val
            else:
                return val, {"alphas": self.alphas.cpu(), "quantiles": m.cpu()}
        elif self._reduction == "AUC" or "AUC_full":
            while m.ndim > 1:
                m = m.mean(0)

            val = torch.trapz(m, self.alphas)
            if self._reduction == "AUC":
                return val
            else:
                return val, {"alphas": self.alphas.cpu(), "quantiles": m.cpu()}
        else:
            val = m.reshape(-1, self.alphas.shape[-1])
            return val

    def _eval(self, xs: Tensor, thetas: Tensor) -> Tensor:
        q = self.model(xs)
        logq_theta = q.log_prob(thetas)
        samples = q.sample((self.mc_samples,))
        logq_samples = q.log_prob(samples)
        return torch.stack(
            [
                self.expected_alpha_coverage(logq_theta, logq_samples, a)
                for a in self.alphas
            ]
        )

    def expected_alpha_coverage(
        self, logq_theta, logq_samples, alpha: Tensor
    ) -> Tensor:
        if float(alpha) == 0:
            return torch.zeros(1, device=self.device).mean()
        elif float(alpha) == 1:
            return torch.ones(1, device=self.device).mean()
        else:
            alpha = 1 - alpha
            cut_off = int(self.mc_samples * alpha)
            logq_samples_sorted, _ = torch.sort(logq_samples, dim=0)
            alpha_logprob_min = logq_samples_sorted[cut_off:].min(0).values

            return (alpha_logprob_min < logq_theta).float().mean()


class SimulationBasedCalibrationMetric(ApproximationMetric):

    requires_thetas = True
    can_eval_single = False

    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        mc_samples: int = 1000,
        device="cpu",
        batch_size: Optional[int] = 1000,
        **kwargs
    ) -> None:
        """Simulation based calibration metric

        Args:
            model (Union[ParametricProbabilisticModel, PyroFlowModel]): Model
            mc_samples (int, optional): Samples used in test. Defaults to 1000.
            device (str, optional): Devce. Defaults to "cpu".
            batch_size (Optional[int], optional): Batch size. Defaults to 1000.
        """
        super().__init__(model, descending=True, device=device, batch_size=batch_size)
        self.mc_samples = mc_samples

    def _eval(self, xs: Tensor, thetas: Tensor) -> Tensor:
        qs = self.model(xs)
        samples = qs.sample((self.mc_samples,))
        d = thetas.shape[-1]
        ranks = torch.zeros(d, thetas.shape[0], device=xs.device)
        for i in range(d):
            samples_marginal = samples[..., i]
            thetas_marginal = thetas[..., i]
            rank = (samples_marginal < thetas_marginal).sum(0)
            ranks[i] = rank

        kstest_pvals = torch.tensor(
            [
                kstest(rks, uniform(loc=0, scale=self.mc_samples).cdf)[1]
                for rks in ranks.cpu()
            ],
            dtype=torch.float32,
        )

        return kstest_pvals.to(xs.device)


class R2LinearFit2Potential(ApproximationMetric):

    requires_potential = True
    can_eval_x_xtilde = True

    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        potential_fn: Callable,
        mc_samples: int = 500,
        batch_size: Optional[int] = 100,
        reduction: Optional[str] = "mean",
        device: str = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            model,
            potential_fn=potential_fn,
            batch_size=batch_size,
            descending=True,
            reduction=reduction,
            device=device,
        )
        self.mc_samples = mc_samples

    def _eval(self, xs: Tensor, target: Optional[Tensor] = None) -> Tensor:
        with torch.no_grad():
            q = self.model(xs)

            samples = q.sample((self.mc_samples,))

            log_q = q.log_prob(samples)

            if target is None:
                log_potential = self.potential_fn(xs, samples)  # type: ignore
            else:

                xs, target = torch.broadcast_tensors(xs, target)
                log_potential = self.potential_fn(target, samples)  # type: ignore

            log_q, log_potential = torch.broadcast_tensors(log_q, log_potential)

            pot_max = torch.max(log_potential, dim=0, keepdim=True).values
            logM = (
                pot_max
                + torch.mean(
                    torch.exp(log_potential - pot_max), dim=0, keepdim=True
                ).log()
            )

            X = log_q.exp().transpose(0, -1).unsqueeze(-1)
            Y = torch.exp(log_potential - logM).transpose(0, -1).unsqueeze(-1)

            w = torch.linalg.solve(
                torch.transpose(X, dim0=-2, dim1=-1) @ X + 1e-10,
                torch.transpose(X, dim0=-2, dim1=-1) @ Y,
            )  # Linear regression

            residuals = Y - w * X
            var_res = torch.sum(residuals ** 2, dim=0)
            var_tot = torch.sum((Y - Y.mean(0, keepdim=True)) ** 2, dim=0)
            r2 = torch.clip(
                1 - var_res / (var_tot + 1e-10), 0.0, 1.0
            )  # R2 statistic to evaluate fit

            r2 = torch.nan_to_num(r2, 0.0)

            return r2


class RelativeTotalVariance(ApproximationMetric):

    requires_potential = False
    can_eval_x_xtilde = True

    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        mc_samples: int = 1000,
        batch_size: Optional[int] = 100,
        reduction: Optional[str] = "mean",
        device: str = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            model,
            batch_size=batch_size,
            descending=True,
            reduction=reduction,
            device=device,
        )
        self.mc_samples = mc_samples

    def _eval(self, xs: Tensor, target: Optional[Tensor] = None) -> Tensor:
        with torch.no_grad():
            q = self.model(xs)
            try:
                total_var = q.variance.sum(-1)
            except:
                samples = q.sample((self.mc_samples,))
                total_var = samples.var(0).sum(-1)

            if target is None:
                return total_var
            else:
                q_t = self.model(target)
                try:
                    total_var_t = q_t.variance.sum(-1)
                except:
                    samples = q_t.sample((self.mc_samples,))
                    total_var_t = samples.var(0).sum(-1)
                return total_var / total_var_t


class MeanDifference(ApproximationMetric):

    requires_potential = False
    can_eval_x_xtilde = True

    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        mc_samples: int = 1000,
        batch_size: Optional[int] = 100,
        reduction: Optional[str] = "mean",
        device: str = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            model,
            batch_size=batch_size,
            descending=True,
            reduction=reduction,
            device=device,
        )
        self.mc_samples = mc_samples

    def _eval(self, xs: Tensor, target: Optional[Tensor] = None) -> Tensor:
        with torch.no_grad():
            q = self.model(xs)
            try:
                mean = q.mean
            except:
                samples = q.sample((self.mc_samples,))
                mean = samples.mean(0)
            total_diff = torch.linalg.norm(mean - mean, axis=1)

            if target is None:
                return total_diff
            else:
                q_t = self.model(target)
                try:
                    mean_t = q_t.mean
                except:
                    samples = q_t.sample((self.mc_samples,))
                    mean_t = samples.mean(0)
                total_diff_t = torch.linalg.norm(mean - mean_t, axis=1)
                return total_diff_t


class Correlation2Potential(ApproximationMetric):

    requires_potential = True
    can_eval_x_xtilde = True

    def __init__(
        self,
        model: Union[ParametricProbabilisticModel, PyroFlowModel],
        potential_fn: Callable,
        mc_samples: int = 500,
        batch_size: Optional[int] = 100,
        reduction: Optional[str] = "mean",
        device: str = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            model,
            potential_fn=potential_fn,
            batch_size=batch_size,
            descending=True,
            reduction=reduction,
            device=device,
        )
        self.mc_samples = mc_samples

    def _eval(self, xs: Tensor, target: Optional[Tensor] = None) -> Tensor:
        with torch.no_grad():
            q = self.model(xs)

            samples = q.sample((self.mc_samples,))

            log_q = q.log_prob(samples)

            if target is None:
                log_potential = self.potential_fn(xs, samples)  # type: ignore
            else:
                log_potential = self.potential_fn(xs, target)  # type: ignore

            log_q, log_potential = torch.broadcast_tensors(log_q, log_potential)

            pot_max = torch.max(log_potential, dim=0, keepdim=True).values
            logM = (
                pot_max
                + torch.mean(
                    torch.exp(log_potential - pot_max), dim=0, keepdim=True
                ).log()
            )

            X = log_q.exp().transpose(0, -1).unsqueeze(-1)
            Y = torch.exp(log_potential - logM).transpose(0, -1).unsqueeze(-1)

            std_X = X.std()
            std_Y = Y.std()
            cov = torch.mean(
                (X - X.mean(0, keepdim=True)) * (Y - Y.mean(0, keepdim=True)), 0
            )
            denominator = std_X * std_Y + 1e-10

            pcc = cov / denominator
            return pcc