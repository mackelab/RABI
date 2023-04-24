import torch
from rbibm.metric.base import PredictiveMetric


class MedianLpDistanceToObsMetric(PredictiveMetric):
    def __init__(
        self, model, simulator, ord, n_samples=500, batch_size=100, **kwargs
    ) -> None:
        super().__init__(model, simulator, n_samples, batch_size=batch_size, **kwargs)
        self.ord = ord

    def _compare_predictes_xos(self, x_os, x_predictives):
        distance = torch.linalg.norm(x_predictives - x_os, dim=-1, ord=self.ord)
        return torch.median(distance, dim=0).values


class MedianL1DistanceToObsMetric(MedianLpDistanceToObsMetric):
    def __init__(
        self, model, simulator, n_samples=500, batch_size=100, **kwargs
    ) -> None:
        super().__init__(
            model, simulator, 1.0, n_samples, batch_size=batch_size, **kwargs
        )


class MedianL2DistanceToObsMetric(MedianLpDistanceToObsMetric):
    def __init__(
        self, model, simulator, n_samples=500, batch_size=100, **kwargs
    ) -> None:
        super().__init__(
            model, simulator, 2.0, n_samples, batch_size=batch_size, **kwargs
        )


class MedianLinfDistanceToObsMetric(MedianLpDistanceToObsMetric):
    def __init__(
        self, model, simulator, n_samples=500, batch_size=100, **kwargs
    ) -> None:
        super().__init__(
            model, simulator, torch.inf, n_samples, batch_size=batch_size, **kwargs
        )
