
import torch

from rbi.defenses.base import DataAugmentationRegularizer


class DoublyRandomizedGaussianSmoothing(DataAugmentationRegularizer):
    def __init__(
        self,
        model,
        loss_fn,
        target_support_mapping=torch.nn.Identity(),
        eps=0.1,
    ):
        super().__init__(model, loss_fn)
        self.pertubation_scale = eps
        self.target_support_mapping = target_support_mapping

    def _transform(self, input, target):
        return input + torch.randn_like(
            input
        ) * self.pertubation_scale, self.target_support_mapping(
            target + torch.randn_like(input) * self.pertubation_scale
        )
