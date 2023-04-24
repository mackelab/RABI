from copy import deepcopy

from .base import Defense
import torch

from rbi.models.base import ParametricProbabilisticModel

from rbi.defenses.base import DataAugmentationRegularizer

from torch.distributions import transform_to


# TODO REFACTOR

# TODO Add settorch

