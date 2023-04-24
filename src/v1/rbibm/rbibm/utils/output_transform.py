from torch.distributions import biject_to
from torch.distributions.constraints import _IndependentConstraint, _Real
import torch


def get_output_transform(name, support=None):
    if name == "biject_to":
        if isinstance(support, _Real) or (
            isinstance(support, _IndependentConstraint)
            and isinstance(support.base_constraint, _Real)
        ):
            return None
        else:
            return biject_to(support)
    else:
        return None
