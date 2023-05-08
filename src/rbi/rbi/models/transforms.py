from functools import partial

import torch
import torch.nn as nn

from typing import Optional, List

from pyro.nn import ConditionalAutoRegressiveNN, DenseNN
from pyro.distributions.transforms import (
    AffineAutoregressive,
    SplineAutoregressive,
    ConditionalAffineAutoregressive,
    ConditionalSplineAutoregressive,
    SplineCoupling,
    conditional_affine_autoregressive,
    conditional_spline_autoregressive,
    conditional_affine_coupling,
)
from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions import constraints



class ConditionalInverseAffineAutoregressive(ConditionalTransformModule):

    __doc__ = ConditionalAffineAutoregressive.__doc__

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, autoregressive_nn, **kwargs):
        super().__init__()
        self.nn = autoregressive_nn
        self.kwargs = kwargs

    def condition(self, context):
        """
        Conditions on a context variable, returning a non-conditional transform of
        of type :class:`~pyro.distributions.transforms.AffineAutoregressive`.
        """

        cond_nn = partial(self.nn, context=context)
        cond_nn.permutation = cond_nn.func.permutation  # type: ignore
        cond_nn.get_permutation = cond_nn.func.get_permutation  # type: ignore
        return AffineAutoregressive(cond_nn, **self.kwargs).inv


class ConditionalInverseSplineAutoregressive(ConditionalTransformModule):

    __doc__ = ConditionalSplineAutoregressive.__doc__

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, input_dim, autoregressive_nn, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.nn = autoregressive_nn
        self.kwargs = kwargs

    def condition(self, context):
        """
        Conditions on a context variable, returning a non-conditional transform of
        of type :class:`~pyro.distributions.transforms.SplineAutoregressive`.
        """

        # Note that nn.condition doesn't copy the weights of the ConditionalAutoregressiveNN
        cond_nn = partial(self.nn, context=context)
        cond_nn.permutation = cond_nn.func.permutation  # type: ignore
        cond_nn.get_permutation = cond_nn.func.get_permutation  # type: ignore
        return SplineAutoregressive(self.input_dim, cond_nn, **self.kwargs).inv


class ConditionalSplineCoupling(ConditionalTransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, split_dim, hypernet, **kwargs):
        super().__init__()
        self.split_dim = split_dim
        self.nn = hypernet
        self.kwargs = kwargs

    def condition(self, context):
        cond_nn = partial(self.nn, context=context)
        return SplineCoupling(self.split_dim, cond_nn, **self.kwargs)


def conditional_inverse_affine_autoregressive(
    input_dim, context_dim, hidden_dims=None, **kwargs
):
    if hidden_dims is None:
        hidden_dims = [10 * input_dim]
    nn = ConditionalAutoRegressiveNN(input_dim, context_dim, hidden_dims)
    return ConditionalInverseAffineAutoregressive(nn, **kwargs)


def conditional_inverse_spline_autoregressive(
    input_dim, context_dim, hidden_dims=None, count_bins=8, bound=3.0, order="linear"
):
    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]

    param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
    arn = ConditionalAutoRegressiveNN(
        input_dim, context_dim, hidden_dims, param_dims=param_dims
    )
    return ConditionalInverseSplineAutoregressive(
        input_dim, arn, count_bins=count_bins, bound=bound, order=order
    )


def conditional_spline_coupling(
    input_dim: int,
    split_dim: Optional[int] = None,
    hidden_dims: Optional[List] = None,
    count_bins: int = 8,
    bound: float = 3.0,
):

    if split_dim is None:
        split_dim = input_dim // 2

    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]

    nn = DenseNN(
        split_dim,
        hidden_dims,
        param_dims=[
            (input_dim - split_dim) * count_bins,
            (input_dim - split_dim) * count_bins,
            (input_dim - split_dim) * (count_bins - 1),
            (input_dim - split_dim) * count_bins,
        ],
    )

    return SplineCoupling(input_dim, split_dim, nn, count_bins=count_bins, bound=bound)
