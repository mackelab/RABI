import torch
from torch import nn

from rbi.models.base import ParametricProbabilisticModel
from rbi.models.generators import (
    BernoulliGenerator,
    CategoricalGenerator,
    DiagonalGaussianGenerator,
    MultivariateNormalGenerator,
    MixtureDiagNormalGenerator,
)
from rbi.models.module import generate_dense_net, generate_conv_net
from rbi.models.activations import PartialNonlinearity

from typing import Callable, Union, List


class BernoulliNet(ParametricProbabilisticModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: List[int] = [50, 50],
        nonlinearity: Callable = nn.ReLU,
        prediction_fn: Union[str, Callable] = "argmax",
        **kwargs
    ):
      
        if output_dim > 1:
            raise ValueError("The output dimension here must always be one...")

        self.input_dim = input_dim 
        self.output_dim = output_dim
        embedding_net = kwargs.get("embedding_net", nn.Identity())
        input_dim = embedding_net(torch.randn(1, input_dim)).shape[-1]
        net = generate_dense_net(
            input_dim, output_dim, hidden_dims, nonlinearity, nn.Sigmoid()  # type: ignore
        )
        super().__init__(net, BernoulliGenerator(output_dim), **kwargs)
        self._set_prediction_fn(prediction_fn)


class CategoricalNet(ParametricProbabilisticModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [50, 50],
        nonlinearity: Callable = nn.ReLU,
        prediction_fn: Union[str, Callable] = "argmax",
        **kwargs
    ):
        self.input_dim = input_dim 
        self.output_dim = output_dim
        embedding_net = kwargs.get("embedding_net", nn.Identity())
        input_dim = embedding_net(torch.randn(1, input_dim)).shape[-1]
        net = generate_dense_net(
            input_dim, output_dim, hidden_dims, nonlinearity, nn.Softmax()  # type: ignore
        )
        super().__init__(net, CategoricalGenerator(output_dim), **kwargs)
        self._set_prediction_fn(prediction_fn)


class IndependentGaussianNet(ParametricProbabilisticModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [50, 50],
        nonlinearity: Callable = nn.ReLU,
        prediction_fn: Union[str, Callable] = "mean",
        min_scale: float = 1e-8,
        **kwargs
    ):
        self.input_dim = input_dim 
        self.output_dim = output_dim
        embedding_net = kwargs.get("embedding_net", nn.Identity())
        input_dim = embedding_net(torch.randn(1, input_dim)).shape[-1]
        net = generate_dense_net(
            input_dim,
            2 * output_dim,
            hidden_dims,
            nonlinearity,
            PartialNonlinearity(
                [nn.Identity(), nn.Softplus()], [output_dim, output_dim]
            ),  # type: ignore
        )
        super().__init__(
            net, DiagonalGaussianGenerator(output_dim, min_scale=min_scale), **kwargs
        )
        self._set_prediction_fn(prediction_fn)


class MultivariateGaussianNet(ParametricProbabilisticModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [50, 50],
        nonlinearity: Callable = nn.ReLU,
        prediction_fn: Union[str, Callable] = "mean",
        min_scale: float = 1e-8,
        **kwargs
    ):
        self.input_dim = input_dim 
        self.output_dim = output_dim
        embedding_net = kwargs.get("embedding_net", nn.Identity())
        input_dim = embedding_net(torch.randn(1, input_dim)).shape[-1]
        net = generate_dense_net(
            input_dim,
            output_dim + int(output_dim * (output_dim + 1) / 2),
            hidden_dims,
            nonlinearity,
            nn.Identity(),
        )
        self.output_dim = output_dim
        super().__init__(
            net, MultivariateNormalGenerator(output_dim, min_scale=min_scale), **kwargs
        )
        self._set_prediction_fn(prediction_fn)


class MixtureDiagGaussianModel(ParametricProbabilisticModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_components: int = 10,
        hidden_dims: List[int] = [50, 50],
        nonlinearity: Callable = nn.ReLU,
        prediction_fn: Union[str, Callable] = "mean",
        min_scale: float = 1e-8,
        **kwargs
    ):
        self.input_dim = input_dim 
        self.output_dim = output_dim
        embedding_net = kwargs.get("embedding_net", nn.Identity())
        input_dim = embedding_net(torch.randn(1, input_dim)).shape[-1]
        net = generate_dense_net(
            input_dim,
            num_components + num_components * output_dim * 2,
            hidden_dims,
            nonlinearity,
            nn.Identity(),
        )
        super().__init__(
            net,
            MixtureDiagNormalGenerator(output_dim, num_components, min_scale=min_scale),
            **kwargs
        )

        self._set_prediction_fn(prediction_fn)
