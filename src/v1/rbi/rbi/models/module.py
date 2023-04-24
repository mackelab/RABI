from typing import Callable, List
import torch
from torch import nn

from math import sqrt

from typing import Union, Tuple


def generate_conv_net(
    input_dim: int,
    output_dim: int,
    in_channels: int = 1,
    nonlinearity: Callable = nn.ReLU,
    kernel_size: int = 3,
    strides: List[int] = [1, 2, 2],
    hidden_channels: List[int] = [16, 32, 64],
    group_norm: bool = True,
    num_groups: List[int] = [4, 32, 32],
    max_pool: bool = False,
) -> nn.Module:
    """Generates a convolutional neural network.

    Args:
        input_dim (int): Input dimension (not the shape)
        output_dim (int): Output dimension
        in_channels (int, optional): Input channels. Defaults to 1.
        nonlinearity (Callable, optional): Nonlinearity. Defaults to nn.ReLU.
        kernel_size (int, optional): Kernel size. Defaults to 3.
        strides (List[int], optional): Strides. Defaults to [1, 2, 2].
        hidden_channels (List[int], optional): Hidden channels. Defaults to [16, 32, 64].
        group_norm (bool, optional): If group norm should be applied. Defaults to True.
        num_groups (List[int], optional): Number of groups in group norm. Defaults to [4, 32, 32].
        max_pool (bool, optional): If max pooling should be performed. Defaults to False.

    Returns:
        Module: Convolutional neural net.
    """
    input_shape = (-1, in_channels, int(sqrt(input_dim)), int(sqrt(input_dim)))
    layers = [Reshape(input_shape)]
    i = 0
    input_channels = [in_channels] + hidden_channels[:-1]
    for s, c, i_c in zip(strides, hidden_channels, input_channels):
        layers.append(nn.Conv2d(i_c, c, kernel_size, stride=s))  # type: ignore
        if group_norm:
            layers.append(nn.GroupNorm(num_groups[i], c))  # type: ignore
        layers.append(nonlinearity())

        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size, s))  # type: ignore

    layers += [nn.Flatten()]
    net = nn.Sequential(*layers)
    input_test = torch.randn(1, input_dim)
    out = net(input_test)
    layers += [nn.Linear(out.shape[-1], output_dim)]
    net = nn.Sequential(*layers)

    return net


def generate_dense_net(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int] = [50, 50],
    nonlinearity: Callable = nn.ReLU,
    output_nonlinearity=nn.Identity(),
    batch_norm: bool = False,
) -> nn.Module:
    """Generates a dense net.

    Args:
        input_dim (int): Input dim.
        output_dim (int): Output dim.
        hidden_dims (List[int], optional): Hidden dimensions. Defaults to [50, 50].
        nonlinearity (Callable, optional): Nonlinearity. Defaults to nn.ReLU.
        output_nonlinearity (_type_, optional): Output nonlinearity. Defaults to nn.Identity().
        batch_norm (bool, optional): If we should use batch norm. Defaults to False.

    Returns:
        nn.Module: Dense neural net.
    """
    layers = []

    layers += [nn.Linear(input_dim, hidden_dims[0]), nonlinearity()]
    for i in range(len(hidden_dims) - 1):
        layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nonlinearity()])
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
    layers += [nn.Linear(hidden_dims[-1], output_dim), output_nonlinearity]
    net = nn.Sequential(*layers)
    return net


class Reshape(nn.Module):
    def __init__(self, shape: Union[torch.Size, Tuple]) -> None:
        """This module reshapes the input.

        Args:
            shape (Union[torch.Size, Tuple]): Output shape
        """
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)


class ZScoreLayer(nn.Module):
    def __init__(self, mean, std):
        """This module performs z-score normalization, given a mean and std.

        Args:
            mean (_type_): _description_
            std (_type_): _description_
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std


class ExchangableLinearLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        output_dim_phi:int=100,
        aggregation_dim:int=1,
        aggregation_fn=torch.sum,
        hidden_dims: List[int]=[50, 50],
        nonlinearity: Callable=nn.ReLU,
    ):
        """Generates a permutation invariant neural network.

        Args:
            input_dim (int): Input dim.
            output_dim (int): Output dim
            output_dim_phi (int, optional): Output feature or each trial. Defaults to 100.
            aggregation_dim (int, optional): Aggregation dim of trials. Defaults to 1.
            aggregation_fn (Callable, optional): Aggregation function of trials. Defaults to torch.sum.
            hidden_dims (list, optional): Hidden dimensions. Defaults to [50, 50].
            nonlinearity (Callable, optional): Nonlinearity. Defaults to nn.ReLU.
        """
        super().__init__()
        self.aggregation_dim = aggregation_dim
        self.phi = generate_dense_net(
            input_dim,
            output_dim=output_dim_phi,
            hidden_dims=hidden_dims,
            nonlinearity=nonlinearity,
            output_nonlinearity=nn.Identity(),
        )
        self.g = generate_dense_net(
            output_dim_phi,
            output_dim,
            hidden_dims=hidden_dims,
            nonlinearity=nonlinearity,
            output_nonlinearity=nn.Identity(),
        )
        self.aggregation_fn = aggregation_fn

    def forward(self, x):

        n_unsqueezes = max(3 - len(x.shape), 0)
        for i in range(n_unsqueezes):
            x = x.unsqueeze(0)

        h = self.phi(x)
        with torch.no_grad():
            h = torch.nan_to_num(h, 0.0)
        out = self.aggregation_fn(h, axis=self.aggregation_dim)  # type: ignore
        out = self.g(out)
        return out
