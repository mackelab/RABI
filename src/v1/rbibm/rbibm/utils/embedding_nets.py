from torch.distributions import biject_to
from torch import nn
import torch

from rbi.models.module import (
    ExchangableLinearLayer,
    generate_conv_net,
    generate_dense_net,
    ZScoreLayer,
)
from sbi.neural_nets.embedding_nets import CNNEmbedding


def get_embedding_net(cfg, input_dim):
    name = cfg.name
    if name == "identity":
        return nn.Identity()
    elif name == "exchangable":
        return ExchangableLinearLayer(
            input_dim=input_dim,
            output_dim=cfg.output_dim,
            output_dim_phi=cfg.output_dim_phi,
            aggregation_dim=cfg.aggregation_dim,
            aggregation_fn=eval(cfg.aggregation_fn),
            hidden_dims=cfg.hidden_dims,
            nonlinearity=cfg.nonlinearity,
        )
    elif name == "conv2d":
        return generate_conv_net(
            input_dim=input_dim,
            output_dim=cfg.output_dim,
            in_channels=cfg.in_channels,
            nonlinearity=eval(cfg.nonlinearity),
            kernel_size=cfg.kernel_size,
            strides=cfg.strides,
            hidden_channels=cfg.hidden_channels,
            group_norm=cfg.group_norm,
            max_pool=cfg.max_pool,
        )
    elif name == "mlp":
        return generate_dense_net(
            input_dim=input_dim,
            output_dim=cfg.output_dim,
            hidden_dims=cfg.hidden_dims,
            nonlinearity=eval(cfg.nonlinearity),
            output_nonlinearity=eval(cfg.output_nonlinearity),
            batch_norm=cfg.batch_norm,
        )
    elif name == "pyloric":
        return generate_pyloric_embedding_net(cnn_output_dim=cfg.cnn_output_dim)
    else:
        raise NotImplementedError("Yet not implemented...")


class CombinedEmbedding(nn.Module):
    def __init__(self, cnn_output_dim, cnn1, cnn2, cnn3, mlp):
        super().__init__()
        self.cnn_output_dim = cnn_output_dim
        self.cnn1 = cnn1
        self.cnn2 = cnn2
        self.cnn3 = cnn3
        self.mlp = mlp

    def forward(self, x):
        batch_shape = x.shape[:-1]
        x = x.reshape(batch_shape + (3, 800))
        cnn_embedded_vals = torch.hstack(
            [
                self.cnn1(x[..., 0, :].reshape(-1, 800)),
                self.cnn2(x[..., 1, :].reshape(-1, 800)),
                self.cnn3(x[..., 2, :].reshape(-1, 800)),
            ]
        )
        cnn_embedded_vals = cnn_embedded_vals.reshape(
            batch_shape + (3 * self.cnn_output_dim,)
        )

        mlp_embedding = self.mlp(cnn_embedded_vals)
        return mlp_embedding


def generate_pyloric_embedding_net(cnn_output_dim):

    cnn_nets = [
        CNNEmbedding(
            input_shape=(800,),
            num_conv_layers=3,
            out_channels_per_layer=[6, 9, 12],
            output_dim=cnn_output_dim,
            pool_kernel_size=3,
        )
        for _ in range(3)
    ]

    mlp_net = nn.Sequential(
        nn.Linear(3 * cnn_output_dim, 2 * cnn_output_dim),
        nn.ReLU(),
        nn.Linear(2 * cnn_output_dim, cnn_output_dim),
        nn.ReLU(),
        nn.Linear(cnn_output_dim, cnn_output_dim),
    )

    return CombinedEmbedding(cnn_output_dim, *cnn_nets, mlp_net)
