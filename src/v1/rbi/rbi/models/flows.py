from torch import nn
from rbi.models.base import PyroFlowModel, ZukoFlowModel

from rbi.models.transforms import (
    conditional_affine_autoregressive,  # type: ignore
    conditional_affine_coupling,  # type: ignore
    conditional_spline_coupling,
    conditional_inverse_affine_autoregressive,
    conditional_spline_autoregressive,  # type: ignore
    conditional_inverse_spline_autoregressive,
)
from pyro.distributions.transforms import Transform
from zuko.flows import MAF,NSF

from typing import List, Optional, Any


class MaskedAutoregressiveFlow(ZukoFlowModel):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [50,50], num_transforms: int = 5, randperm: bool = True, output_transform: Optional[Transform] = None, embedding_net: Optional[nn.Module] = None, **kwargs) -> None:
        super().__init__(MAF, input_dim, output_dim, hidden_dims, num_transforms, randperm, output_transform, embedding_net, **kwargs)


class NeuralSplineFlow(ZukoFlowModel):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [50,50], num_transforms: int = 5, randperm: bool = True, output_transform: Optional[Transform] = None, embedding_net: Optional[nn.Module] = None, **kwargs) -> None:
        super().__init__(NSF, input_dim, output_dim, hidden_dims, num_transforms, randperm, output_transform, embedding_net, **kwargs)


class AffineAutoregressiveModel(PyroFlowModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [50, 50],
        num_transforms: int = 5,
        shuffle: bool = True,
        with_cache: bool = False,
        output_transform: Optional[Transform] = None,
        embedding_net: Optional[nn.Module] = None,
        **kwargs
    ):
        """Affine autoregressive model. This is also known as IAF i.e. fast sampling and density evaluation on samples. But slow density evaluation on arbitrary values.

        Args:
            transform_generator (Callable): Function that generates a learnable transformation.
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            hidden_dims (List[int], optional): Hidden dims of nns used in learnable transformations . Defaults to [50, 50].
            num_transforms (int, optional): Number of transformations. Defaults to 5.
            shuffle (bool, optional): Permuting dimensions, good for autoregressive or coupling flows. Defaults to True.
            with_cache (bool, optional): If intermediate results should be cached to speed up computation. Defaults to False.
            output_transform (Optional[Transform], optional): Output transform. Defaults to None.
            embedding_net (Optional[nn.Module], optional): Embedding net. Defaults to None.
            log_scale_min_clip (float, optional): The minimum value for clipping the log(scale) from the autoregressive nn.
            log_scale_max_clip (float, optional): The maximum value for clipping the log(scale) from the autoregressive nn.
            stable: (bool, optional) When true, uses an alternative "stable" version. Defaults to False.
            sigmoid_bias: A term added to the logits when using the stable transform.
        """
        super().__init__(
            conditional_affine_autoregressive,
            input_dim,
            output_dim,
            hidden_dims,
            num_transforms,
            shuffle,
            with_cache,
            output_transform,
            embedding_net,
            **kwargs
        )


class InverseAffineAutoregressiveModel(PyroFlowModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [50, 50],
        num_transforms: int = 5,
        shuffle: bool = True,
        with_cache: bool = False,
        output_transform: Optional[Transform] = None,
        embedding_net: Optional[nn.Module] = None,
        **kwargs
    ):
        """Inverse affine autoregressive model. This is also known as MAF i.e. slow sampling, but fast density evaluation on arbitrary values.

        Args:
            transform_generator (Callable): Function that generates a learnable transformation.
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            hidden_dims (List[int], optional): Hidden dims of nns used in learnable transformations . Defaults to [50, 50].
            num_transforms (int, optional): Number of transformations. Defaults to 5.
            shuffle (bool, optional): Permuting dimensions, good for autoregressive or coupling flows. Defaults to True.
            with_cache (bool, optional): If intermediate results should be cached to speed up computation. Defaults to False.
            output_transform (Optional[Transform], optional): Output transform. Defaults to None.
            embedding_net (Optional[nn.Module], optional): Embedding net. Defaults to None.
            log_scale_min_clip (float, optional): The minimum value for clipping the log(scale) from the autoregressive nn.
            log_scale_max_clip (float, optional): The maximum value for clipping the log(scale) from the autoregressive nn.
            stable: (bool, optional) When true, uses an alternative "stable" version. Defaults to False.
            sigmoid_bias: A term added to the logits when using the stable transform.
        """
        super().__init__(
            conditional_inverse_affine_autoregressive,
            input_dim,
            output_dim,
            hidden_dims,
            num_transforms,
            shuffle,
            with_cache,
            output_transform,
            embedding_net,
            **kwargs
        )


class AffineCouplingModel(PyroFlowModel):
    """Affine coupling model

    Args:
        transform_generator (Callable): Function that generates a learnable transformation.
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        hidden_dims (List[int], optional): Hidden dims of nns used in learnable transformations . Defaults to [50, 50].
        num_transforms (int, optional): Number of transformations. Defaults to 5.
        shuffle (bool, optional): Permuting dimensions, good for autoregressive or coupling flows. Defaults to True.
        with_cache (bool, optional): If intermediate results should be cached to speed up computation. Defaults to False.
        output_transform (Optional[Transform], optional): Output transform. Defaults to None.
        embedding_net (Optional[nn.Module], optional): Embedding net. Defaults to None.
        log_scale_min_clip (float, optional): The minimum value for clipping the log(scale) from the autoregressive nn.
        log_scale_max_clip (float, optional): The maximum value for clipping the log(scale) from the autoregressive nn.
        stable: (bool, optional) When true, uses an alternative "stable" version. Defaults to False.
        sigmoid_bias: A term added to the logits when using the stable transform.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [50, 50],
        num_transforms: int = 5,
        shuffle: bool = True,
        with_cache: bool = False,
        output_transform: Optional[Transform] = None,
        embedding_net: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(
            conditional_affine_coupling,
            input_dim,
            output_dim,
            hidden_dims,
            num_transforms,
            shuffle,
            with_cache,
            output_transform,
            embedding_net,
            **kwargs
        )


class SplineAutoregressiveModel(PyroFlowModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [50, 50],
        num_transforms: int = 5,
        shuffle: bool = True,
        with_cache: bool = False,
        output_transform: Optional[Transform] = None,
        embedding_net: Optional[nn.Module] = None,
        **kwargs
    ):
        """Spline autoregressive model. This is also known as IAF i.e. fast sampling and density evaluation on samples. But slow density evaluation on arbitrary values.

        Args:
            transform_generator (Callable): Function that generates a learnable transformation.
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            hidden_dims (List[int], optional): Hidden dims of nns used in learnable transformations . Defaults to [50, 50].
            num_transforms (int, optional): Number of transformations. Defaults to 5.
            shuffle (bool, optional): Permuting dimensions, good for autoregressive or coupling flows. Defaults to True.
            with_cache (bool, optional): If intermediate results should be cached to speed up computation. Defaults to False.
            output_transform (Optional[Transform], optional): Output transform. Defaults to None.
            embedding_net (Optional[nn.Module], optional): Embedding net. Defaults to None.
            count_bins (int, optional): The number of segments comprising hte spline.
            bound (int, optional): Determines the bounding box of the spline i.e. from [-bound, bound].
            order (str, optional): One of 'linear' or 'quadratic' specifiying the order of the spline.
        """
        super().__init__(
            conditional_spline_autoregressive,
            input_dim,
            output_dim,
            hidden_dims,
            num_transforms,
            shuffle,
            with_cache,
            output_transform,
            embedding_net,
            **kwargs
        )


class InverseSplineAutoregressiveModel(PyroFlowModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [50, 50],
        num_transforms: int = 5,
        shuffle: bool = True,
        with_cache: bool = False,
        output_transform: Optional[Transform] = None,
        embedding_net: Optional[nn.Module] = None,
        **kwargs
    ):
        """Inverse spline autoregressive model.  This has slow sampling, but fast density evaluation on arbitrary values.


        Args:
            transform_generator (Callable): Function that generates a learnable transformation.
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            hidden_dims (List[int], optional): Hidden dims of nns used in learnable transformations . Defaults to [50, 50].
            num_transforms (int, optional): Number of transformations. Defaults to 5.
            shuffle (bool, optional): Permuting dimensions, good for autoregressive or coupling flows. Defaults to True.
            with_cache (bool, optional): If intermediate results should be cached to speed up computation. Defaults to False.
            output_transform (Optional[Transform], optional): Output transform. Defaults to None.
            embedding_net (Optional[nn.Module], optional): Embedding net. Defaults to None.
            count_bins (int, optional): The number of segments comprising hte spline.
            bound (int, optional): Determines the bounding box of the spline i.e. from [-bound, bound].
            order (str, optional): One of 'linear' or 'quadratic' specifiying the order of the spline.
        """
        super().__init__(
            conditional_inverse_spline_autoregressive,
            input_dim,
            output_dim,
            hidden_dims,
            num_transforms,
            shuffle,
            with_cache,
            output_transform,
            embedding_net,
            **kwargs
        )


class SplineCouplingModel(PyroFlowModel):
    """Spline coupling model


    Args:
        transform_generator (Callable): Function that generates a learnable transformation.
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        hidden_dims (List[int], optional): Hidden dims of nns used in learnable transformations . Defaults to [50, 50].
        num_transforms (int, optional): Number of transformations. Defaults to 5.
        shuffle (bool, optional): Permuting dimensions, good for autoregressive or coupling flows. Defaults to True.
        with_cache (bool, optional): If intermediate results should be cached to speed up computation. Defaults to False.
        output_transform (Optional[Transform], optional): Output transform. Defaults to None.
        embedding_net (Optional[nn.Module], optional): Embedding net. Defaults to None.
        count_bins (int, optional): The number of segments comprising hte spline.
        bound (int, optional): Determines the bounding box of the spline i.e. from [-bound, bound].
        order (str, optional): One of 'linear' or 'quadratic' specifiying the order of the spline.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [50, 50],
        num_transforms: int = 5,
        shuffle: bool = True,
        with_cache: bool = False,
        output_transform: Optional[Transform] = None,
        embedding_net: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(
            conditional_affine_coupling,
            input_dim,
            output_dim,
            hidden_dims,
            num_transforms,
            shuffle,
            with_cache,
            output_transform,
            embedding_net,
            **kwargs
        )
