from typing import Optional
from torch.nn.utils.parametrize import  register_parametrization
from torch.nn.utils.parametrizations import orthogonal, spectral_norm
from torch import nn
import torch.nn.functional as F
import torch
from torch import Tensor

def scaling(module: nn.Module, name: str = "weight", scaling_factor: float=1.0) -> None:
    """Scales the weights of the module by a constant

    Args:
        module (Module): Module to parameterize
        name (str, optional): Parameter to which this parameterization applies. Defaults to "weight".
        scaling_factor (float, optional): Scaling factor. Defaults to 1.0.
    """
    weight = getattr(module, name)
    register_parametrization(module, name, ScaleWeights(scaling_factor).to(weight.device))


def learnable_scaling(module, name: str = "weight", scaling_factor=1.0) -> None:
    """Learnable scaling for the weights of the module.

    Args:
        module (Module): Module to parameterize
        name (str, optional): Parameter to which this parameterization applies. Defaults to "weight".
        scaling_factor (float, optional): Scaling factor. Defaults to 1.0.
    """
    weight = getattr(module, name)
    register_parametrization(module, name, LearnableScaleWeights(scaling_factor).to(weight.device))


def linf_norm_bound(module: nn.Module, name: str = "weight", dim: Optional[int]=None, **kwargs) -> None:
    """ Bounds the linf norm of the weight by 1.

    Args:
        module (Module): Module to reparameterize.
        name (str, optional): Name of the parameter to apply. Defaults to "weight".
        dim (Optional[int], optional): Dims, should be determined correctly for dense an conv. Defaults to None.

    Raises:
        ValueError: Invalid input.
    """
    weight = getattr(module, name, None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError(
            "Module '{}' has no parameter or buffer with name '{}'".format(module, name)
        )

    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    register_parametrization(module, name, LinfNormBoundWeight(dim=dim).to(weight.device))


def l1_norm_bound(module, name: str = "weight", dim=None, **kwargs) -> None:
    """ Bounds the l1 norm of the weight by 1.

    Args:
        module (Module): Module to reparameterize.
        name (str, optional): Name of the parameter to apply. Defaults to "weight".
        dim (Optional[int], optional): Dims, should be determined correctly for dense an conv. Defaults to None.

    Raises:
        ValueError: Invalid input.
    """
    weight = getattr(module, name, None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError(
            "Module '{}' has no parameter or buffer with name '{}'".format(module, name)
        )

    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0

    register_parametrization(module, name, L1NormBoundWeight(dim=dim).to(weight.device))

def l2_norm_bound(module, name: str = "weight", dim=None, **kwargs) -> None:
    """ Bounds the l2 norm of the weight by 1.

    Args:
        module (Module): Module to reparameterize.
        name (str, optional): Name of the parameter to apply. Defaults to "weight".
        dim (Optional[int], optional): Dims, should be determined correctly for dense an conv. Defaults to None.

    Raises:
        ValueError: Invalid input.
    """
    weight = getattr(module, name, None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError(
            "Module '{}' has no parameter or buffer with name '{}'".format(module, name)
        )

    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0

    register_parametrization(module, name, L2NormBoundWeight(dim=dim).to(weight.device))


def frob_norm_bound(module, name: str = "weight", dim=None, **kwargs):
    """ Bounds the frobinious norm of the weight by 1.

    Args:
        module (Module): Module to reparameterize.
        name (str, optional): Name of the parameter to apply. Defaults to "weight".
        dim (Optional[int], optional): Dims, should be determined correctly for dense an conv. Defaults to None.

    Raises:
        ValueError: Invalid input.
    """
    weight = getattr(module, name, None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError(
            "Module '{}' has no parameter or buffer with name '{}'".format(module, name)
        )

    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0

    register_parametrization(module, name, FroNormBoundWeight(dim=dim).to(weight.device))

def nuc_norm_bound(module, name: str = "weight", dim=None, **kwargs):
    """ Bounds the frobinious norm of the weight by 1.

    Args:
        module (Module): Module to reparameterize.
        name (str, optional): Name of the parameter to apply. Defaults to "weight".
        dim (Optional[int], optional): Dims, should be determined correctly for dense an conv. Defaults to None.

    Raises:
        ValueError: Invalid input.
    """
    weight = getattr(module, name, None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError(
            "Module '{}' has no parameter or buffer with name '{}'".format(module, name)
        )

    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0

    register_parametrization(module, name, FroNormBoundWeight(dim=dim).to(weight.device))


def l2_lipschitz_bound(
    module: nn.Module, name: str = "weight", L:float=1.0, method="spectral", learnable_L:bool = False, **kwargs
):
    """ Bounds the L2 Lipchitz constant of a module by L.

    Args:
        module (Module): Module with lipchitz constant < L
        name (str, optional): Name of parameter to act on. Defaults to "weight".
        L (float, optional): Lipchitz bound. Defaults to 1.0.
        method (str, optional): Method to use. Defaults to "spectral".
        learnable_L (bool, optional): If L should be learnable. Defaults to False.

    Raises:
        ValueError: Strange method.
    """
    if method == "spectral":
        spectral_norm(module, name, **kwargs)
    elif method == "orthogonal":
        orthogonal(module, name, **kwargs)
    elif method == "full":
        l2_norm_bound(module, name, **kwargs)
    else:
        raise ValueError("Unknown normalization method")

    learn_scale = learnable_L
    if not learn_scale:
        scaling(module, name, L)
    else:
        learnable_scaling(module, name, L)


def linf_lipschitz_bound(module, name: str = "weight", L:float=1.0, learnable_L:bool=False, **kwargs):
    """ Bounds the linf Lipchitz constant of a module by L.

    Args:
        module (Module): Module with lipchitz constant < L
        name (str, optional): Name of parameter to act on. Defaults to "weight".
        L (float, optional): Lipchitz bound. Defaults to 1.0.
        method (str, optional): Method to use. Defaults to "spectral".
        learnable_L (bool, optional): If L should be learnable. Defaults to False.

    Raises:
        ValueError: Strange method.
    """
    linf_norm_bound(module, name, **kwargs)
    learn_scale = learnable_L
    if not learn_scale:
        scaling(module, name, L)
    else:
        learnable_scaling(module, name, L)


def l1_lipschitz_bound(module, name: str = "weight", L: float=1.0, learnable_L:bool=False, **kwargs):
    """ Bounds the l1 Lipchitz constant of a module by L.

    Args:
        module (Module): Module with lipchitz constant < L
        name (str, optional): Name of parameter to act on. Defaults to "weight".
        L (float, optional): Lipchitz bound. Defaults to 1.0.
        method (str, optional): Method to use. Defaults to "spectral".
        learnable_L (bool, optional): If L should be learnable. Defaults to False.

    Raises:
        ValueError: Strange method.
    """
    l1_norm_bound(module, name, **kwargs)
    learn_scale = learnable_L
    if not learn_scale:
        scaling(module, name, L)
    else:
        learnable_scaling(module, name, L)


class ScaleWeights(nn.Module):
    def __init__(self, scaling_factor: float=1.0):
        """This parameterization simply scales e.g. the weights by a constant

        Args:
            scaling_factor (float, optional): Scaling factor. Defaults to 1.0.
        """
        super().__init__()
        self.register_buffer("s", torch.tensor([scaling_factor]).float())

    def forward(self, x: Tensor) -> Tensor:
        return x * self.s


class LearnableScaleWeights(nn.Module):
    def __init__(self, scaling_factor: float=1.0):
        super().__init__()
        s = torch.tensor([scaling_factor], requires_grad=True).float()
        self.register_parameter("s", s)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        return x * self.s


class NormBoundWeight(nn.Module):
    def __init__(self, ord, dim: int = 0):
        """This bounds the matrix norm of the weight.

        Args:
            ord (int): Matrix norm order
            dim (int, optional): Dim. Defaults to 0.
        """
        super().__init__()
        self.dim = dim
        self.ord = ord

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        """ Majorly gets convolutional weights in the right form.

        Args:
            weight (torch.Tensor): Weight

        Returns:
            torch.Tensor: Reshaped weights
        """
        # Precondition
        assert weight.ndim > 1
        dim = self.dim if self.dim >= 0 else self.dim + weight.ndim

        if dim != 0:
            # permute dim to front
            weight = weight.permute(dim, *(d for d in range(weight.dim()) if d != dim))

        return weight.flatten(1)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.training:
            if weight.ndim == 1:
                return F.normalize(weight, p=self.ord, dim=0)
            else:
                weight_mat = self._reshape_weight_to_matrix(weight)
                weight_norm = torch.linalg.matrix_norm(weight_mat, ord=self.ord)
                normalizer = weight_norm.clip(min=1.)
                weight_normed = 1/normalizer * weight_mat
                return weight_normed.reshape(weight.shape)
        else:
            return weight

class LinfNormBoundWeight(NormBoundWeight):
    def __init__(self, dim: int = 0):
        """This bounds the infinity matrix norm of the weight.

        Args:
            ord (int): Matrix norm order
            dim (int, optional): Dim. Defaults to 0.
        """
        super().__init__(ord = torch.inf, dim=dim)
        self.dim = dim

class L1NormBoundWeight(NormBoundWeight):
    def __init__(self, dim: int = 0):
        """This bounds the l1 matrix norm of the weight.

        Args:
            ord (int): Matrix norm order
            dim (int, optional): Dim. Defaults to 0.
        """
        super().__init__(ord = 1, dim=dim)
        self.dim = dim

class L2NormBoundWeight(NormBoundWeight):
    """This bounds the l2 matrix norm of the weight.

    Args:
        ord (int): Matrix norm order
        dim (int, optional): Dim. Defaults to 0.
    """
    def __init__(self, dim: int = 0):
        super().__init__(ord = 2, dim=dim)
        self.dim = dim

class FroNormBoundWeight(NormBoundWeight):
    def __init__(self, dim: int = 0):
        """This bounds the frobinious matrix norm of the weight.

        Args:
            ord (int): Matrix norm order
            dim (int, optional): Dim. Defaults to 0.
        """
        super().__init__(ord = "fro", dim=dim)
        self.dim = dim

class NucBoundWeight(NormBoundWeight):
    """This bounds the nuclear matrix norm of the weight.

    Args:
        ord (int): Matrix norm order
        dim (int, optional): Dim. Defaults to 0.
    """
    def __init__(self, dim: int = 0):
        super().__init__(ord = "fro", dim=dim)
        self.dim = dim


class SVDReparametrization(nn.Module):
    # TODO: This makes it maybe cheaper and gives full control over the singular values...
    def __init__(self, weight):
        super().__init__()
        svd = torch.linalg.svd(weight, full_matrices=False)
        self.dim1 = weight.shape[0]
        self.dim2 = weight.shape[1]
        if self.dim1 > 1.0 and self.dim1 > 1.0:
            self.register_parameter("_U", nn.Parameter(svd.U))  #type: ignore
            self.register_parameter("_Vh", nn.Parameter(svd.Vh))#type: ignore
            self.register_parameter("_S", nn.Parameter(svd.S))  #type: ignore
            orthogonal(self, "_U")
            orthogonal(self, "_Vh")
        else:
            self.register_parameter("_S", nn.Parameter(torch.linalg.norm(weight)))  #type: ignore

    def forward(self, X):
        if self.dim1 > 1.0 and self.dim1 > 1.0:
            return self._U @ torch.diag_embed(self._S) @ self._Vh  #type: ignore
        else:
            return (1 / torch.linalg.norm(X)) * X * self._S

    def right_inverse(self, X):
        self._U.data, self._S.data, self._Vh.data = torch.linalg.svd(
            X, full_matrices=False
        )
        return X
