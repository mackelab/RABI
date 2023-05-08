from abc import abstractmethod
import math
from typing import Callable, List, Union

import torch
from torch import Tensor


class Kernel:
    """This implements a kernel function

    Raises:
        ValueError: If a combination of kernels is not p.s.d
        ValueError: If a combination of kernels is not p.s.d

    Returns:
        Tensor: k(x,y)
    """

    @abstractmethod
    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        pass

    def __add__(self, other: Union[Tensor, "Kernel"]) -> "Kernel":
        """Addition of kernels is a kernel

        Args:
            other (Union[Tensor, &#39;Kernel&#39;]): Kernel or constant.

        Raises:
            ValueError: Only addition with positive constants results in a valid kernel

        Returns:
            Kernel: Resulting new kernel.
        """
        if isinstance(other, Kernel):
            return KernelWrapper(lambda x, y: self(x, y) + other(x, y))
        else:
            condition = other >= 0
            if isinstance(condition, Tensor):
                condition = condition.all()
            if condition:
                return KernelWrapper(lambda x, y: self(x, y) + other)
            else:
                raise ValueError("May not postitive definite")

    __radd__ = __add__

    def __mul__(self, other: Union[Tensor, "Kernel"]) -> "Kernel":
        """Mulitplication of kernels is a Kernel

        Args:
            other (Union[Tensor, &#39;Kernel&#39;]): Kernel or constant

        Raises:
            ValueError: Only multiplication with positive constants results in a valid kernel.

        Returns:
            Kernel: Resulting new kernel.
        """

        if isinstance(other, Kernel):
            return KernelWrapper(lambda x, y: self(x, y) * other(x, y))
        else:
            condition = other >= 0
            if isinstance(condition, Tensor):
                condition = condition.all()
                
            if condition:
                return KernelWrapper(lambda x, y: self(x, y) * other)
            else:
                raise ValueError("May not postitive definite")

    __rmul__ = __mul__


class KernelWrapper(Kernel):
    """Wrapper used to combine two kernels."""

    def __init__(self, kernel_fn: Callable) -> None:
        """Gets a callable which computes the kernel value and wraps it a Kernel

        Args:
            kernel_fn (Callable): Computes kernel value.
        """
        super().__init__()
        self.kernel_fn = kernel_fn

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return self.kernel_fn(x, y)


class MultiKernel(Kernel):
    """Evaluates multiple kernels
    """
    def __init__(self, kernels: List[Kernel]) -> None:
        """Evaluates a list of kernels

        Args:
            kernels (List[Kernel]): List of kernels
        """
        super().__init__()
        self.kernels = kernels

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        Ks = []
        for k in self.kernels:
            Ks.append(k(x, y))

        return torch.stack(Ks, dim=-1)


class RBFKernel(Kernel):
    """RBF kernel.

    Args:
        sigma (float, optional): Variance within kernel. Defaults to 1.0.
        l (float, optional): Length scale. Defaults to 1.0.

    Returns:
        Tensor: Kernel matrix.
    """

    def __init__(self, sigma: float = 1.0, l: float = 10.0) -> None:
        super().__init__()
        self.sigma = sigma
        self.l = l

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return self.sigma**2 * torch.exp(-0.5 * torch.sum((x - y) ** 2, -1) / self.l)


class MultiDimRBFKernel(Kernel):
    """RBF kernel, per dimension! Each dimension can have different variance and length scale.

    Args:
        sigma (float, optional): Variance within kernel. Defaults to 1.0.
        l (float, optional): Length scale. Defaults to 1.0.

    Returns:
        Tensor: Kernel matrix.
    """

    def __init__(
        self, sigma: Union[Tensor, float] = 1.0, l: Union[Tensor, float] = 1.0, reduction:str="prod"
    ) -> None:
        super().__init__()
        self.sigma = torch.as_tensor(sigma)
        self.l = torch.as_tensor(l)
        self.reduction = reduction


    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        sigma = self.sigma
        l = self.l
        while sigma.ndim < x.ndim:
            sigma = sigma.unsqueeze(0)

        while sigma.ndim < x.ndim:
            l = l.unsqueeze(0)

        if self.reduction == "prod":
            return torch.prod(self.sigma**2 * torch.exp(-0.5 * (x - y) ** 2 / self.l), -1)
        elif self.reduction == "sum":
            return torch.sum(self.sigma**2 * torch.exp(-0.5 * (x - y) ** 2 / self.l), -1)
        else:
            raise ValueError("...")



class RationalQuadraticKernel(Kernel):
    """Ratio quadratic kernel.

    Args:
        sigma (float, optional): Variance within kernel. Defaults to 1.0.
        l (float, optional): Length scale. Defaults to 1.0.
        alpha (float, optional): Order i.e. how often the function should be differentible . Defaults to 2.0.

    Returns:
        Tensor: Kernel matrix
    """

    def __init__(self, sigma: float = 1.0, l: float = 10.0, alpha: float = 2.0) -> None:
        super().__init__()
        self.sigma = sigma
        self.l = l
        self.alpha = alpha

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return self.sigma**2 * (
            (1 + torch.sum((x - y) ** 2, dim=-1) / (2 * self.alpha * self.l))
        ) ** (-self.alpha)

class LaplaceKernel(Kernel):
    """Periodic kernel.

    Args:
        sigma (float, optional): Variance within kernel. Defaults to 1.0.
        l (float, optional): Length scale. Defaults to 1.0.

    Returns:
        Tensor: Kernel matrix
    """

    def __init__(self, sigma: float = 2.0, l: float = 2.0) -> None:
        super().__init__()
        self.sigma = sigma
        self.l = l

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return self.sigma**2 * torch.exp(- torch.sum(torch.abs(x-y), -1)/self.l)
        


class LinearKernel(Kernel):
    """Linear kernel.

    Args:
        sigma (float, optional): Variance within kernel. Defaults to 1.0.
        l (float, optional): Length scale. Defaults to 1.0.

    Returns:
        Tensor: Kernel matrix
    """

    def __init__(self, sigma: float = 2.0) -> None:
        super().__init__()
        self.sigma = sigma

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return self.sigma**2  + torch.sum(x * y,-1)
        
    

class WhiteNoiseKernel(Kernel):
    """Constant kernel

    Args:
        Kernel (_type_): _description_
    """
    def __init__(self, sigma:float=1.0) -> None:
        super().__init__()
        self.sigma = sigma 
    
    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return self.sigma * (x == y).float()
    

class ConstantKernel(Kernel):
    """Constant kernel

    Args:
        Kernel (_type_): _description_
    """
    def __init__(self, sigma:float=1.0) -> None:
        super().__init__()
        self.sigma = sigma 
    
    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.as_tensor(self.sigma)
