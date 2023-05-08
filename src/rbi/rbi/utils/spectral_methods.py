import torch
from torch import Tensor

from typing import Optional, Tuple

from rbi.utils.streaming_estimators import ExponentialMovingAverageEstimator, MovingAverageEstimator


@torch.jit.script  # type: ignore
def power_iterations(A: Tensor, init_vec: Optional[Tensor]=None, tol: float = 1e-2, max_iters: int=50) -> Tuple[Tensor, Tensor]:
    """Performs power iterations to compute largest eigenvector.

    Args:
        A (Tensor): Matrix of shape [b,i,j]
        init_vec (Optional[Tensor], optional): Init vec of shape [b,i] if not given it will be random. Defaults to None.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-2.
        max_iters (int, optional): Max number of iterations. Defaults to 50.

    Returns:
        Tuple[Tensor, Tensor]: Eigenvector and eigenvalue
    """
    if init_vec is None:
        x = torch.rand(A.shape[0], A.shape[-1])*2 -1
    else:
        x = init_vec

    converged = False
    i = 0

    while not converged:
        i += 1
        x_new = torch.einsum("bij, bj -> bi", A, x)
        x_new = x_new/torch.max(x_new.abs(), dim=-1, keepdim=True)[0]
        converged = ((x-x_new).abs() < tol).float().mean() > 0.5 or i > max_iters
        x = x_new

    eigenvec = x/torch.linalg.norm(x, dim=-1, keepdim=True)
    eigenval = torch.einsum("bi, bij, bj -> b", eigenvec, A, eigenvec)

    return eigenvec, eigenval


@torch.jit.script  # type: ignore
def power_iterations_with_momentum(A: Tensor, init_vec: Optional[Tensor]=None, tol: float = 1e-2, max_iters: int=50, beta: float= 0.6):
    """Performs power iterations with momentum to compute largest eigenvector.

    Args:
        A (Tensor): Matrix of shape [b,i,j]
        init_vec (Optional[Tensor], optional): Init vec of shape [b,i] if not given it will be random. Defaults to None.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-2.
        max_iters (int, optional): Max number of iterations. Defaults to 50.
        beta (float, optional): Momentum term.

    Returns:
        Tuple[Tensor, Tensor]: Eigenvector and eigenvalue
    """
    if init_vec is None:
        x = torch.rand(A.shape[0], A.shape[-1])*2 - 1
    else:
        x = init_vec

    converged = False
    i = 0
    xs = torch.zeros((3,) + x.shape)
    xs[1] = x
    while not converged:
        i += 1
        idx_old = (i-1)% 3
        idx_current = (i)%3 
        idx_new = (i+1)%3

        xs[idx_new] = torch.einsum("bij, bj -> bi", A, xs[idx_current]) - beta*xs[idx_old]
        xs[idx_new] = xs[idx_new]/torch.max( xs[idx_new].abs(), dim=-1, keepdim=True)[0]
        xs[idx_current] = xs[idx_current]/torch.max( xs[idx_new].abs(), dim=-1, keepdim=True)[0]

        converged = ((xs[idx_current]-xs[idx_new]).abs() < tol).float().mean() > 0.5 or i > max_iters

    eigenvec = xs[-1]/torch.linalg.norm(xs[-1], dim=-1, keepdim=True)
    eigenval = torch.einsum("bi, bij, bj -> b", eigenvec, A, eigenvec)

    return eigenvec, eigenval


@torch.jit.script  # type: ignore
def ada_oja_update(A: Tensor, x_old: Tensor, eta_old: Tensor, t: float) -> Tuple[Tensor, Tensor]:
    """Oja update with adaptive learning rate

    See: https://arxiv.org/pdf/1905.12115.pdf

    Args:
        A (Tensor): Matrix
        x_old (Tensor): Old eigenvector
        eta_old (Tensor): Old learning rate
        t (float): Time step

    Returns:
        Tuple[Tensor, Tensor]: Updated eigenvector and learning rate
    """
    G = torch.einsum("bij, bj -> bi", A, x_old)
    eta_new = torch.sqrt(eta_old**2 + torch.sum(G**2, dim=-1, keepdim=True) )
    x_new = x_old + 1/eta_new * G
    x_new = x_new/torch.linalg.norm(x_new, dim=-1, keepdim= True)
    return x_new, eta_new

@torch.jit.script  # type: ignore
def oja_update(A: Tensor, x_old: Tensor, eta_old: Tensor, t: float) -> Tuple[Tensor, Tensor]:
    """Oja update with decaying learning rate.

    Args:
        A (Tensor): Matrix
        x_old (Tensor): Old eigenvector
        eta_old (Tensor): Old learning rate
        t (float): Time step

    Returns:
         Tuple[Tensor, Tensor]: Updated eigenvector and learning rate
    """
    G = torch.einsum("bij, bj -> bi", A, x_old)
    x_new = x_old + eta_old * G 
    x_new = x_new/torch.linalg.norm(x_new, dim=-1, keepdim= True)
    eta_new = eta_old / t
    return x_new, eta_new
    


class StreamingOjaAlgorithm():
    def __init__(self, method = "oja", biased=True, ema_decay: float=0.95) -> None:
        """Implements an EMA estimator

        Args:
            decay (float, optional): Decay rete on history dependence. Defaults to 0.99.
        """
        self.t = 0
        if method == "adaoja":
            self._update = ada_oja_update
            self.eta_init = 1e-5
        elif method == "oja":
            self._update = oja_update
            self.eta_init = 10.

        if not biased:
            self._A = MovingAverageEstimator()
        else:
            self._A = ExponentialMovingAverageEstimator(decay=ema_decay)
        self._eigenvec = None
        self._eigenval = None
        self._eta = None

    def __call__(self, A: Tensor):
        """ Adds a new sample to the EMA."""
        self._A(A)
        A = self._A.value  # type: ignore
        if self._eigenvec is None:
            self._eigenvec = torch.randn(A.shape[0], A.shape[-1])
            self._eigenvec /= torch.linalg.norm(self._eigenvec, dim=-1, keepdim=True)
        if self._eta is None:
            self._eta = torch.ones(A.shape[0], 1)*self.eta_init

        self.t += 1
        x_new, eta_new =  self._update(A, self._eigenvec, self._eta, self.t)
        self._eigenvec = x_new 
        self._eigenval =  torch.einsum("bi, bij, bj -> b", self._eigenvec, A, self._eigenvec)
        self._eta = eta_new
        return x_new

    @property
    def eigenvec(self) -> Optional[Tensor]:
        """ Return current eigenvec estimate"""
        return self._eigenvec

    @property
    def eigenval(self) -> Optional[Tensor]:
        """ Return current eigenval estimate"""
        return self._eigenval

