from grpc import Call
import torch
from torch import Tensor

from typing import List, Callable, Tuple, Union

from functorch import jvp, vjp, vmap


def batch_jacobian(
    func: Callable,
    x: Tensor,
    create_graph: bool = False,
    vectorize: bool = True,
    strategy: str = "reverse-mode",
) -> Tensor:
    """This computes the Jacobian matrix of func with respect to batches of inputs x.

    Args:
        func (Callable): The function.
        x (Tensor): The inputs.
        create_graph (bool, optional): If a computation graph should be created. Defaults to False.
        vectorize (bool, optional): Speeds up computation, but may needs a lot of memory. Defaults to True.
        strategy (str, optional): Autodiff algorithm. Defaults to "reverse-mode".

    Returns:
        Tensor: Jacobian matrices for each x
    """
    batch_shape = x.shape[:-1]
    dim1 = x.shape[-1]

    x = x.reshape(-1, dim1)

    def _func_sum(x):
        return func(x).sum(dim=0)

    jacs = torch.autograd.functional.jacobian(
        _func_sum,
        x,
        create_graph=create_graph,
        vectorize=vectorize,
        strategy=strategy,
        strict=False,
    ).permute(  # type: ignore
        1, 0, 2
    )

    dim2 = jacs.shape[-2]

    return jacs.reshape(*batch_shape, dim2, dim1)


def batch_hessian(
    func: Callable,
    x: Tensor,
    create_graph: bool = False,
    vectorize: bool = True,
    strategy: str = "reverse-mode",
) -> Tensor:
    """Computes the Hessian matrix for each input x

    Args:
        func (Callable): The functional.
        x (Tensor): The inputs.
        create_graph (bool, optional): If a computation graph should be created. Defaults to False.
        vectorize (bool, optional): Speeds up computation, but may needs a lot of memory. Defaults to True.
        strategy (str, optional): Autodiff algorithm. Defaults to "reverse-mode".

    Returns:
        Tensor: Hessian matrices for each x.
    """

    def _func(x):
        return batch_jacobian(
            func,
            x,
            create_graph=True,
            strategy="reverse-mode",
            vectorize=vectorize,  # Here reverse mode is always more efficient
        ).squeeze(-2)

    return batch_jacobian(
        _func, x, create_graph=create_graph, vectorize=vectorize, strategy=strategy
    )


def batch_jacobian_norm(
    func: Callable,
    x: Tensor,
    ord: Union[str, float] = "fro",
    create_graph: bool = False,
    vectorize: bool = True,
    strategy: str = "reverse-mode",
) -> Tensor:
    """Computes the Jacobian matrix norms.

    Args:
        func (Callable): The function.
        x (Tensor): The inputs
        ord (Union[str, float], optional): The matrix norm. Defaults to "fro".
        create_graph (bool, optional): If a computation graph should be created. Defaults to False.
        vectorize (bool, optional): Speeds up computation but may requires a lot of memory. Defaults to True.
        strategy (str, optional): Autodiff algorithm. Defaults to "reverse-mode".

    Returns:
        Tensor: The norms.
    """
    jacs = batch_jacobian(
        func, x, create_graph=create_graph, vectorize=vectorize, strategy=strategy
    )
    return torch.linalg.matrix_norm(jacs, ord=ord)


def batch_jvp(func: Callable, x: Tensor, v: Tensor, return_eval: bool=False) -> Union[Tensor, Tuple]:
    """Performs a jacobian vector product. This function supports batch dimensions.

    Args:
        func (Callable): Function
        x (Tensor): Input of shape (b2, d) or (b1, b2, d)
        v (Tensor): Product vector of shape (b2, d) or (b1,b2,d)
        return_eval (bool, optional): If function also should return point evaluations at x. Defaults to False.

    Raises:
        ValueError: Wrong shapes...

    Returns:
        Tensor, Tuple: Point evaluations (optional) and JVP
    """
    if (x.ndim == 2 and v.ndim == 2) or (x.ndim == 1 and v.ndim == 1):
        assert (
            x.shape == v.shape
        ), "Wrong ndims, if ndim is 2 of x and v then shapes must match"
        batched_jvp = jvp
    elif (x.ndim == 3 and v.ndim == 2) or (x.ndim == 2 and v.ndim == 1):
        assert (
            x.shape[1:] == v.shape
        ), "Wrong shapes, if ndim of x is 3 then x.shape[1:] must mathc v.shape"
        batched_jvp = vmap(jvp, in_dims=(None, 0, None))
    elif (x.ndim == 2 and v.ndim == 3) or (x.ndim == 1 and v.ndim == 2):
        assert (
            x.shape == v.shape[1:]
        ), "Wrong shapes, if ndim of x is 3 then x.shape[1:] must mathc v.shape"
        batched_jvp = vmap(jvp, in_dims=(None, None, 0))
    else:
        raise ValueError("We only support 1 to 3 ndims with batches...")

    out = batched_jvp(func, (x,), (v,), strict=False)
    if not return_eval:
        return out[-1]
    else:
        return out


def batch_vjp(func: Callable, x: Tensor, v: Tensor, return_eval: bool=False) -> Union[Tuple, Tensor]:
    """Performs a vector jacobian product. This function supports batch dimensions.

    Args:
        func (Callable): Function
        x (Tensor): Input of shape (b2, d) or (b1, b2, d)
        v (Tensor): Product vector of shape (b2, d) or (b1,b2,d)
        return_eval (bool, optional): If function also should return point evaluations at x. Defaults to False.

    Raises:
        ValueError: Wrong shapes...

    Returns:
        Tensor, Tuple: Point evaluations (optional) and JVP
    """
    if (x.ndim == 2 and v.ndim == 2) or (x.ndim == 1 and v.ndim == 1):
        eval, batched_vjp = vjp(func, x)  # type: ignore
    elif (x.ndim == 3 and v.ndim == 2) or (x.ndim == 2 and v.ndim == 1):
        eval, vjb_b = vjp(func, x)  # type: ignore
        b = x.shape[0]
        v_dim = v.ndim
        batched_vjp = lambda v: vjb_b(v.repeat(b, *([1]*v_dim)))
    elif (x.ndim == 2 and v.ndim == 3) or (x.ndim == 1 and v.ndim == 2):
        eval, batch_jvp = vjp(func, x)  # type: ignore
        batched_vjp = vmap(batch_jvp)
    else:
        raise ValueError("We only support 1 to 3 ndims with batches...")

    if not return_eval:
        return batched_vjp(v)[0]
    else:
        return eval, batched_vjp(v)[0]


def batch_vjp_fn(func: Callable, x: Tensor, v_ndim:int=3, return_eval=False) -> Union[Tuple, Callable]:

    """ Returns a function that performs vector jacobian products.

    Args:
        func (Callable): Function
        x (Tensor): Input of shape (b2, d) or (b1, b2, d)
        v (Tensor): Product vector of shape (b2, d) or (b1,b2,d)
        return_eval (bool, optional): If function also should return point evaluations at x. Defaults to False.

    Raises:
        ValueError: Wrong shapes...

    Returns:
        Tensor, Tuple: Point evaluations (optional) and JVP
    """
    if (x.ndim == 2 and v_ndim == 2) or (x.ndim == 1 and v_ndim == 1):
        eval, batched_vjp = vjp(func, x)  # type: ignore
    elif (x.ndim == 3 and v_ndim == 2) or (x.ndim == 2 and v_ndim == 1):
        eval, vjb_b = vjp(func, x)  # type: ignore
        b = x.shape[0]
        v_dim = v_ndim
        batched_vjp = lambda v: vjb_b(v.repeat(b, *([1]*v_dim)))
    elif (x.ndim == 2 and v_ndim == 3) or (x.ndim == 1 and v_ndim == 2):
        eval, batch_jvp = vjp(func, x)  # type: ignore
        batched_vjp = vmap(batch_jvp)
    else:
        raise ValueError("We only support 1 to 3 ndims with batches...")

    if return_eval:
        return eval, lambda v: batched_vjp(v)[0]
    else:
        return lambda v: batched_vjp(v)[0]



def batch_jacobian_outer_product_hutchinson_trace(func, x, mc_samples=10):
    out, vjp = batch_vjp_fn(func, x, return_eval=True)  # type: ignore
    out_dim = out.shape[-1]
    v = torch.randn(mc_samples, *x.shape[:-1], out_dim)
    trace = (vjp(v) ** 2).mean(0).sum(-1)
    return trace


def batched_jacobian_outer_product(func, x, b):
    """Computes J^T E[(b b^T)] J"""
    out, vjp = batch_vjp_fn(func, x, return_eval=True)  # type: ignore
    out_dim = out.shape[-1]
    assert b.shape[-1] == out_dim, "Vector must mathc outputdimension..."
    v = vjp(b)
    F = torch.einsum("mbi, mbj -> bij", v, v) / b.shape[0]
    return F
