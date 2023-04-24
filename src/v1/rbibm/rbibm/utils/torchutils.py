import torch
from torch.cuda.amp import custom_bwd, custom_fwd  # type: ignore
from torch import Tensor


class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input: Tensor, min: float, max: float):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)


def tensors_to_floats(input):
    if isinstance(input, dict):
        output = {}
        for key, val in input.items():
            output[key] = tensors_to_floats(val)
        return output
    elif isinstance(input, list) or isinstance(input, tuple):
        output = [tensors_to_floats(i) for i in input]
        return output
    elif isinstance(input, torch.Tensor):
        if input.numel() <= 1:
            return float(input.cpu())
        else:
            output = [tensors_to_floats(i) for i in list(input)]
            return output
    else:
        return None
