from ast import Call
import torch
from torch import nn

from typing import Optional, Callable, Iterable

_hooks_disabled: bool = False
_registered_forward_hooks = {}
_supported_modules_forward_hooks = {}

# Nice hook control-------------------------------------------------------


def register_forward_hook(
    hook: Optional[Callable] = None,
    name: Optional[str] = None,
    supported_modules: Optional[list] = None,
) -> Callable:
    def _register(hook):
        def deactivalable_hook(*args, **kwargs):
            global _hooks_disabled
            if _hooks_disabled:
                return
            else:
                return hook(*args, **kwargs)

        if name is None:
            hook_name = hook.__name__
        else:
            hook_name = name

        if supported_modules is None:
            supp_modules = "ALL"
        else:
            if not isinstance(supported_modules, Iterable):
                raise ValueError(
                    "We require a list of supported modules. If None is specified then it ALL are supported"
                )
            else:
                supp_modules = supported_modules

        if hook_name in _registered_forward_hooks:
            raise ValueError("Hook already registered")
        else:
            _registered_forward_hooks[hook_name] = deactivalable_hook
            _supported_modules_forward_hooks[hook_name] = supp_modules
        return deactivalable_hook

    if hook is None:
        return _register
    else:
        return _register(hook)


def get_forward_hook_and_supported_modules(name: str):
    if name not in _registered_forward_hooks:
        raise ValueError("Hook not registered, use 'register_hook' to do so...")
    else:
        return _registered_forward_hooks[name], _supported_modules_forward_hooks[name]


def add_forward_hook(model: nn.Module, name: str) -> None:
    hook, supported_moduels = get_forward_hook_and_supported_modules(name)
    handles = []
    for layer in model.modules():
        if supported_moduels == "ALL" or (layer.__class__ in supported_moduels):
            handles.append(layer.register_forward_hook(hook))

    model.__dict__.setdefault("hook_control_center", []).extend(handles)


def disable_all_hooks():
    global _hooks_disabled
    _hooks_disabled = True


def enable_all_hooks():
    global _hooks_disabled
    _hooks_disabled = False


def is_enabled_hooks():
    global _hooks_disabled
    return not _hooks_disabled


def remove_forward_hooks(model: nn.Module) -> None:
    if not hasattr(model, "hook_control_center"):
        print("Warning, asked to remove hooks, but no hooks found")
    else:
        for handle in model.hook_control_center:  # type: ignore
            handle.remove()
        del model.hook_control_center


class enabled_hooks:
    def __enter__(self):
        self.was_enabled = is_enabled_hooks()
        if not self.was_enabled:
            enable_all_hooks()

    def __exit__(self, *args, **kwargs):
        if not self.was_enabled:
            disable_all_hooks()


class disabled_hooks:
    def __enter__(self):
        self.was_enabled = is_enabled_hooks()
        if self.was_enabled:
            disable_all_hooks()

    def __exit__(self, *args, **kwargs):
        if self.was_enabled:
            enable_all_hooks()


# Forward hooks ----------------------------------------------------------------------


@register_forward_hook(name="relu_to_softplus", supported_modules=[nn.ReLU])
def relu_to_softplus_hook(layer, input, output, beta=10.0):
    """This maybe usefull to get a twice continously differentiable model"""
    return torch.nn.functional.softplus(input[0], beta=beta)


