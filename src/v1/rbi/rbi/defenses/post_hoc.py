from multiprocessing.sharedctypes import Value
from rbi.defenses.base import PostHocDefense
from rbi.loss import NLLLoss
import torch
from torch.nn import Module

from torch.distributions import MixtureSameFamily
from typing import Callable
from rbi.utils.distributions import SIRDistirbution
from rbi.models import IndependentGaussianNet

from functools import partial

from copy import deepcopy

# TODO: In construction ....

def sir_forward_hook(task, K, module, input, output):
    return SIRDistirbution(output, task.get_potential_fn(), input[0], K =K)

class SIRPostHocAdjustment(PostHocDefense):
    def __init__(self, model: Module, task, K:int = 10) -> None:
        super().__init__(model)
        self.K = K
        self.task = task

    def activate(self, **kwargs):
        self.model.register_forward_hook(partial(sir_forward_hook, self.task, self.K))


    def deactivate(self, **kwargs):
        self.model._forward_hooks.clear()



class ModifiedForward:
    def __init__(self, mix_distribution, old_generator, attack_model) -> None:
        self.old_generator = old_generator
        self.mix_distribution = mix_distribution
        self.attack_model = attack_model

    def __call__(self, x_tilde):
        x = self.attack_model(x_tilde).sample(
            (self.mix_distribution.logits.shape.numel(),)
        )
        p = self.old_generator(x)
        components = torch.distributions.Independent(p, 1)
        return torch.distributions.MixtureSameFamily(self.mix_distribution, components)


class AdversarialDenoisePostHoc(PostHocDefense):
    def __init__(
        self,
        model,
        loss_fn,
        attack,
        train_loader,
        attack_model=IndependentGaussianNet,
        attack_model_kwargs={},
        max_eps=5.0,
        iters=10,
        components=100,
    ):
        self.attack_model = attack_model(
            model.input_dim, model.input_dim, **attack_model_kwargs
        )
        self.attack = attack
        self.train_loader = train_loader
        self.max_eps = max_eps
        self.iters = iters
        self.components = components
        super().__init__(model, loss_fn)

    def activate(self, **kwargs):
        if self.model.training:
            raise ValueError(
                "Did you trained your model? This defense requires the model to be trained. Please swith into evaluation mode if you finished training i.e. using model.eval()"
            )
        else:
            self.adversarial_denoise()

        mix_distribution = torch.distributions.Categorical(
            logits=torch.zeros(self.components)
        )
        new_forward = ModifiedForward(mix_distribution, self.model, self.attack_model)
        # TODO

    def deactivate(self, **kwargs):
        return super().deactivate(**kwargs)

    def adversarial_denoise(self):
        print(f"Training adverarial denoiser")
        optim = torch.optim.Adam(self.attack_model.parameters(), lr=1e-3)
        loss_fn = NLLLoss(self.attack_model)
        for i in range(self.iters):
            l = 0
            n = 0
            for X, _ in self.train_loader:
                self.attack.eps = float(torch.rand(1)) * self.max_eps
                x_tilde = self.attack.perturb(X)
                optim.zero_grad()
                loss = loss_fn(x_tilde, X)
                loss.backward()
                optim.step()
                l += loss.detach()
                n += 1

            print(f"Loss (Epoch {i}): {float(l/n)}")

    def construct_robust_model(self):
        pass
        # p_rob = MixtureSameFamily(
        #     torch.distributions.Categorical(logits=torch.zeros(1000)),
        #     torch.distributions.Independent(
        #         net(net_attack(x_tilde).sample((1000,))), 1
        #     ),
        # )
