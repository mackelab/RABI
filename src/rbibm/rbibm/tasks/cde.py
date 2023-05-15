from torch.distributions import Distribution
import torch
from rbibm.tasks.base import CDETask
from typing import Optional, Callable




class SnakeCDE(CDETask):
    def __init__(self, input_dim: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

    def get_generator(self, device: str = "cpu") -> Callable:
        def generate_snake_dist(N:int):
            X = torch.rand((N, int(self.input_dim)), device=device) * 10 - 5
            Y = 0.1 * X**3 + 5 * torch.sin(
                X + 0.2 * torch.randn_like(X, device=device)
            ) * torch.randn_like(X, device=device)

            return X, Y

        return generate_snake_dist

class MixtureDist(CDETask):
    def __init__(self, input_dim: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

    def get_generator(self, device: str = "cpu") -> Callable:
        def generate_mixture(N=1000):
            X = torch.rand((N, ))*20-5

            f1 = lambda x:  - 5*torch.exp(-0.1*(x-2)**2) - 0.1*x 
            f2 = lambda x: 5*torch.exp(-0.1*(x-5)**2) + 0.1*x

            mixing = torch.distributions.Categorical(torch.tensor([[0.65,0.35]]).repeat(len(X),1))
            components = torch.distributions.Independent(torch.distributions.Normal(torch.vstack([f1(X), f2(X)]).T, torch.ones_like(torch.vstack([X,X])).T), 0)
            p = torch.distributions.MixtureSameFamily(mixing, components)

            Y = p.sample()
            return X.reshape(-1, 1), Y.reshape(-1,1)

        return generate_mixture
    
    def get_ground_truth(self):
        def closed_form(X):
            f1 = lambda x:  - 5*torch.exp(-0.1*(x-2)**2) - 0.1*x 
            f2 = lambda x: 5*torch.exp(-0.1*(x-5)**2) + 0.1*x

            mixing = torch.distributions.Categorical(torch.tensor([[0.65,0.35]]).repeat(len(X),1))
            components = torch.distributions.Independent(torch.distributions.Normal(torch.vstack([f1(X), f2(X)]).T, torch.ones_like(torch.vstack([X,X])).T), 0)
            p = torch.distributions.MixtureSameFamily(mixing, components)

            return p 
        return closed_form
