from torch.distributions import Distribution
import torch
from rbibm.tasks.base import CDETask
from typing import Optional, Callable


class BinaryClassificationTask(CDETask):
    def __init__(self, input_dim: int = 2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 1

    def get_generator(self, device: str = "cpu") -> Callable:
        def generate_binary_toy_dataset(N: int):
            N_1 = int(N / 2)
            N_2 = N - N_1

            class_1 = (torch.randn((N_1, self.input_dim), device=device).T).T

            class_2 = torch.randn((N_2, self.input_dim), device=device)
            class_2 /= torch.linalg.norm(class_2, dim=1).unsqueeze(-1)
            class_2 = (
                class_2 * 4 + torch.randn((N_2, self.input_dim), device=device) * 0.9
            )

            data = torch.vstack([class_1, class_2])
            labels = torch.hstack(
                [torch.ones(N_1, device=device), torch.zeros(N_2, device=device)]
            )
            permutation = torch.randperm(N, device=device)

            return data[permutation, :], labels[permutation]

        return generate_binary_toy_dataset


class TernaryClassificationTask(CDETask):
    def __init__(self, input_dim: int = 2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 3

    def get_generator(self, device: str = "cpu") -> Callable:
        def generate_3class_toy_dataset(N: int):
            N_1 = int(N / 3)
            N_2 = int(N / 3)
            N_3 = N - N_1 - N_2

            class_1 = torch.randn((N_1, self.input_dim), device=device)

            class_2 = torch.randn((N_2, self.input_dim), device=device)
            class_2 /= torch.linalg.norm(class_2, dim=1).unsqueeze(-1)
            class_2 = (
                class_2 * 4 + torch.randn((N_2, self.input_dim), device=device) * 0.9
            )

            class_3 = torch.randn((N_3, self.input_dim), device=device)
            class_3 /= torch.linalg.norm(class_3, dim=1).unsqueeze(-1)
            class_3 = class_3 * 5.5 + torch.randn((N_3, self.input_dim)) * 0.1

            data = torch.vstack([class_1, class_2, class_3])
            labels = torch.hstack(
                [
                    torch.zeros(N_1, device=device),
                    torch.ones(N_2, device=device),
                    2 * torch.ones(N_3, device=device),
                ]
            )
            permutation = torch.randperm(N, device=device)
            return data[permutation, :], labels[permutation]

        return generate_3class_toy_dataset
