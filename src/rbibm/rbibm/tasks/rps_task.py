import torch
from rbibm.tasks.base import InferenceTask



def simulator_rps(param, iters=100, N_grid=50):

    param = param.reshape(-1, 6)
    device = param.device
    batch_shape = param.shape[0]

    param[:, :3] = param[:, :3].softmax(-1)
    param[:, 3:] = param[:, 3:].sigmoid()

    grid = torch.randint(0, 4, (batch_shape, 1, N_grid, N_grid), device=device).float()
    A_prior = param[:, 0]
    B_prior = param[:, 1]
    C_prior = param[:, 2]

    death_rate_B_C = param[:, 3]
    death_rate_A = param[:, 4]
    toxicity_C_to_A = param[:, 5]

    with torch.no_grad():
        l = torch.nn.Conv2d(1, 1, 3, bias=False, padding=1).to(device)
        l._parameters["weight"].data = torch.ones(1, 1, 3, 3, device=device)

        for i in range(iters):

            empty = (grid == 0.0).float()
            As = (grid == 1.0).float()
            Bs = (grid == 2.0).float()
            Cs = (grid == 3.0).float()

            really_empty = l(empty)
            A_neighbor_count = l(As)
            B_neighbor_count = l(Bs)
            C_neighbor_count = l(Cs)

            A_neighbor_count_weighted = (
                A_neighbor_count * A_prior.reshape(-1, 1, 1, 1) + 1e-6
            )
            B_neighbor_count_weighted = (
                B_neighbor_count * B_prior.reshape(-1, 1, 1, 1) + 1e-6
            )
            C_neighbor_count_weighted = (
                C_neighbor_count * C_prior.reshape(-1, 1, 1, 1) + 1e-6
            )

            counts = torch.concat(
                [
                    A_neighbor_count_weighted.unsqueeze(0),
                    B_neighbor_count_weighted.unsqueeze(0),
                    C_neighbor_count_weighted.unsqueeze(0),
                ]
            )

            # For empty cells
            win_prob_by_counts = counts / counts.sum(0, keepdim=True)
            new_grid = (
                torch.distributions.Categorical(
                    win_prob_by_counts.T, validate_args=False
                )
                .sample()
                .T.float()
                .to(device)
            )
            grid[empty.bool()] = new_grid[empty.bool()] + 1.0
            grid[really_empty == 9.0] = 0.0

            # Dead B and C
            dead_B = Bs * torch.bernoulli(
                death_rate_B_C.reshape(-1, 1, 1, 1) * torch.ones_like(Bs, device=device)
            )
            grid[dead_B.bool()] = 0.0

            dead_C = Cs * torch.bernoulli(
                death_rate_B_C.reshape(-1, 1, 1, 1) * torch.ones_like(Cs, device=device)
            )
            grid[dead_C.bool()] = 0.0

            # Dead
            death_prob_A_by_C = (
                toxicity_C_to_A.reshape(-1, 1, 1, 1) * C_neighbor_count / 9
            )
            death_prob_A = 0.5 * (death_rate_A.reshape(-1, 1, 1, 1) + death_prob_A_by_C)
            dead_A = As * torch.bernoulli(
                death_prob_A * torch.ones_like(death_prob_A, device=device)
            )
            grid[dead_A.bool()] = 0.0
    return grid


def simulate_in_batches(theta, simulator, batch_shape=10000):
    N = theta.shape[0]
    iters = max((N // batch_shape), 1)
    xs = []
    for i in range(iters):
        x = simulator(theta[i * batch_shape : (i + 1) * batch_shape])
        xs.append(x)

    return torch.vstack(xs)


class RPSTask(InferenceTask):
    def __init__(self, N_grid=50, T=100):

        self.T = T
        self.input_dim = N_grid**2
        self.output_dim = 6
        prior = torch.distributions.Independent(
                torch.distributions.Normal(
                    torch.tensor([0.0, 0.0, 0.0, -3.0, -3.0, 3.0]),
                    torch.tensor([0.1, 0.1, 0.1, 1, 1, 1]),
                ),
            1,
        )

        def simulator(theta):
            return simulator_rps(theta, iters=T, N_grid=N_grid).reshape(
                *theta.shape[:-1], self.input_dim
            )

        super().__init__(prior, None, lambda x: simulate_in_batches(x, simulator))
