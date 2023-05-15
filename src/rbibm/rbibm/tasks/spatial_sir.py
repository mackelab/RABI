import torch


from rbibm.tasks.base import InferenceTask


def simulate_sir(theta, shape=(30, 30), T=100, test_accuracy=0.95):
    # Extract the simulation parameters.
    batch_shape = theta.shape[:-1]
    theta = theta.reshape(-1, theta.shape[-1])
    device = theta.device
    beta = theta[:, 0]  # Infection rate
    gamma = theta[:, 1]  # Recovery rate
    # Allocate the data grids.
    infected = torch.zeros((theta.shape[0], 1) + shape, device=device)
    recovered = torch.zeros((theta.shape[0], 1) + shape, device=device)
    # Convolution
    l = torch.nn.Conv2d(1, 1, 3, bias=False, padding=1).to(device)
    l._parameters["weight"].data = torch.ones(1, 1, 3, 3, device=device)
    # Seed the grid with the initial infections.

    infected = torch.bernoulli(infected, p=4 / (shape[0] * shape[1])).to(device)
    # Derrive the maximum number of simulation steps.

    susceptible = ((1 - recovered) * (1 - infected)).float()
    for _ in range(T):
        if infected.sum() == 0:
            break
        # Infection
        potential = l(infected)
        potential *= susceptible
        potential = potential * beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) / 8
        next_infected = ((potential > torch.rand(shape, device=device)) + infected) * (1 - recovered)
        next_infected = next_infected >= 1
        # Recover
        potential = infected * gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        next_recovered = (potential > torch.rand(shape, device=device)) + recovered
        next_recovered = next_recovered >= 1
        # Next parameters
        recovered = next_recovered.float()
        infected = next_infected.float()
        susceptible = ((1 - recovered) * (1 - infected)).float()
    # Test all subjects
    testing_outcome = torch.distributions.Beta(infected + (1-test_accuracy), 1-infected + (1-test_accuracy)).sample()

    return testing_outcome.reshape(batch_shape + (int(shape[0] * shape[1]),))


class SpatialSIRTask(InferenceTask):
    def __init__(self, N_grid=40, T=30):

        self.T = T
        self.N_grid = N_grid
        self.input_dim = N_grid**2
        self.output_dim = 2
        prior = torch.distributions.Independent(
            torch.distributions.LogNormal(torch.zeros(1), 0.5*torch.ones(2)), 1
        )

        def simulator(theta):
            return simulate_sir(theta, T=self.T, shape=(self.N_grid, self.N_grid))

        super().__init__(prior, None, simulator)

    def get_simulator(self, batch_size: int = 100, device: str = "cpu"):
        return super().get_simulator(batch_size, device)
