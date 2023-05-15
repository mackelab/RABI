from rbibm.tasks.base import InferenceTask
import torch

from pyro.distributions import ConditionalDistribution, Distribution


class PotentialBasedDistribution(Distribution):
    def __init__(self, potential_fn, x_o, theta_grid, proposal):
        self.potential_fn = potential_fn
        self.x_o = x_o
        self.theta_grid = theta_grid
        self.proposal = proposal

        grid = self.theta_grid.reshape(-1, 1, 1)
        pot = self.potential_fn(self.x_o, grid)
        Zs = torch.trapz(
            pot.exp(),
            self.theta_grid,
            dim=0,
        ).log()
        self.Zs = Zs
        print(Zs.shape)

        fx = (pot - Zs).reshape(-1, self.x_o.shape[0])
        gx = proposal.log_prob(self.theta_grid.reshape(-1, 1)).reshape(
            -1, self.x_o.shape[0]
        )
        logM = (fx - gx).max(dim=0).values
        self._M = logM.exp().clip(max=100.0)

    def sample(self, shape=torch.Size((1,))):
        # An easy rejection sampler....
        shape = torch.Size(shape)
        N = shape.numel()
        accepted_samples = [None for _ in range(self.x_o.shape.numel())]

        while any(
            [True if a_s is None else a_s.shape[0] < N for a_s in accepted_samples]
        ):
            proposed = self.proposal.sample((1000,))

            u = torch.rand((1000,))

            f_proposed = self.log_prob(proposed).exp()
            g_proposed = self.proposal.log_prob(proposed).exp().unsqueeze(-1)
            mask = u.unsqueeze(-1) < (f_proposed / (self._M.unsqueeze(0) * g_proposed))

            for i in range(len(accepted_samples)):
                if accepted_samples[i] is not None:
                    accepted_samples[i] = torch.vstack(
                        [accepted_samples[i], proposed[mask[:, i]]]
                    )
                else:
                    accepted_samples[i] = proposed[mask[:, i]]

        return torch.stack([a_s[:N].squeeze(-1).T for a_s in accepted_samples]).reshape(
            *shape, -1
        )

    def log_prob(self, thetas):
        return self.potential_fn(self.x_o, thetas.unsqueeze(-1)) - self.Zs


class SquarePosterior(ConditionalDistribution):
    def __init__(
        self,
        potential_fn,
        grid_size=1000,
        prior_mean=0.0,
        prior_scale=2.0,
        device="cpu",
    ):
        self.potential_fn = potential_fn
        self.theta_grid = torch.linspace(
            -4 * float(prior_scale), 4 * float(prior_scale), grid_size, device=device
        )
        self.proposal = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.as_tensor(prior_mean, device=device),
                torch.as_tensor(2 * prior_scale, device=device),
            ),
            1,
        )

    def condition(self, context):
        return PotentialBasedDistribution(
            self.potential_fn, context, self.theta_grid, self.proposal
        )


class SquareTask(InferenceTask):
    input_dim = 1
    output_dim = 1

    def __init__(self, prior_mean=0.0, prior_scale=2.0, likelihood_scale=1.0):
        self.prior_mean = torch.tensor([prior_mean])
        self.prior_scale = torch.tensor([prior_scale])
        self.likelihood_scale = torch.tensor([likelihood_scale])
        prior = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.tensor([self.prior_mean]), torch.tensor([self.prior_scale])
            ),
            1,
        )

        def likelihood_fn(theta):
            likelihood_scale = self.likelihood_scale.to(theta.device)
            return torch.distributions.Independent(
                torch.distributions.Normal(theta**2, likelihood_scale), 1
            )

        super().__init__(prior, likelihood_fn, None)

    def get_true_posterior(self, device="cpu"):

        return SquarePosterior(
            self.get_potential_fn(device=device),
            prior_mean=self.prior_mean,
            prior_scale=self.prior_scale,
            device=device,
        )
