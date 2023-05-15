from distutils.errors import LinkError
import torch  # type: ignore
from rbibm.tasks.base import InferenceTask
from pyro.distributions import ConditionalDistribution, Distribution # type: ignore


class GLRDistribution(Distribution):
    def __init__(self, theta, phi_x, likelihood_scale=0.1):
        self.theta = theta
        self.likelihood_scale = likelihood_scale
        self.phi_x = phi_x

    def sample(self):
        raise NotImplementedError("test")

    def log_prob(self, xys):

        if len(self.theta.shape) > 2:
            theta = self.theta.reshape(-1, self.theta.shape[-1])
        else:
            theta = self.theta

        batches_theta = theta.shape[0]
        batches_xys = xys.shape[0]

        if batches_theta > batches_xys:
            repeats = batches_theta // batches_xys
            xys = xys.repeat(repeats, 1, 1)
        elif batches_xys < batches_theta:
            repeats = batches_xys // batches_theta
            theta = theta.repeat(repeats, 1)

        # Differentiable with nans

        phi_x = self.phi_x(xys[..., :-1])
        phi_x = phi_x.nan_to_num()
        y = xys[..., -1]
        y_pred = torch.einsum("ijd, id -> ij", phi_x, theta)

        mask = torch.isfinite(y)

        p = torch.distributions.Normal(
            y_pred.reshape(y.shape)[mask], self.likelihood_scale, validate_args=False
        )
        log_probs = p.log_prob(y[mask])

        result = torch.zeros(y_pred.shape)
        result[mask] = log_probs

        return result.sum(1).reshape(*self.theta.shape[:-1])


class GLRLikelihood(ConditionalDistribution):
    def __init__(self, likelihood_scale, phi_x):
        super().__init__()
        self.likelihood_scale = likelihood_scale
        self.phi_x = phi_x

    def condition(self, context):
        return GLRDistribution(context, self.phi_x, likelihood_scale=self.likelihood_scale)


class GLRPosterior(ConditionalDistribution):
    def __init__(self, prior, likelihood_scale):
        super().__init__()
        self.prior = prior
        self.likelihood_scale = likelihood_scale

    def condition(self, context):
        if len(context.shape) == 2:
            context = context.unsqueeze(0)

        batch_shape = context.shape[0]
        means = []
        covs = []
        for i in range(batch_shape):
            phi_x = context[i, ..., :-1]
            y = context[i, ..., -1]

            mask = torch.isfinite(y)

            phi_x = phi_x[mask]
            y = y[mask]

            c = phi_x @ phi_x.T

            prior_mean = self.prior.mean
            prior_cov = torch.eye(prior_mean.shape[-1]) * self.prior.variance

            likelihood_cov = torch.eye(c.shape[-1]) * self.likelihood_scale**2

            c += likelihood_cov

            L_c = torch.cholesky(c)
            c_inv = torch.cholesky_inverse(L_c)

            post_m = prior_mean + prior_cov @ phi_x.T @ c_inv @ (y - phi_x @ prior_mean)
            post_cov = prior_cov - prior_cov @ phi_x.T @ c_inv @ phi_x @ prior_cov

            means.append(post_m.unsqueeze(0))
            covs.append(post_cov.unsqueeze(0))

        return torch.distributions.MultivariateNormal(
            torch.stack(means), torch.stack(covs)
        )


class GLRTaskVariableObservation(InferenceTask):

    single_observation = False

    def __init__(
        self,
        feature_mapping,
        min_N=1,
        max_N=10,
        x_bound=10,
        prior_mean=0.0,
        prior_scale=1.0,
        likelihood_scale=0.5,
    ):
        self.max_N = max_N
        self.min_N = min_N
        self.likelihood_scale = likelihood_scale
        self.x_bound = x_bound
        self.output_dim = feature_mapping(torch.randn((1, 1))).shape[-1]
        self.input_dim = self.output_dim + 1

        self.feature_mapping = feature_mapping

        prior = torch.distributions.Independent(
            torch.distributions.Normal(
                prior_mean * torch.ones(self.output_dim),
                prior_scale * torch.ones(self.output_dim),
            ),
            1,
        )

        def simulator(theta, x=None):

            old_shape = theta.shape

            theta = theta.reshape(-1, theta.shape[-1])

            if x is None:
                x = self.generate_datasets_x(theta.shape[0])

            phi_x = self.feature_mapping(x).reshape(*x.shape[:-1], self.output_dim)
            y = torch.einsum("ijd, id -> ij", phi_x, theta)
            y += torch.randn_like(y) * likelihood_scale
            y = y.reshape(x.shape)
            xy = torch.concat([x, y], dim=-1)
            return xy.reshape(*old_shape[:-1], *xy.shape[1:])

        def likelihood_fn(theta):
            p = GLRDistribution(theta, self.feature_mapping, self.likelihood_scale)
            return p

        super().__init__(prior, likelihood_fn, simulator)

    def get_true_posterior(self, device:str = "cpu"):
        return GLRPosterior(self.prior, self.likelihood_scale)

    def generate_datasets_x(self, N):
        if self.max_N > self.min_N:
            Ns = torch.randint(1, self.max_N, (N,))
            x = torch.rand((N, self.max_N, 1)) * 2 * self.x_bound - self.x_bound
            for i in range(N):
                x[i, Ns[i] :] = torch.nan
            return x
        else:
            return torch.rand((N, self.min_N, 1)) * 2 * self.x_bound - self.x_bound



class PolynomialRegressionTaskVariabelObservations(GLRTaskVariableObservation):
    def __init__(self, degree=3, **kwargs):
        def phi(x):
            phi_x = torch.stack([x**i for i in range(degree + 1)], dim=-1)
            mask = torch.isfinite(phi_x.sum(-1))
            phi_x[~mask] = torch.nan
            return phi_x

        super().__init__(phi, **kwargs)


class LinearRegressionTaskVariableObservations(PolynomialRegressionTaskVariabelObservations):
    def __init__(self, **kwargs):
        super().__init__(degree=1, **kwargs)


class RBFRegressionTaskVariableObservations(GLRTaskVariableObservation):
    def __init__(
        self, sigma=5.0, l=1.0, inducing_points=10, min_x=-10, max_x=10, **kwargs
    ):
        self.x_grid = torch.linspace(min_x, max_x, inducing_points).reshape(-1, 1)

        def phi(x):
            return torch.exp(-(sigma**2) * torch.cdist(x, self.x_grid) ** 2 / l)

        super().__init__(phi, **kwargs)


