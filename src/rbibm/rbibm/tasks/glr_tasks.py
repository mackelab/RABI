from distutils.errors import LinkError
import torch # type: ignore
from rbibm.tasks.base import InferenceTask
from pyro.distributions import ConditionalDistribution # type: ignore

from torch.distributions import Distribution # type: ignore
from rbibm.tasks.gaussian_linear import GaussianLinearPosterior

from typing import Callable, List

class DeltaDistribution(Distribution):
    """ Point mass distribution"""
    arg_constraints = {}
    def __init__(self, loc) -> None:
        self.loc = loc 
        super().__init__(batch_shape = self.loc.shape)
       

    def sample(self, shape=()):
        shape = torch.Size(shape)
        loc = self.loc.unsqueeze(0)
        ndim = self.loc.ndim
        samples = loc.repeat([shape.numel()] + [1 for n in range(ndim)] )
        return samples.reshape(shape + self.batch_shape + self.event_shape)

    def log_prob(self, x):
        x,loc = torch.broadcast_tensors(x, self.loc)
        return torch.isclose(x, loc).float().log()


class MultiIndependent(Distribution):
    """" Takes a list of distributions and concatenates them. """
    arg_constraints = {}
    def __init__(self, ps: List[Distribution]) -> None:
        self.ps = ps 
        super().__init__()

    def sample(self, shape=()):
        shape = torch.Size(shape)
        samples = []
        for p in self.ps:
            samples.append(p.sample(shape))
        return torch.concat(samples, dim=-1)

    def log_prob(self, x):
        log_probs = 0
        cum_event_shape = 0
        for p in self.ps:
            event_shape = p.event_shape[0]
            log_probs += p.log_prob(x[...,cum_event_shape: cum_event_shape + event_shape ])
            cum_event_shape += event_shape
        return log_probs





class GLRDistribution(MultiIndependent):
    def __init__(self, ps: List[Distribution], theta, feature_mapping, likelihood_scale) -> None:
        self.theta = theta
        self.feature_mapping = feature_mapping
        self.likelihood_scale = likelihood_scale
        super().__init__(ps)

    def recompute(self, xy):

        if self.theta.shape[:-1] == xy.shape[:-1]:
            theta = self.theta.unsqueeze(0)
        else:
            theta = self.theta
    
        x,y = xy.split(xy.shape[-1]//2, dim=-1)
       
        phi_x = self.feature_mapping(x.unsqueeze(-1))
        y = torch.einsum("bij, lbj -> lbi", phi_x, theta)
        x = x.repeat(y.shape[0],1,1).squeeze()
        y = y.squeeze()

        p_x = torch.distributions.Independent(DeltaDistribution(x), 1)
        p_y = torch.distributions.Independent(torch.distributions.Normal(y, self.likelihood_scale), 1)
        self.__init__([p_x, p_y], self.theta, self.feature_mapping, self.likelihood_scale)

    def log_prob(self, x):
        self.recompute(x)
        return super().log_prob(x)



        


class GLRPosterior():
    """ GLR posterior """
    def __init__(self, prior,feature_mapping, likelihood_scale=0.5, eps=1e-8) -> None:
        self.prior = prior 
        self.likelihood_scale = likelihood_scale
        self.feature_mapping = feature_mapping
        self.eps =eps
        
    def condition(self, context):
        batch_shape = context.shape[:-1]
        context = context.reshape(-1, context.shape[-1])

        N = context.shape[-1] // 2
        x,y = context.split(N, dim=-1)
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        phi_x = self.feature_mapping(x)
        d = phi_x.shape[-1]
       
        prior_mean = self.prior.mean  # type: ignore
        if hasattr(self.prior, "covariance_matrix"):
            prior_cov = self.prior.covariance_matrix.to(context.device) # type: ignore
        else:
            prior_cov = torch.eye(
                prior_mean.shape[-1], device=context.device
            ) * self.prior.variance.to(  # type: ignore
                context.device
            )

        likelihood_cov = (torch.eye(N, device=context.device)*self.likelihood_scale**2).unsqueeze(0)

        gramm_matrix = phi_x@prior_cov@phi_x.transpose(dim0=-2, dim1=-1) + likelihood_cov
        L_c = torch.linalg.cholesky(
            gramm_matrix + torch.eye(N, device=context.device) * self.eps
        )
        prior_prediction = phi_x @ prior_mean
        res = y.squeeze(-1) - prior_prediction
        
        adjusted_res = torch.cholesky_solve(
            res.unsqueeze(-1), L_c.unsqueeze(0)
        ).squeeze(0).squeeze(-1)
        adjusted_cov = torch.cholesky_solve(phi_x@prior_cov, L_c.unsqueeze(0)).squeeze(0).squeeze(-1)


        p = prior_cov  @ phi_x.transpose(-2,-1)

        post_m = prior_mean + torch.einsum("bij, bj -> bi",p, adjusted_res)
        post_cov = prior_cov - prior_cov @ phi_x.transpose(-2, -1) @ adjusted_cov


        return torch.distributions.MultivariateNormal(
            post_m.reshape(batch_shape + (d,)),
            post_cov.reshape(batch_shape + (d,d)),
        )




class GLRTask(InferenceTask):

    single_observation = True

    def __init__(
        self,
        feature_mapping: Callable,
        N: int = 5,
        x_bound: float=10.,
        prior_mean: float=0.0,
        prior_scale: float=1.0,
        likelihood_scale: float=0.5,
    ):
        """ A Gaussian generalized linear regression task.

        Args:
            feature_mapping (Callable): Feature mapping
            N (int, optional): Number of observations. Defaults to 10.
            x_bound (float, optional): Bound on x axis for domain. Defaults to 10..
            prior_mean (float, optional): Prior mean. Defaults to 0.0.
            prior_scale (float, optional): Prior scale. Defaults to 1.0.
            likelihood_scale (float, optional): Likelihood scale. Defaults to 0.5.

        Returns:
            _type_: _description_
        """
        self.N = N
        self.likelihood_scale = likelihood_scale
        self.x_bound = x_bound
        self.output_dim = feature_mapping(torch.randn((1, 1))).shape[-1]
        self.input_dim = 2*self.N

        self.feature_mapping = feature_mapping

        prior = torch.distributions.Independent(
            torch.distributions.Normal(
                prior_mean * torch.ones(self.output_dim),
                prior_scale * torch.ones(self.output_dim),
            ),
            1,
        )


        def likelihood_fn(theta, x=None):
            old_shape = theta.shape[:-1]

            theta = theta.reshape(-1, theta.shape[-1])

            if x is None and float(theta.sum()):
                x = torch.rand(theta.shape[0], self.N,1, device=theta.device)*2*self.x_bound - self.x_bound

            phi_x = self.feature_mapping(x)
            y = torch.einsum("bnd, bd -> bn", phi_x, theta)
            x = x.reshape(old_shape + (self.N,))
            y = y.reshape(old_shape + (self.N,))

            p_x = torch.distributions.Independent(DeltaDistribution(x), 1)
            p_y = torch.distributions.Independent(torch.distributions.Normal(y, self.likelihood_scale), 1)
            return GLRDistribution([p_x, p_y], theta.reshape(old_shape + (-1,)), self.feature_mapping, self.likelihood_scale)

        super().__init__(prior, likelihood_fn, None)

    def sample_functions(self, theta, N_grid=1000):
        x = torch.linspace(-self.x_bound, self.x_bound, N_grid).reshape(1,-1,1).repeat(theta.shape[0], 1,1)
        phi_x = self.feature_mapping(x)
        y = torch.einsum("bnd, bd -> bn", phi_x, theta)
        return x.squeeze(-1), y

    def get_true_posterior(self, device:str="cpu"):
        return GLRPosterior(self.prior, self.feature_mapping, likelihood_scale= self.likelihood_scale)


class RBFRegressionTask(GLRTask):
    def __init__(
        self, sigma=2.0, l=0.3, inducing_points=6, x_bound=2, **kwargs
    ):
        self.x_grid = torch.linspace(-x_bound+ 0.5, x_bound - 0.5, inducing_points).reshape(-1, 1)
    
        def phi(x):
            return (sigma**2) *torch.exp(- torch.cdist(x, self.x_grid) ** 2 /(2*l))

        super().__init__(phi, x_bound=x_bound,**kwargs)