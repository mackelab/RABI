from rbibm.tasks.base import InferenceTask
import os
from torch import Tensor

from pyro.distributions import ConditionalDistribution
from rbi.utils.mcmc import MCMC 
from rbi.utils.distributions import SIRDistribution, MCMCDistribution
from rbi.utils.mcmc_kernels import LatentSliceKernel,LearnableIndependentKernel,KernelScheduler


import torch
from torch import nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import numpy as np

class VAE(nn.Module):
    def __init__(self,input_shape, latent_dim=3,hidden_dim=500):
        super(VAE,self).__init__()
        self.input_shape = input_shape
        self.input_dim = int(input_shape[0]*input_shape[1])
        self.fc_e = nn.Sequential(nn.Linear(self.input_dim,hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.ReLU())
        self.fc_mean = nn.Linear(hidden_dim,latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim,latent_dim)
        self.fc_d = nn.Sequential(nn.Linear(latent_dim,hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim,self.input_dim),
                                  nn.Sigmoid())
            
    def encoder(self,x_in):
        x = self.fc_e(x_in.view(-1,self.input_dim))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar
    
    def decoder(self,z):
        x_out = self.fc_d(z)
        return x_out.view(-1,1,*self.input_shape)
    
    def sample_normal(self,mean,logvar):
        sd = torch.exp(logvar*0.5)
        e = torch.tensor(torch.randn(sd.size()))
        z = e.mul(sd).add_(mean)
        return z
    
    def forward(self,x_in):
        z_mean, z_logvar = self.encoder(x_in)
        z = self.sample_normal(z_mean,z_logvar)
        x_out = self.decoder(z)
        return x_out, z_mean, z_logvar

def nll_gauss(mean, std, x, axis=None):
    """Gaussian log likelihood"""
    var = std**2
    const = torch.log(torch.tensor(2*np.pi))*torch.ones(x.shape)
    ll = const - 0.5*torch.log(var) - 0.5*torch.div((mean-x)**2,var)
    return -torch.sum(ll) if axis is None else -torch.sum(ll, axis=axis) 

def elbo(x, z_mu, z_logvar, out_mean, sigma=0.5,beta=1.0):
    out_std = torch.ones_like(out_mean) *sigma
    elbo_KL = beta*(-0.5*torch.sum(1+ z_logvar - (z_mu**2) - torch.exp(z_logvar)))
    elbo_nll = nll_gauss(out_mean, out_std, x)
    return (elbo_nll + elbo_KL)/x.size(0)


def train_vae(latent_dim, hidden_dims=500, epochs=60):
 

    trainloader = DataLoader(MNIST(root=".mnist",train=True,download=True,transform=transforms.ToTensor()),batch_size=256,shuffle=True)

    torch.manual_seed(0)
    np.random.seed(0)

    model = VAE((28,28), latent_dim=latent_dim,hidden_dim=hidden_dims)

    optimizer = torch.optim.Adam(model.parameters())
    
    for _ in range(epochs):
        for images,_ in trainloader:
            x_in = images
            optimizer.zero_grad()
            x_out_mean, z_mu, z_logvar = model(x_in)
            loss = elbo(x_in, z_mu, z_logvar, x_out_mean)
            loss.backward() 
            optimizer.step()

    return model

class VAEPosterior(ConditionalDistribution):
    def __init__(self, prior, potential_fn) -> None:
        super().__init__()
        self.prior = prior
        self.potential_fn = potential_fn



    def condition(self, context:Tensor):

        k1 = LatentSliceKernel(self.potential_fn, context=context, step_size=0.01)
        k2 = LearnableIndependentKernel()
        k = KernelScheduler([k1,k2, k1], [0, 100, 150])

        proposal = SIRDistribution(self.prior, self.potential_fn, context=context, K= 10)
        mcmc = MCMC(k , self.potential_fn, proposal, context=context ,warmup_steps=200, thinning=1, num_chains=100, device=context.device)

        return MCMCDistribution(mcmc)


class VAETask(InferenceTask):

    """VAE-like inverse problem. The generator is pre-trained and fixed."""

    likelihood_scale = torch.ones(1) * 0.05
    input_dim = 784

    def __init__(self, latent_dim: int = 5):
        """This is a generative model with a pretrained VAE

        Args:
            latent_dim (int, optional): Latent dimension. Defaults to 2. Can also be 5...

        Raises:
            ValueError: If no pretrained model is available.


        """
        prior = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(latent_dim), torch.ones(latent_dim)),
            1,
        )
        self.output_dim = latent_dim

        try:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            self.decoder = torch.load(
                dir_path + os.sep + f"vae_decoder_{latent_dim}latentdims.pkl"
            )
        except:
            model = train_vae(latent_dim)
            self.decoder = model.fc_d
            torch.save(self.decoder, dir_path + os.sep + f"vae_decoder_{latent_dim}latentdims.pkl")
            

        def likelihood_fn(theta):
            decoder = self.decoder.to(theta.device)
            x = decoder(theta)
            return torch.distributions.Independent(
                torch.distributions.Normal(
                    x, VAETask.likelihood_scale.to(theta.device)
                ),
                1,
            )

        super().__init__(prior, likelihood_fn, None)

    def get_true_posterior(self, device: str = "cpu"):
        return VAEPosterior(self.get_prior(device), self.get_potential_fn(device))
