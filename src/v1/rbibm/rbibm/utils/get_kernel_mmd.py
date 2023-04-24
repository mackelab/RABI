from rbi.loss.kernels import RBFKernel, MultiKernel, MultiDimRBFKernel
from rbi.loss.mmd import MMDsquaredOptimalKernel, MMDsquared
import torch

def get_kernel(model, xs, kernel_name="rbf", **kwargs):
    if kernel_name == "rbf":
        return RBFKernel(**kwargs)
    elif kernel_name == "rbf_family":
        return rbf_kernel_family(**kwargs)
    elif kernel_name == "rbf_multidim_family":
        q = model(xs[0])
        dim = q.event_shape[-1]
        return rbf_multi_dim_family(dim, **kwargs)
    elif kernel_name == "preselected_rbf_multidim_family":
        N = xs.shape[0]
        q = model(xs[:N//2])
        p = model(xs[N//2:])
        dim = q.event_shape[-1]
        ks = rbf_multi_dim_family(dim,**kwargs)
        return preselect_rbf_family(p,q, ks)
    elif kernel_name == "preselected_rbf_family":
        N = xs.shape[0]
        q = model(xs[:N//2])
        p = model(xs[N//2:])
        ks = rbf_kernel_family(**kwargs)
        return preselect_rbf_family(p,q, ks)
    


def rbf_kernel_family(min_sigma=0.5, max_simga=2., min_l = 0.1, max_l = 2., num_kernels=5):
    ls = torch.linspace(min_l, max_l, num_kernels)
    sigmas = torch.linspace(min_sigma, max_simga, num_kernels)
    kernels = [RBFKernel(sigma=s, l = l) for s in sigmas for l in ls]
    k = MultiKernel(kernels)
    return k

def rbf_multi_dim_family(d, min_sigma=0.5, max_simga=1., min_l = 0.1, max_l = 2., num_kernels=5):
    ls = torch.linspace(min_l, max_l, 3)
    sigmas = torch.linspace(min_sigma, max_simga, 3)
    dimensionwise_sigmas = [min_sigma + max_simga * torch.rand(d) for _ in range(num_kernels)]
    dimensionwise_ls = [min_l + max_l * torch.rand(d) for _ in range(num_kernels)]

    normal_rbfs = [RBFKernel(sigma=s, l = l) for s in sigmas for l in ls]
    multi_dim_rbfs = [MultiDimRBFKernel(sigma=s, l = l, reduction="prod") for s in dimensionwise_sigmas for l in dimensionwise_ls]
    multi_dim_rbfs_sum = [MultiDimRBFKernel(sigma=s, l = l, reduction="sum") for s in dimensionwise_sigmas for l in dimensionwise_ls]

    k = MultiKernel(normal_rbfs + multi_dim_rbfs + multi_dim_rbfs_sum)
    return k
    
def preselect_rbf_family(p, q, family):
    k = MMDsquaredOptimalKernel(family, mc_samples=100)
    samples1 = k.get_samples(p)
    samples2 = k.get_samples(q)

    h = k.compute_h(samples1, samples2)
    mmd = k.compute_mmd(h)
    betas = torch.zeros(mmd.shape[0], mmd.shape[-1])
    mask = mmd < 0.0
    all_neg_mask = mask.all(-1)
    some_pos_mask = ~all_neg_mask

    mmd_var = k.compute_mmd_var(h)

    # All negative case...
    if all_neg_mask.any():
        criterium = torch.argmax(
            mmd[all_neg_mask] / mmd_var[all_neg_mask].sqrt(), keepdim=True, dim=-1
        )
        betas[all_neg_mask, criterium] = 1.
    if some_pos_mask.any():
        with torch.no_grad():
            betas = k.get_optimal_kernel_coefficients(mmd[some_pos_mask], h[:,some_pos_mask], lam=1e-2)
            # We normalize to 1, thats not done in the paper but whatever
            betas /= betas.sum(-1, keepdim=True)
            betas = betas.float()
    betas = betas.mean(0).clamp(min=1e-20, max=1.)
    return sum([float(beta)*k for beta,k in zip(betas, family.kernels)])
