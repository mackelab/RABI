from typing import Tuple, Iterable, final, Union

import cvxpy as cp

import torch
from torch.distributions import Distribution
from rbi.loss.kernels import *
from rbi.loss.base import EvalLoss




class MMDsquared(EvalLoss):
    def __init__(
        self,
        kernel: Kernel = RBFKernel(),
        mc_samples: int = 200,
        reduction: str = "mean",
        **kwargs
    ):
        """Computes the squared MMD statistic.

        Args:
            kernel (Kernel, optional): Kernel used in computation. Defaults to RBFKernel".
            mc_samples (int, optional): Samples to use. Defaults to 200.
            reduction (str, optional): Reduction type. Defaults to "mean".
        """
        super().__init__(reduction)
        self.kernel = kernel
        self.mc_samples = mc_samples

    def _loss(self, output: Union[Distribution, Tensor], target: Union[Distribution, Tensor]) -> Tensor:
        
        samples1 = self.get_samples(output)
        samples2 = self.get_samples(target)

        h = self.compute_h(samples1, samples2)
        mmd = self.compute_mmd(h)
        return mmd

    def get_samples(self, p: Union[Distribution, Tensor]) -> Tensor:
        """Return samples from the distribution

        Args:
            p (Distribution): Distribution

        Returns:
            Tensor: mc_samples samples form the distribution.
        """
        if isinstance(p, Distribution):
            if p.has_rsample:
                samples = p.rsample((self.mc_samples,))  # type: ignore
            else:
                samples = p.sample((self.mc_samples,))  # type: ignore
        else:
            samples = p.clone()

        return samples

    def compute_h(self, samples1: Tensor, samples2: Tensor) -> Tensor:
        """Linear time h-statistic for MMD.

        Args:
            samples1 (Tensor): Samples from first distribution.
            samples2 (Tensor): Samples form second distribution.

        Returns:
            Tensor: h-statistic
        """

        x_p1 = samples1[::2]
        x_p2 = samples1[1::2]
        x_q1 = samples2[::2]
        x_q2 = samples2[1::2]

        h = (
            self.kernel(x_p2, x_p1)
            + self.kernel(x_q2, x_q1)
            - self.kernel(x_p2, x_q1)
            - self.kernel(x_p1, x_q2)
        )

        return h

    def compute_mmd(self, h: Tensor) -> Tensor:
        """Computes the MMD given a h-statistic

        Args:
            h (Tensor): h-statistic

        Returns:
            Tensor: MMD per input
        """
        return h.mean(0)

    def compute_mmd_var(self, h: Tensor) -> Tensor:
        """An linear time estimate of the standard deviation of the MMD estimate from the true mmd.

        Args:
            h (Tensor): h-statistic

        Raises:
            ValueError: Unexpected shape

        Returns:
            Tensor: Standard deviation
        """
        h_even = h[::2]
        h_uneven = h[1::2]
        h_even = h_even[: h_uneven.shape[0]]
        h_uneven = h_uneven[: h_even.shape[0]]
        h_var = h_even - h_uneven
        mmd_var = 4 * torch.mean(h_var**2, dim=0)
        return mmd_var


class KernelTwoSampleTest(MMDsquared):
    """Computes an p-value for a the test:
    H_0: p == q
    H_1: p != q
    """

    def _loss(self, output: Distribution, target: Distribution) -> Tensor:

        samples1 = self.get_samples(output)
        samples2 = self.get_samples(target)

        h = self.compute_h(samples1, samples2)
        mmd = self.compute_mmd(h)
        mmd_var = self.compute_mmd_var(h)

        p_vals = self.compute_pvalue(mmd, mmd_var)
        return p_vals

    def compute_pvalue(self, mmd: Tensor, mmd_var: Tensor) -> Tensor:
        """Computes an p-value for a the test:
            H_0: p == q
            H_1: p != q

        Args:
            mmd (Tensor): _description_
            mmd_var (Tensor): _description_

        Returns:
            Tensor: _description_
        """
        mmd_std = mmd_var.sqrt()
        n_sqrt = math.sqrt(self.mc_samples)
        p_0 = torch.distributions.Normal(0, 1)

        return 1 - p_0.cdf((mmd * n_sqrt) / (mmd_std * math.sqrt(2)))


class MMDsquaredKernelSelection(MMDsquared):
    def _loss(self, output: Distribution, target: Distribution) -> Tensor:

        samples1 = self.get_samples(output)
        samples2 = self.get_samples(target)

        h = self.compute_h(samples1, samples2)
        mmd = self.compute_mmd(h)

        if isinstance(self.kernel, MultiKernel):
            mmd_var = self.compute_mmd_var(h)
            mmd = self.get_optimal_kernel(mmd, mmd_var)

        return mmd

    def get_optimal_kernel(self, mmd: Tensor, mmd_var: Tensor) -> Tensor:
        criterium = torch.argmax(mmd / mmd_var.sqrt(), keepdim=True)
        mmd = torch.gather(mmd, -1, criterium)
        return mmd


class MMDsquaredOptimalKernel(MMDsquared):

    def _loss(self, output: Distribution, target: Distribution) -> Tensor:


        samples1 = self.get_samples(output)
        samples2 = self.get_samples(target)

        h = self.compute_h(samples1, samples2)
        mmd = self.compute_mmd(h)

        if isinstance(self.kernel, MultiKernel):
            final_mmd = torch.zeros(mmd.shape[:-1])
            mask = mmd < 0.0
            all_neg_mask = mask.all(-1)
            some_pos_mask = ~all_neg_mask

            mmd_var = self.compute_mmd_var(h)

            # All negative case...
            if all_neg_mask.any():
                criterium = torch.argmax(
                    mmd[all_neg_mask] / mmd_var[all_neg_mask].sqrt(), keepdim=True, dim=-1
                )
                final_mmd[all_neg_mask] = torch.gather(mmd[all_neg_mask], -1, criterium).squeeze()
            
            if some_pos_mask.any():
                with torch.no_grad():
                    betas = self.get_optimal_kernel_coefficients(mmd[some_pos_mask], h[:,some_pos_mask])
                    # We normalize to 1, thats not done in the paper but whatever
                    betas /= betas.sum(-1, keepdim=True)
                    betas = betas.float()
                
                final_mmd[some_pos_mask] = torch.sum(betas*mmd[some_pos_mask],-1)
        else:
            final_mmd = mmd 
        
        return final_mmd

        

    def get_optimal_kernel_coefficients(
        self, mmd: Tensor, h: Tensor, lam=1e-2
    ) -> Tensor:

        num_kernels = h.shape[-1]
        batches = mmd.shape[:-1]
        num_el = batches.numel()
        mmd = mmd.reshape(-1, num_kernels)

        Q = (
            4 * (h.unsqueeze(-1) * h.unsqueeze(-2)).mean(0)
            + torch.eye(num_kernels) * lam
        )



        betas = [cp.Variable(num_kernels) for _ in range(num_el)]  # type: ignore
        constraints = [b >= 0.0 for b in betas]  # type: ignore
        constraints2 = [e @ b == 1.0 for b, e in zip(betas, mmd)]

        prob = cp.Problem(  # type: ignore
            cp.Minimize(sum([cp.quad_form(b, cp.atoms.affine.wraps.psd_wrap(P)) for b, P in zip(betas, Q)])),  # type: ignore
            constraints2 + constraints,  # type: ignore
        )
        try:
            prob.solve()
            collect_optimal_betas = torch.stack([torch.as_tensor(b.value) for b in betas])
            return collect_optimal_betas.reshape(*batches, num_kernels)
        except Exception as e:
            print(str(e))
            return torch.ones(*batches, num_kernels)


def select_kernel_combination(kernels, samples1, samples2, lam=1e-2):
    k = MultiKernel(kernels)

    m = MMDsquared(k)
    h = m.compute_h(samples1, samples2)
    mmd = m.compute_mmd(h)

    num_kernels = h.shape[-1]
    batches = mmd.shape[:-1]
    num_el = batches.numel()
    mmd = mmd.reshape(-1, num_kernels)

    Q = (
        4 * (h.unsqueeze(-1) * h.unsqueeze(-2)).mean(0)
        + torch.eye(num_kernels) * lam
    )



    betas = [cp.Variable(num_kernels) for _ in range(num_el)]  # type: ignore
    constraints = [b >= 0.0 for b in betas]  # type: ignore
    constraints2 = [e @ b == 1.0 for b, e in zip(betas, mmd)]

    prob = cp.Problem(  # type: ignore
        cp.Minimize(sum([cp.quad_form(b, cp.atoms.affine.wraps.psd_wrap(P)) for b, P in zip(betas, Q)])),  # type: ignore
        constraints2 + constraints,  # type: ignore
    )
    try:
        prob.solve()
        collect_optimal_betas = torch.stack([torch.as_tensor(b.value) for b in betas])
        betas =  collect_optimal_betas
    except Exception as e:
        print(str(e))
        betas =  torch.ones(mmd.shape[0], num_kernels)

    beta = betas.mean(0)
    print(beta)
    final_kernel = kernels[0] * beta[0]
    for i in range(1, len(beta)):
        final_kernel = final_kernel + kernels[i] * beta[i]

    return final_kernel

    

def select_kernel(kernels, samples1, samples2):
    k = MultiKernel(kernels)

    m = MMDsquared(k)
    h = m.compute_h(samples1, samples2)
    mmd = m.compute_mmd(h)
    mmd_var = m.compute_mmd_var(h)
    criterium = torch.argmax(mmd / mmd_var.sqrt(), dim=-1, keepdim=True)
    kernel_idx = int(torch.mode(criterium.flatten()).values)
    return kernels[kernel_idx]

def select_bandwith_by_median_distance(samples1, samples2):
    dist = torch.cdist(samples1, samples2)
    median = float(dist.median())
    return RBFKernel(l = median)

