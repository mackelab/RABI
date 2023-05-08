


import torch 
from torch import Tensor
from rbibm.tasks.base import CDETask, InferenceTask, Task


def run_true_posterior_samples(task: InferenceTask, x: Tensor, x_tilde: Tensor, num_samples_top:int = 10000, num_samples_rand:int = 1000, top_xs=10, rand_xs:int=100, device:str= "cpu"):

    try:
        posterior = task.get_true_posterior(device)
    except:
        return {}, {}
    
    x_selected = x[:top_xs]
    x_tilde_selected = x_tilde[:top_xs]

    p_x = posterior.condition(x_selected.to(device))
    p_x_tilde = posterior.condition(x_tilde_selected.to(device))

    print("Sampling")
    samples  =p_x.sample((num_samples_top,)).cpu()
    del p_x 
    samples_tilde = p_x_tilde.sample((num_samples_top,)).cpu()
    del p_x_tilde

    samples_dict_top = {}
    samples_tilde_dict_top = {}
    for i in range(top_xs):
        samples_dict_top[i] = samples[:, i, :]
        samples_tilde_dict_top[i] = samples_tilde[:, i, :]

    index = torch.randperm(x.shape[0])[:rand_xs]
    x_rand = x[index]
    x_tilde_rand = x_tilde[index]

    print("Sampling rand")
    iters = rand_xs // 50 + 1
    samples_r = []
    samples_tilde_r =[]
    for i in range(iters):
        if i*50 < rand_xs:
            p_x_rand = posterior.condition(x_rand[i * 50: (i+1)*50].to(device))
            p_x_tilde_rand = posterior.condition(x_tilde_rand[i * 50: (i+1)*50].to(device))

            samples_rand  =p_x_rand.sample((num_samples_rand,)).cpu()
            samples_tilde_rand = p_x_tilde_rand.sample((num_samples_rand,)).cpu()
            samples_r.append(samples_rand)
            samples_tilde_r.append(samples_tilde_rand)

            del p_x_rand 
            del p_x_tilde_rand

    samples_r = torch.concat(samples_r, 1)
    samples_tilde_r = torch.concat(samples_tilde_r, 1)

    samples_dict_rand = {}
    samples_tilde_dict_rand = {}
    for i in range(rand_xs):
        samples_dict_rand[int(index[i])] = samples_r[:, i, :]
        samples_tilde_dict_rand[int(index[i])] = samples_tilde_r[:, i, :]

    return samples_dict_top, samples_tilde_dict_top, samples_dict_rand,samples_tilde_dict_rand
    


