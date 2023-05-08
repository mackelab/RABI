from rbi.loss import MMDsquared, MMDsquaredOptimalKernel
from rbi.loss.kernels import MultiKernel, RBFKernel, LinearKernel, RationalQuadraticKernel, LaplaceKernel
from rbi.loss.mmd import RBFKernel, select_kernel_combination, select_kernel, select_bandwith_by_median_distance

from rbibm.utils.get_task import get_task
import torch
from rbibm.utils.utils_data import query, query_main, remove_entry_by_id, query_rob_metric, get_model_by_idx, get_adversarial_examples_by_id, load_posterior_samples_by_id, get_model_by_id


def compute_ground_truth_mmds(name, task, defense="None", model_name="maf", N_train=100000, attack="L2PGDAttack", metric_rob="ReverseKLRobMetric", q_l = 0.25, q_u = 0.75, kernel=RBFKernel(l=0.1) + LaplaceKernel(l=10.), **kwargs):

    df = query(name, task=task, defense=defense, model_name=model_name, N_train=N_train,attack=attack,metric_rob=metric_rob, **kwargs)
    id_adv = df.id_adversarial.iloc[0]
    id = df.id.iloc[0]
    q = get_model_by_id(name, id)
    xs, _ , xs_tilde = get_adversarial_examples_by_id(name, id_adv)

    k = MMDsquared(kernel, reduction=None)

    epsilons = [0.1,0.2,0.3,0.5,1.,2.]

    dxxq = []
    dxtildex_tildeq = []

    dxxtilde = []
    dxxtilde_q = []

    dxtrue_xtilde = []
    for eps in epsilons:
        df = query(name, task=task, eps=eps, defense=defense, model_name=model_name, N_train=N_train,attack=attack,metric_rob=metric_rob, **kwargs)
        id_adv = df.id_adversarial.iloc[0]
        samples_xs = torch.stack(list(load_posterior_samples_by_id(name, id_adv)["xs_rand"].values()),1)
        samples_xs_tilde = torch.stack(list(load_posterior_samples_by_id(name, id_adv)["xs_tilde_rand"].values()),1)
        xs, _ , xs_tilde = get_adversarial_examples_by_id(name, id_adv)
        index = list(load_posterior_samples_by_id(name, id_adv)["xs_tilde_rand"].keys())
        q = get_model_by_id(name, id)
        q_tilde = q(xs_tilde[index])
        q_clean = q(xs[index])

        # Model samples
        samples_q_tilde = q_tilde.sample((1000,))
        samples_q = q_clean.sample((1000,))
        
        # Compute metric
        d_x_xtilde = k(samples_xs, samples_xs_tilde)
        dxxtilde.append((d_x_xtilde.mean(), d_x_xtilde.quantile(q_l), d_x_xtilde.quantile(q_u)))
        d_x_xtilde_q = k(samples_q, samples_q_tilde)
        dxxtilde_q.append((d_x_xtilde_q.mean(), d_x_xtilde_q.quantile(q_l), d_x_xtilde_q.quantile(q_u)))

        d_x_true_xtilde_q = k(samples_xs, samples_q_tilde)
        dxtrue_xtilde.append((d_x_true_xtilde_q.mean(),d_x_true_xtilde_q.quantile(q_l),d_x_true_xtilde_q.quantile(q_u)))

        d_x_x_q = k(samples_xs, samples_q)
        dxxq.append((d_x_x_q.mean(), d_x_x_q.quantile(q_l), d_x_x_q.quantile(q_u)))
        d_x_tilde_x_tilde_q = k(samples_xs_tilde, samples_q_tilde)
        dxtildex_tildeq.append((d_x_tilde_x_tilde_q.mean(), d_x_tilde_x_tilde_q.quantile(q_l),d_x_tilde_x_tilde_q.quantile(q_u)))

    return dxxq, dxtildex_tildeq, dxxtilde, dxxtilde_q, dxtrue_xtilde