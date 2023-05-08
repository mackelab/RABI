import torch



def eval_function_batched_sequential(func, *args, batch_size=1000, dim=0, device="cpu"):
    batched_x = list(
        zip(*[torch.split(x, batch_size, dim=dim) for x in args if x is not None])
    )

    ys = []
    for arg in batched_x:
        ys.append(func(*[a.to(device) for a in arg]))
    ys = torch.vstack(ys)

    return ys


def get_x_theta_from_loader(data_loader):
    xs = []
    thetas = []

    for x, theta in data_loader:
        xs.append(x)
        thetas.append(theta)

    return torch.vstack(xs), torch.vstack(thetas)


# May also implement multiprocessing...

# from ray.util.multiprocessing import Pool

# def eval_function_batched_mp(func, x, batch_size=1000, dim=0):
#     pool = Pool()
#     batched_x = torch.split(x, batch_size, dim=dim)
#     ys = pool.map(func, batched_x)
#     pool.close()
#     return torch.vstack(ys)

# def init_ray():
#     num_cpus = psutil.cpu_count(logical=False)
#     if not ray.is_initialized():
#         ray.init(num_cpus=num_cpus)


# def wrap_remote_fn(func):

#     @ray.remote
#     def f(*args, **kwargs):
#         return func(*args, **kwargs)

#     return f
