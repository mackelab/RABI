

import matplotlib.pyplot as plt
from rbi.models.base import ParametricProbabilisticModel
import torch


def plot_2d_surface(net, x_min:float=-6., x_max:float=6., y_min:float=-6., y_max:float=6., steps:int=1000, cmap="viridis", figsize=(10,10), plt_context="neurips2022", plt_context_kwargs={}):
   
    with torch.no_grad():
        xx = torch.linspace(x_min,x_max, steps)
        yy = torch.linspace(y_min,y_max, steps)
        XX, YY = torch.meshgrid(xx,yy)

        if isinstance(net, ParametricProbabilisticModel):
            cs = net.net(torch.hstack([XX.reshape(-1,1), YY.reshape(-1,1)]))
        else:
            cs = net(torch.hstack([XX.reshape(-1,1), YY.reshape(-1,1)]))
        output_dim = cs.shape[-1]

        fig, axes = plt.subplots(1,output_dim,figsize=figsize,subplot_kw=dict(projection='3d'))
        if output_dim == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            c = cs[..., i].reshape(steps, steps)
            ax.plot_surface(XX,YY, c, cmap=cmap, linewidth=0, antialiased=False)
            ax.grid(False)
    return fig

def plot_2d_decision_boundary(net, x_min:float=-6., x_max:float=6., y_min:float=-6., y_max:float=6., steps:int=1000, soft=False, cmap="viridis", figsize=(10,10),plt_context="neurips2022", plt_context_kwargs={}):

    with torch.no_grad():
        xx = torch.linspace(x_min,x_max, steps)
        yy = torch.linspace(y_min,y_max, steps)
        XX, YY = torch.meshgrid(xx,yy)

        cs = net.net(torch.hstack([XX.reshape(-1,1), YY.reshape(-1,1)]))
        output_dim = cs.shape[-1]
        if not soft:
            colors = torch.argmax(cs, axis=-1)/output_dim
        else:
            raise NotImplementedError()
        
        fig = plt.figure(figsize=figsize)
        plt.imshow(colors.reshape(steps, steps), extent=[x_min, x_max, y_min, y_max], cmap = cmap, alpha=0.5)
        plt.axis("off")
        return fig


def plot_2d_surface_direction(net, origin, direction1, direction2, x_min:float=-5., x_max:float=5., y_min:float=-5, y_max:float=5., steps: int=1000, cmap="viridis", figsize=(50,10), z_lim_min=0., z_lim_max=1., plt_context="neurips2022", plt_context_kwargs={}):
    assert direction1.shape[0] == 1, "The direction should be a (1, input_dim) vector"
    assert len(direction1.shape) == 2, "The direction should be a (1, input_dim) vector"
    assert origin.shape == direction1.shape and origin.shape == direction2.shape
   
    with torch.no_grad():
        xx = torch.linspace(x_min,x_max, steps)
        yy = torch.linspace(y_min,y_max, steps)
        XX, YY = torch.meshgrid(xx,yy)

        scales_x, scales_y = XX.reshape(-1,1), YY.reshape(-1,1)
        inputs = origin + direction1*scales_x + direction2*scales_y
        cs = net.net(inputs)
        output_dim = cs.shape[-1]
        fig, axes = plt.subplots(1,output_dim,figsize=figsize,subplot_kw=dict(projection='3d'))

        for i, ax in enumerate(axes):
            c = cs[..., i].reshape(steps, steps)
            ax.plot_surface(XX,YY, c, cmap=cmap, linewidth=0, antialiased=False, vmin=z_lim_min, vmax=z_lim_max)
            ax.grid(False)
            ax.set_zlim(z_lim_min,z_lim_max)

    return fig, inputs, cs