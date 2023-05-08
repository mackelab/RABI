import numpy as np
from torch import nn
import torch
import numpy as np

def process_group_size(x, num_units, axis=-1):
    size = list(x.size())
    num_channels = size[axis]

    if num_channels % num_units:
        raise ValueError('number of features({}) is not a '
                         'multiple of num_units({})'.format(num_channels, num_units))
    size[axis] = -1
    if axis == -1:
        size += [num_channels // num_units]
    else:
        size.insert(axis+1, num_channels // num_units)
    return size

def group_sort(x, num_units, axis=-1):
    size = process_group_size(x, num_units, axis)
    grouped_x = x.view(*size)
    sort_dim = axis if axis == -1 else axis + 1
    sorted_grouped_x, _ = grouped_x.sort(dim=sort_dim)
    sorted_x = sorted_grouped_x.view(*list(x.shape))

    return sorted_x

def check_group_sorted(x, num_units, axis=-1):
    size = process_group_size(x, num_units, axis)

    x_np = x.cpu().data.numpy()
    x_np = x_np.reshape(*size)
    axis = axis if axis == -1 else axis + 1
    x_np_diff = np.diff(x_np, axis=axis)

    # Return 1 iff all elements are increasing.
    if np.sum(x_np_diff < 0) > 0:
        return 0
    else:
        return 1

class GroupSort(nn.Module):

    def __init__(self, num_units, axis=-1):
        super(GroupSort, self).__init__()
        self.num_units = num_units
        self.axis = axis

    def forward(self, x):
        group_sorted = group_sort(x, self.num_units, self.axis)
        assert check_group_sorted(group_sorted, self.num_units, axis=self.axis) == 1, "GroupSort failed. "

        return group_sorted

    def extra_repr(self):
        return 'num_groups: {}'.format(self.num_units)

class PartialNonlinearity(nn.Module):
    def __init__(self, nonlinearities, split_lengths):
        super().__init__()
        self.nonlinearities = nonlinearities
        self.split_lengths = split_lengths
        if len(self.nonlinearities) != len(self.split_lengths):
            raise ValueError("For each split you must specify a nonlinearity.")

    def forward(self, x):
        splits = torch.split(x, self.split_lengths, dim=-1)
        outs = []
        for f, split in zip(self.nonlinearities, splits):
            outs.append(f(split))
        return torch.concat(outs, dim=-1)


class Maxout(nn.Module):
    def __init__(self, num_units, axis=-1):
        super(Maxout, self).__init__()
        self.num_units = num_units
        self.axis = axis

    def forward(self, x):
        return maxout(x, self.num_units, self.axis)

    def extra_repr(self):
        return 'num_units: {}'.format(self.num_units)


class MaxMin(nn.Module):

    def __init__(self, num_units, axis=-1):
        super(MaxMin, self).__init__()
        self.num_units = num_units
        self.axis = axis

    def forward(self, x):
        maxes = maxout(x, self.num_units, self.axis)
        mins = minout(x, self.num_units, self.axis)
        maxmin = torch.cat((maxes, mins), dim=1)
        return maxmin

    def extra_repr(self):
        return 'num_units: {}'.format(self.num_units)


def process_maxmin_size(x, num_units, axis=-1):
    size = list(x.size())
    num_channels = size[axis]

    if num_channels % num_units:
        raise ValueError('number of features({}) is not a '
                         'multiple of num_units({})'.format(num_channels, num_units))
    size[axis] = -1
    if axis == -1:
        size += [num_channels // num_units]
    else:
        size.insert(axis+1, num_channels // num_units)
    return size


def maxout(x, num_units, axis=-1):
    size = process_maxmin_size(x, num_units, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.max(x.view(*size), sort_dim)[0]


def minout(x, num_units, axis=-1):
    size = process_maxmin_size(x, num_units, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.min(x.view(*size), sort_dim)[0]
