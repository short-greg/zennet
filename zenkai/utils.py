import typing
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn as nn


def batch_flatten(x: torch.Tensor):
    return x.view(x.size(0), -1)


def expand_dim0(x: torch.Tensor, k: int, reshape: bool=True):
    y = x[None].repeat(k, *([1] * len(x.size())))
    if reshape:
        return y.view(y.shape[0] * y.shape[1], *y.shape[2:])
    return y


def freshen(x: torch.Tensor, requires_grad: bool=True):
    x = x.detach().requires_grad_(requires_grad)
    x.retain_grad()
    return x


def set_parameters(parameters: torch.Tensor, net: nn.Module):
    vector_to_parameters(parameters, net.parameters())


def get_parameters(net: nn.Module):
    return parameters_to_vector(net.parameters())
    

def to_float(x: typing.List[torch.Tensor]):
    return list(map(lambda xi: xi.mean().item(), x))
