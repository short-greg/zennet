import torch

def expand_dim0(x: torch.Tensor, k: int, reshape: bool=True):
    y = x[None].repeat(k, *([1] * len(x.size())))
    if reshape:
        return y.view(y.shape[0] * y.shape[1], *y.shape[2:])
    return y
