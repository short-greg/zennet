import torch
from .base import ScoreReverse
import typing
import torch.nn as nn
import scipy.optimize as sciopt
from .utils import freshen


class MSELossReverse(ScoreReverse):

    def __init__(self):
        super().__init__(nn.MSELoss)
    
    def reverse(self, x: torch.Tensor, t: torch.Tensor, lr: float=1e-2) -> torch.Tensor:
        # if x - t == positive, dx = negative. vice versa
        return x + torch.sqrt(lr * self._loss(x, t)) * -torch.sign(x - t)


class GeneralLossReverse(ScoreReverse):

    def __init__(self, loss: typing.Type[nn.Module], maxiter: int=20):
        super().__init__(loss)

        # TODO: Use this
        self._maxiter = maxiter

    def reverse(self, x: torch.Tensor, t: torch.Tensor, lr: float=1e-2) -> torch.Tensor:
        # if x - t == positive, dx = negative. vice versa
        # 1) convert to numpy
        # 2) 

        # set up bounds
        # if x > t.. make t a lowerbound and x an upper bound
        # otherwise vice versa
        # ensure that 0 <= lr <= 1
        # use nelder mead
        # maxiter = 5pip in

        # This is the basics of how it should work... Need to test

        shape = x.shape
        x = x.flatten()
        t = t.flatten()
        t = t.detach().cpu()

        target_loss = (1 - lr) * self._loss(x, t)

        def objective(pt):
            # print('Shapes: ',  pt.shape,t.shape )
            result = ((target_loss - self._loss(torch.tensor(pt), t)) ** 2)
            print(result)
            return result.mean().item()
            
            # may need to compute the jacobian
            # pt = torch.tensor(pt)
            # pt = freshen(pt)
            # self._loss(pt, t).mean().backward()
            # grad = pt.grad.detach().cpu().numpy()
            # print(grad)
            # return grad

        lb = torch.min(x, t).detach().cpu().numpy()
        ub = torch.max(x, t).detach().cpu().numpy()
        
        bounds = sciopt.Bounds(lb, ub) 
        # find out how to set the maximium number of iterations
        x_prime = sciopt.minimize(objective, x.detach().cpu().numpy(), method='Powell', bounds=bounds).x
        
        return torch.tensor(x_prime, dtype=x.dtype, device=x.device).view(shape)


class GradLossReverse(ScoreReverse):
    """Approximate the new target using a gradient. 
    """

    def __init__(self):
        super().__init__(nn.MSELoss)
    
    def reverse(self, x: torch.Tensor, t: torch.Tensor, lr: float=1e-2) -> torch.Tensor:
        # if x - t == positive, dx = negative. vice versa
        x = freshen(x)
        self._loss.forward(x, t).mean().backward()
        return x - x.grad * lr
