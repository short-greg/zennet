import numpy as np
import torch
from torch import nn
from abc import ABC, abstractmethod
from .modules import Objective
from .optimizers import InputOptim, Scorer, ThetaOptim, get_theta, update_theta
from .utils import expand_dim0


def get_best(value: torch.Tensor, evaluations: torch.Tensor, maximize: bool=True):

    if evaluations.ndim == 2:
        if maximize:
            idx = torch.argmax(evaluations, 0, True)
        else:
            idx = torch.argmin(evaluations, 0, True)
        idx.unsqueeze_(2)
        idx = idx.repeat(1, 1, value.shape[2])
        result = value.gather(0, idx)
        return result[0]

    if maximize:
        idx = torch.argmax(evaluations)
    else:
        idx = torch.argmin(evaluations)
    return value[idx]


class HillClimbMixin(ABC):

    @abstractmethod
    def _perturb(self, value):
        pass

    @abstractmethod
    def _report(self, evalutions):
        pass


class HillClimbDiscreteMixin(HillClimbMixin):

    def __init__(self):
        pass

    def _perturb(self, value):
        pass


class HillClimbPerturber(ABC):

    @abstractmethod
    def __call__(self, value: torch.Tensor, evaluations) -> torch.Tensor:
        pass


class SimpleHillClimbPerturber(HillClimbPerturber):

    def __call__(self, value: torch.Tensor, evaluations) -> torch.Tensor:
        v = value.unsqueeze(0) + torch.randn(1, *value.size()) * 1e-1
        return torch.cat(
            [value[None], v]
        )


class GaussianHillClimbPerturber(HillClimbPerturber):

    def __init__(self, s_mean: int=-2, s_std=1, k: int=1, momentum: float=None, maximize: bool=True):
        self.s_mean = s_mean
        self.s_std = s_std
        self.k = k  
        self.maximize = maximize
        self._i = 0
        self._keep_s = 100
        # self._update_after = update_after
        self._mean_offset = 0.

    def __call__(self, value: torch.Tensor, evaluations) -> torch.Tensor:

        self._s = torch.randn(self.k)  * self.s_std + self.s_mean
        s = (10 ** self._s).view(-1, *([1] * value.dim())).repeat(1, *value.size())
        y = torch.randn(self.k, *value.size()) * s + value
        
        self._i += 1
        # if self._i % self._update_after == 0:
        #     self.s_mean -= 0.5
        #     # self._mean_offset -= 0.5
        return torch.cat([value.unsqueeze(0), y])


class BinaryHillClimbPerturber(HillClimbPerturber):

    def __init__(self, k: int=1, p: float=0.1, maximize: bool=False):
        self.p = p
        self.k = k  
        self.maximize = maximize

    def __call__(self, value: torch.Tensor, evaluations, dim: int=0) -> torch.Tensor:

        keep = (torch.rand(self.k, *value.shape) > self.p).type(value.dtype)
        values = value[None].repeat(self.k, *([1] * len(value.shape)))
        values = keep * values + (1 - keep) * (1 - values)
        return torch.cat([value[None], values])


class HillClimbSelector(ABC):

    @abstractmethod
    def __call__(self, cur: torch.Tensor, value: torch.Tensor, evaluations) -> torch.Tensor:
        pass


class SimpleHillClimbSelector(HillClimbSelector):

    def __init__(self, maximize: bool=False):
        self.maximize = maximize
    
    def __call__(self, cur: torch.Tensor, value: torch.Tensor, evaluations) -> torch.Tensor:
        return get_best(value, evaluations, self.maximize)


class GaussianHillClimbSelector(HillClimbSelector):

    def __init__(self, momentum: float=None, maximize: bool=True):

        self._momentum = momentum
        self.maximize = maximize
        self._diff = None
        self._x_updated = None
        self._step_evaluations = {}
        self._keep_s = True

    def __call__(self, cur: torch.Tensor, value: torch.Tensor, evaluations, dim: int=0) -> torch.Tensor:
        best = get_best(value, evaluations, self.maximize)
        if self._diff is not None and self._momentum is not None:
            self._diff = (1 - self._momentum) * (best - cur) + self._momentum * self._diff
            x_updated = cur + self._diff
        elif self._momentum is not None:
            self._diff = (best - cur)
            x_updated = cur + self._diff
        else:
            x_updated = best
        return x_updated


class HillClimbThetaOptim(ThetaOptim):

    def __init__(self, net: nn.Module, objective: Objective, perturber: HillClimbPerturber=None, selector: HillClimbSelector=None):
        super().__init__()
        self._objective = objective
        self._net = net
        self._perturber = perturber or SimpleHillClimbPerturber()
        self._selector = selector or SimpleHillClimbSelector()
    
    @property
    def theta(self):
        return get_theta(self._net)

    def step(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, scorer: Scorer=None):
        assert t is not None

        theta_base = self.theta
        theta = self._perturber(theta_base, self._evaluations)
        evaluations = []
        for theta_i in theta:
            update_theta(self._net, theta_i)
            y = self._net(x)
            if scorer:
                evaluations.append(scorer.assess(y).item())
            else:
                evaluations.append(self._objective(y, t).item())
        theta_best =  self._selector(theta_base, theta, torch.tensor(evaluations))
        update_theta(self._net, theta_best)
        self._evaluations = evaluations


class HillClimbInputOptim(InputOptim):

    def __init__(self, net: nn.Module, objective: Objective, perturber: HillClimbPerturber=None, selector: HillClimbSelector=None):
        super().__init__()
        self._objective = objective
        self._net = net
        self._perturber = perturber or SimpleHillClimbPerturber()
        self._selector = selector or SimpleHillClimbSelector()

    def step(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, scorer: Scorer=None) -> torch.Tensor:
        x_pool = self._perturber(x, self._evaluations)
        y = self._net(x_pool.view(-1, *x_pool.shape[2:]))
        # y = y.view(x_pool.shape[0], x_pool.shape[1], *y.shape[1:])
        # y = y.transpose(1, 0)
        if scorer:
            evaluations = scorer.assess(y)
        else:
            evaluations = self._objective.forward_multi(y, expand_dim0(t, x_pool.shape[0]))
        
        _evaluations = [e.item() for e in evaluations]
        self._evaluations = _evaluations
        # need to reshape the evaluations
        # make evaluatinos a tensor
        result = self._selector(
            x, x_pool, 
            evaluations.reshape(x_pool.shape[0], x_pool.shape[1])
        )
        return result
