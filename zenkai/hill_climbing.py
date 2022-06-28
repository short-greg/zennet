import numpy as np
import torch
from torch import nn
from abc import ABC, abstractmethod
from .modules import Objective
from .optimizers import InputOptim, Scorer, ThetaOptim, get_theta, update_theta



def get_best(x: torch.Tensor, evaluations, maximize: bool=True):
    if maximize:
        idx = torch.argmax(evaluations)
    else:
        idx = torch.argmin(evaluations)
    return x[idx]


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


class HillClimbSelector(ABC):

    @abstractmethod
    def __call__(self, value: torch.Tensor, evaluations) -> torch.Tensor:
        pass


class SimpleHillClimbSelector(HillClimbSelector):

    def __call__(self, value: torch.Tensor, evaluations) -> torch.Tensor:
        return value[np.argmax(evaluations)]


class GaussianHillClimbSelector(HillClimbSelector):

    def __init__(self, momentum: float=None, maximize: bool=True):

        self._momentum = momentum
        self.maximize = maximize

        self._diff = None
        self._x_updated = None
        self._s = None
        self._step_evaluations = {}
        self._keep_s = True

    def __call__(self, value: torch.Tensor, evaluations) -> torch.Tensor:
        self._step_evaluations = {
            k: v for i, (k, v) in enumerate(
                sorted(self._step_evaluations.items(), 
                key=lambda item: item[1], reverse=self.maximize)
            ) if i < self._keep_s
        }
        self._step_evaluations.update(dict(zip(self._s.tolist(), evaluations[1:].tolist())))

        best = get_best(value, evaluations, self.maximize)
        if self._diff is not None and self._momentum is not None:
            self._diff = (1 - self._momentum) * (best - value) + self._momentum * self._diff
            x_updated = value + self._diff
        elif self._momentum is not None:
            self._diff = (best - value)
            x_updated = value + self._diff
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
        theta = self._perturber(self.theta, self._evaluations)
        evaluations = []
        for theta_i in theta:
            update_theta(self._net, theta_i)
            y = self._net(x)
            if scorer:
                evaluations.append(scorer.assess(y).item())
            else:
                evaluations.append(self._objective(y, t).item())
        
        update_theta(self._net, self._selector(theta, evaluations))
        self._evaluations = evaluations


class HillClimbInputOptim(InputOptim):

    def __init__(self, net: nn.Module, objective: Objective, perturber: HillClimbPerturber=None, selector: HillClimbSelector=None):
        super().__init__()
        self._objective = objective
        self._net = net
        self._perturber = perturber or SimpleHillClimbPerturber()
        self._selector = selector or SimpleHillClimbSelector()

    def step(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, scorer: Scorer=None) -> torch.Tensor:
        x = self._perturber(x, self._evaluations)
        y = self._net(x.view(-1, *x.shape[2:]))
        y = y.view(x.shape[0], x.shape[1], *y.shape[1:])
        if scorer:
            evaluations = scorer.assess(y)
        else:
            evaluations = self._objective.forward_multi(y, t)
        evaluations = [e.item() for e in evaluations]
        
        self._evaluations = evaluations
        return self._selector(x, evaluations)
