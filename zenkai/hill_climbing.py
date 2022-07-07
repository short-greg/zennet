import typing
import torch
from torch import nn
from abc import ABC, abstractmethod
from .utils import expand_dim0
from .machinery import TorchNN
from .base import Objective, Evaluation, InputOptim, ThetaOptim, get_best



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
    def __call__(self, cur: torch.Tensor, value: torch.Tensor, evaluation: Evaluation) -> typing.Tuple[torch.Tensor, Evaluation]:
        pass


class SimpleHillClimbSelector(HillClimbSelector):

    def __init__(self, maximize: bool=False):
        self.maximize = maximize
    
    def __call__(self, cur: torch.Tensor, value: torch.Tensor, evaluation: Evaluation) -> typing.Tuple[torch.Tensor, Evaluation]:
        return get_best(value, evaluation, self.maximize), evaluation


class GaussianHillClimbSelector(HillClimbSelector):

    def __init__(self, momentum: float=None, maximize: bool=True):

        self._momentum = momentum
        self.maximize = maximize
        self._diff = None
        self._x_updated = None
        self._step_evaluations = {}
        self._keep_s = True

    def __call__(self, cur: torch.Tensor, value: torch.Tensor, evaluation: Evaluation) -> typing.Tuple[torch.Tensor, Evaluation]:
        best, evaluation = get_best(value, evaluation, self.maximize)
        if self._diff is not None and self._momentum is not None:
            self._diff = (1 - self._momentum) * (best - cur) + self._momentum * self._diff
            x_updated = cur + self._diff
        elif self._momentum is not None:
            self._diff = (best - cur)
            x_updated = cur + self._diff
        else:
            x_updated = best
        return x_updated, evaluation


class HillClimbThetaOptim(ThetaOptim):

    def __init__(self, machine: TorchNN, perturber: HillClimbPerturber=None, selector: HillClimbSelector=None):
        super().__init__()
        self._machine = machine
        self._perturber = perturber or SimpleHillClimbPerturber()
        self._selector = selector or SimpleHillClimbSelector()

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, y: torch.Tensor=None) -> Evaluation:
        assert t is not None

        theta_base = self._machine.theta
        theta = self._perturber(theta_base, self._evaluations)
        evaluations = []
        for theta_i in theta:
            self._machine.theta = theta_i
            evaluations.append(objective(x, t))

        evaluation = Evaluation.from_list(evaluation)
        theta_best, evaluation =  self._selector(theta_base, theta, evaluation)
        
        self._machine.theta = theta_best
        return evaluation


class HillClimbInputOptim(InputOptim):

    def __init__(self, net: nn.Module, perturber: HillClimbPerturber=None, selector: HillClimbSelector=None):
        super().__init__()
        self._net = net
        self._perturber = perturber or SimpleHillClimbPerturber()
        self._selector = selector or SimpleHillClimbSelector()

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, y: torch.Tensor) -> typing.Tuple[torch.Tensor, Evaluation]:
        x_pool = self._perturber(x, self._evaluations)
        y = self._net(x_pool.view(-1, *x_pool.shape[2:]))
        # y = y.view(x_pool.shape[0], x_pool.shape[1], *y.shape[1:])
        # y = y.transpose(1, 0)
        
        evaluation = objective.forward(y, expand_dim0(t, x_pool.shape[0]), True)
        # regularized = [e for e in evaluation.regularized]

        return self._selector(
            x, x_pool, evaluation
            # evaluation.regularized.reshape(x_pool.shape[0], x_pool.shape[1])
        )
