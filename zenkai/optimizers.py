from functools import partial
from re import X
import typing
import sklearn
import torch.nn as nn
from abc import ABC, abstractmethod, abstractproperty
import torch
from .modules import Objective, SklearnModule
from torch.nn import utils as nn_utils
import numpy as np
from . import utils
from .base import Evaluation, ThetaOptim, InputOptim


class SklearnThetaOptim(ThetaOptim):

    def __init__(self, sklearn_machine, partial_fit: bool=False):
        self._partial_fit = partial_fit
        self._machine = sklearn_machine
    
    @property
    def theta(self):
        return self._machine

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, y: torch.Tensor=None) -> Evaluation:
        if self._partial_fit:
            self._machine.partial_fit(x, t)
        else:
            self._machine.fit(x, t)
        return objective(self._machine(x), t)


class NRepeatInputOptim(InputOptim):

    def __init__(self, optim: InputOptim, n: int):
        super().__init__()
        assert n > 0, f'Argument n must be greater than 0 not {n}'
        self.optim = optim
        self._n = n

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, y: torch.Tensor) -> typing.Tuple[torch.Tensor, Evaluation]:
        for _ in range(self.n):
            x, evaluation = self.optim.step(x, t, objective, y)
            y = None
        return x, evaluation


class NRepeatThetaOptim(ThetaOptim):

    def __init__(self, optim: ThetaOptim, n: int):
        super().__init__()
        self.optim = optim
        self.n = n

    @property
    def theta(self):
        return self.optim.theta

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, y: torch.Tensor=None) -> Evaluation:
        for _ in range(self.n):
            evaluation = self.optim.step(x, t, objective, y)
            y = None
        return evaluation


class GradThetaOptim(ThetaOptim):

    def __init__(
        self, net: nn.Module, objective: Objective, 
        optim: typing.Type[torch.optim.Optimizer]=torch.optim.AdamW
    ):
        super().__init__()
        self._net = net
        self.criterion = objective
        self.optim = optim(net.parameters())

    @property
    def theta(self):
        return utils.get_parameters(self._net)
    
    # override the evaluate method to expand on the scoring
    def evaluate(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, y: torch.Tensor=None) -> Evaluation:        
        loss = objective(x, t, y)
        return loss

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, y: torch.Tensor=None) -> Evaluation:
        self.optim.zero_grad()
        evaluation = self.evaluate(x, t, objective, y)
        evaluation.regularized.mean().backward()
        self.optim.step()
        return evaluation


class InputUpdater(ABC):
    
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass 


class BasicInputUpdater(ABC):
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.grad 


class GradInputOptim(InputOptim):

    def __init__(self, net: nn.Module, input_updater: InputUpdater=None, skip_eval: bool=False):
        super().__init__()
        self._net = net
        self.input_updater = input_updater or BasicInputUpdater()
        self.skip_eval = skip_eval
    
    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, y: torch.Tensor) -> typing.Tuple[torch.Tensor, Evaluation]:
        if self.skip_eval and x.grad is not None:
            x = self.input_updater(x)
            return x, Evaluation()
        
        x = utils.freshen(x)
        evaluation = objective.forward(x, t)
        evaluation.regularized.mean().backward()
        x = self.input_updater(x)
        return x, evaluation


class NullThetaOptim(ThetaOptim):

    def __init__(self):
        super().__init__()
        # self.objective = objective
        # self.f = f

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, y: torch.Tensor=None) -> Evaluation:
        
        evaluation = objective(x, t, y)
        return evaluation
