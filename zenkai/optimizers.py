from functools import partial
import typing
import sklearn
import torch.nn as nn
from abc import ABC, abstractmethod, abstractproperty
import torch
from .modules import Objective, SklearnModule, Scorer
from torch.nn import utils as nn_utils
import numpy as np
from . import utils


from .modules import score
from .base import ThetaOptim, InputOptim, Scorer


# def update_theta(module, theta):
#     nn_utils.vector_to_parameters(theta, module.parameters())


# def get_theta(module):
#     return nn_utils.parameters_to_vector(module.parameters())




class SklearnThetaOptim(ThetaOptim):

    def __init__(self, sklearn_machine, objective: Objective, partial_fit: bool=False):
        self._evaluations = None
        self._partial_fit = partial_fit
        self._machine = sklearn_machine
        self.objective = objective
    
    @property
    def theta(self):
        return self._machine

    def step(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, scorer: Scorer=None):
        if self._partial_fit:
            self._machine.partial_fit(x, t)
        else:
            self._machine.fit(x, t)
        self._evaluations = [self.objective(self._machine(x), t)]


class NRepeatInputOptim(InputOptim):

    def __init__(self, optim: InputOptim, n: int):
        super().__init__()
        self.optim = optim
        self.n = n

    def step(self, x, t, y: torch.Tensor=None, scorer: Scorer=None) -> torch.Tensor:
        evaluations = []
        for i in range(self.n):
            x = self.optim.step(x, t, y, scorer=scorer)
            evaluations.append(self.optim.evaluations)
            # y is unknown after first iteration
            y = None
        self._evaluations = evaluations
        return x


class NRepeatThetaOptim(ThetaOptim):

    def __init__(self, optim: ThetaOptim, n: int):
        super().__init__()
        self.optim = optim
        self.n = n

    @property
    def theta(self):
        return self.optim.theta

    def step(self, x, t, y: torch.Tensor=None, scorer: Scorer=None) -> torch.Tensor:
        evaluations = []
        for i in range(self.n):
            self.optim.step(x, t, y, scorer)
            evaluations.append(self.optim.evaluations)
            # y is unknown after first iteration
            y = None
        self._evaluations = evaluations


class GradThetaOptim(ThetaOptim):

    def __init__(
        self, net: nn.Module, objective: Objective, 
        optim: typing.Type[torch.optim.Optimizer]=torch.optim.AdamW
    ):
        super().__init__()
        self._net = net
        self.criterion = objective
        self.optim = optim(self._net.parameters())

    @property
    def theta(self):
        return utils.get_parameters(self._net)
    
    # override the evaluate method to expand on the scoring
    def evaluate(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, scorer: Scorer=None):
        if y is None:
            y = self._net(x)
        loss = score(y, t, self.criterion, scorer)
        self._evaluations = [loss.item()]
        return loss

    def step(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, scorer: Scorer=None):
        self.optim.zero_grad()
        loss = self.evaluate(x, t, scorer)
        loss.mean().backward()
        self.optim.step()


class InputUpdater(ABC):

    def updater(self, x: torch.Tensor):
        pass


class GradInputOptim(InputOptim):

    def __init__(self, net: nn.Module, objective: Objective, regularizer=None, input_updater: InputUpdater=None, skip_eval: bool=False):
        super().__init__()
        self._net = net
        self.objective = objective
        self.regularizer = regularizer
        self.input_updater = input_updater
        self.skip_eval = skip_eval
    
    def _update(self, x: torch.Tensor):
        if self.input_updater:
            return self.input_updater(x)
        return x - x.grad

    def step(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, scorer: Scorer=None) -> torch.Tensor:
        if self.skip_eval and x.grad is not None:
            x = self._update(x)
            self._evaluations = []
            return x
        
        if y is None or not x.requires_grad:
            x = utils.freshen(x)
            y = self._net(x)
        else:
            # TODO: Do I really want to do this
            y.retain_grad()
        loss, score = score(y, t, self.objective, scorer)
        if self.regularizer:
            loss = loss + self.regularizer(x, y)

        self._evaluations = [score.item()]
        loss.mean().backward()
        x = self._update(x)
        return x


class NullThetaOptim(ThetaOptim):

    def __init__(self, f, objective: Objective):
        super().__init__()
        self.objective = objective
        self.f = f

    def step(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, scorer: Scorer=None):
        
        _, self._evaluations = score(self.f(x), t, self.objective, scorer)
