from functools import partial
import typing
import sklearn
import torch.nn as nn
from abc import ABC, abstractmethod
import torch
from torch.nn import utils as nn_utils
import numpy as np
from . import utils
from .base import Assessment, Objective, InputOptim, Recording, Result, ScalarAssessment, ScalarNullAssessment, ThetaOptim


class SklearnThetaOptim(ThetaOptim):

    def __init__(self, sklearn_machine, partial_fit: bool=False):
        self._partial_fit = partial_fit
        self._machine = sklearn_machine
    
    @property
    def theta(self):
        return self._machine

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, result: Result=None) -> ScalarAssessment:
        if self._partial_fit:
            self._machine.partial_fit(x, t)
        else:
            self._machine.fit(x, t)
        return objective.assess(x, t, True)


class NRepeatInputOptim(InputOptim):

    def __init__(self, optim: InputOptim, n: int):
        super().__init__()
        assert n > 0, f'Argument n must be greater than 0 not {n}'
        self.optim = optim
        self._n = n

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, result: Result=None) -> typing.Tuple[torch.Tensor, ScalarAssessment]:
        for _ in range(self.n):
            x, evaluation = self.optim.step(x, t, objective, result)
            result = None
        return x, evaluation


class NRepeatThetaOptim(ThetaOptim):

    def __init__(self, optim: ThetaOptim, n: int):
        super().__init__()
        self.optim = optim
        self.n = n

    @property
    def theta(self):
        return self.optim.theta

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, result: Result=None) -> ScalarAssessment:
        for _ in range(self.n):
            evaluation = self.optim.step(x, t, objective)
            y = None
        return evaluation


class GradThetaOptim(ThetaOptim):

    def __init__(
        self, net: nn.Module, maximize: bool=False,
        optim: typing.Type[torch.optim.Optimizer]=torch.optim.AdamW
    ):
        super().__init__()
        self._net = net
        self._maximize = maximize
        self.optim = optim(net.parameters())

    @property
    def theta(self):
        return utils.get_parameters(self._net)

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, result: Result=None) -> ScalarAssessment:
        self.optim.zero_grad()
        if result is not None:
            assessment = objective.assess_output(result.y, t) + result.assessment()
        else:
            assessment = objective.assess(x, t)
        assessment.regularized.mean().backward()
        self.optim.step()
        return assessment


class InputUpdater(ABC):
    
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass 


class BasicInputUpdater(ABC):
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.grad 


class GradInputOptim(InputOptim):

    def __init__(self, maximize: bool=False, input_updater: InputUpdater=None, skip_eval: bool=False):
        super().__init__()
        # self._net = net
        self.maximize = maximize
        self.input_updater = input_updater or BasicInputUpdater()
        self.skip_eval = skip_eval
    
    def step(self, x: torch.Tensor, t: torch.Tensor,objective: Objective, result: Result=None) -> typing.Tuple[torch.Tensor, ScalarAssessment]:
        if self.skip_eval and x.grad is not None:
            x = self.input_updater(x)
            return x, ScalarNullAssessment(x.dtype, x.device, False)
        
        x = utils.freshen(x)
        assessment = objective.assess(x, t)
        assessment.regularized.mean().backward()
        x = self.input_updater(x) 
        return x, assessment


class NullThetaOptim(ThetaOptim):

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, result: Result=None) -> ScalarAssessment:
        
        evaluation = objective(x, t)
        return evaluation


class InputRecorder(InputOptim):

    def __init__(self, name: str, optim: InputOptim, recording: Recording=None):

        self._name = name
        self._optim = optim
        self._recording = recording or Recording()
    
    @abstractmethod
    def record(self, x: torch.Tensor, x_prime: torch.Tensor, assessment: Assessment):
        pass

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, result: Result=None) -> typing.Tuple[torch.Tensor, ScalarAssessment]:
        x_prime, assessment = self._optim.step(
            x, t, objective
        )
        self.record(x, x_prime, assessment)
        return x_prime, assessment

    @property
    def recording(self):
        return self._recording


class ThetaRecorder(InputOptim):

    def __init__(self, name: str, optim: ThetaOptim, recording: Recording=None):

        self._name = name
        self._optim = optim
        self._recording = recording or Recording()
    
    @abstractmethod
    def record(self, theta: torch.Tensor, theta_prime: torch.Tensor, assessment: Assessment):
        pass

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, result: Result=None) -> ScalarAssessment:
        theta = self._optim.theta
        x_prime, asessment = self._optim.step(
            x, t, objective
        )
        self.record(theta, self._optim.theta, asessment)
        return x_prime, asessment

    @property
    def recording(self):
        return self._recording
