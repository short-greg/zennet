import math
import typing
import torch.nn as nn
from abc import ABC, abstractmethod
import torch


from .modules import Invertable
from . import utils
from .base import (
    Assessment, Objective, InputOptim, Recording, 
    Result, ScalarAssessment, ScalarNullAssessment, 
    ThetaOptim
)
import numpy as np
import scipy.optimize


class NRepeatInputOptim(InputOptim):

    def __init__(self, optim: InputOptim, n: int):
        super().__init__()
        assert n > 0, f'Argument n must be greater than 0 not {n}'
        self.optim = optim
        self._n = n
    
    @property
    def n(self):
        return self._n

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
            evaluation = self.optim.step(x, t, objective, result)
            result = None
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
            assessment = objective.assess_output(result.y, t) + result.regularization
        else:
            assessment = objective.assess(x, t, True)
        assessment.regularized.mean().backward()
        self.optim.step()
        return assessment


class InputUpdater(ABC):
    
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass 


class BasicInputUpdater(ABC):
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.grad is not None, 'Cannot update input unless backpropagated and grad retained on x'
        return x - x.grad 


class GradInputOptim(InputOptim):

    def __init__(self, maximize: bool=False, input_updater: InputUpdater=None, skip_eval: bool=False):
        super().__init__()
        self.maximize = maximize
        self.input_updater = input_updater or BasicInputUpdater()
        self.skip_eval = skip_eval
    
    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, result: Result=None) -> typing.Tuple[torch.Tensor, ScalarAssessment]:
        if self.skip_eval and x.grad is not None:
            x = self.input_updater(x)
            return x, ScalarNullAssessment(x.dtype, x.device, False)
        
        y, result = objective.forward(x, True)
        assessment = objective.assess_output(y, t) + result.regularization
        assessment.regularized.mean().backward()
        x_prime = self.input_updater(result.x) 
        return x_prime, assessment


class NullThetaOptim(ThetaOptim):

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, result: Result=None) -> ScalarAssessment:
        
        return objective.assess_output(result.y, t)


class InputRecorder(InputOptim):

    def __init__(self, optim: InputOptim, recording: Recording=None, name: str=''):

        self._name = name or id(self)
        self._optim = optim
        self._recording = recording or Recording()
    
    @abstractmethod
    def record(self, x: torch.Tensor, x_prime: torch.Tensor, assessment: Assessment):
        pass

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, result: Result=None) -> typing.Tuple[torch.Tensor, ScalarAssessment]:
        x_prime, assessment = self._optim.step(
            x, t, objective, result
        )
        self.record(x, x_prime, assessment)
        return x_prime, assessment

    @property
    def recording(self):
        return self._recording


class ThetaRecorder(InputOptim):

    def __init__(self, optim: ThetaOptim, recording: Recording=None, name: str=''):

        self._name = name or id(self)
        self._optim = optim
        self._recording = recording or Recording()
    
    @abstractmethod
    def record(self, theta: torch.Tensor, theta_prime: torch.Tensor, assessment: Assessment):
        pass

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, result: Result=None) -> ScalarAssessment:
        theta = self._optim.theta
        asessment = self._optim.step(
            x, t, objective, result
        )
        self.record(theta, self._optim.theta, asessment)
        return asessment

    @property
    def recording(self):
        return self._recording


class EuclidInputRecorder(InputRecorder):
    
    def record(self, x: torch.Tensor, x_prime: torch.Tensor, assessment: ScalarAssessment):
        deviation = torch.sqrt(torch.sum((x - x_prime) ** 2)).item()
        self._recording.record_inputs(
            self._name, {
                'Deviation': deviation,
                'Regularized Evaluation': assessment.regularized.item(),
                'Unregularized Evaluation': assessment.unregularized.item()
            }
        )


class EuclidThetaRecorder(ThetaRecorder):
    
    def record(self, theta: torch.Tensor, theta_prime: torch.Tensor, assessment: ScalarAssessment):

        deviation = torch.sqrt(torch.sum((theta - theta_prime) ** 2)).item()
        self._recording.record_theta(
            self._name, {
                'Deviation': deviation,
                'Regularized Evaluation': assessment.regularized.item(),
                'Unregularized Evaluation': assessment.unregularized.item()
            }
        )


class LeastSquaresThetaOptimizer(ThetaOptim):

    def __init__(self, linear: nn.Linear, act_inverse: Invertable):

        self._linear = linear
        self._act_inverse = act_inverse
    
    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, result: Result=None) -> typing.Tuple[torch.Tensor, ScalarAssessment]:

        t_prime = self._act_inverse.forward(t)
        result = torch.linalg.lstsq(x, t_prime)
        self._linear.weight = nn.Parameter(
            result.solution[0].unsqueeze(1)
        )
        self._linear.bias = nn.Parameter(
            result.solution[1]
        )
        if result:
            return objective.assess_output(result.y, t) + result.reg

        return objective.assess(x, t)


class LeastSquaresInputOptimizer(InputOptim):

    def __init__(self, linear: nn.Linear, act_inverse: Invertable):

        self._linear = linear
        self._act_inverse = act_inverse
    
    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, result: Result=None) -> typing.Tuple[torch.Tensor, ScalarAssessment]:
        """
        """

        t_prime = self._act_inverse.forward(t)

        # remove the bias before prediction
        t_prime -= self._linear.bias.unsqueeze(0)
        x_prime = torch.linalg.lstsq(
            self._linear.weight.T, t_prime.T
        ).solution.T
        if result:
            assessment = objective.assess_output(result.y, t) + result.reg
        else:
            assessment = objective.assess(x, t) 
        return x_prime, assessment


class LeastSquaresBoundedInputOptimizer(InputOptim):
    """Optimizer to bind . Use if the incoming value is from a sigmoid
    """

    def __init__(self, linear: nn.Linear, act_inverse: Invertable, bounds=(-math.inf, math.inf), max_iter: int=5):

        self._linear = linear
        self._act_inverse = act_inverse
        self._bounds = bounds
        self._max_iter = max_iter
    
    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, result: Result=None) -> typing.Tuple[torch.Tensor, ScalarAssessment]:
        """
        """

        t_prime = self._act_inverse.forward(t)

        # remove the bias before prediction
        t_prime -= self._linear.bias.unsqueeze(0)
        A = self._linear.weight.T.detach().cpu().numpy()
        b = t_prime.T.detach().cpu().numpy()

        x_prime = scipy.optimize.lsq_linear(
            A, b, bounds=(self._bounds), max_iter=self._max_iter
        ).x.T
        if result:
            assessment = objective.assess_output(result.y, t) + result.reg
        else:
            assessment = objective.assess(x, t) 
        return x_prime, assessment
