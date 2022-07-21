from functools import partial
import typing
from . import machinery
from typing import TypeVar
import torch

from .base import ParameterizedMachine, SklearnModule, SklearnThetaOptim, InputOptimBuilder, ThetaOptimBuilder
from .optimizers import GradInputOptim, GradThetaOptim
from .hill_climbing import HillClimbInputOptim, HillClimbThetaOptim, BinaryHillClimbPerturber, GaussianHillClimbPerturber, GaussianHillClimbSelector, SimpleHillClimbPerturber, SimpleHillClimbSelector

import torch.nn as nn
MachineBuilder = TypeVar("MachineBuilder")


# TODO: Remove the duplicate code below
class HillClimberInputBuilder(InputOptimBuilder):

    def __init__(self):
        super().__init__()
        self._perturber = None
        self._selector = None
        self.step()

    def step(self, maximize: bool=False):
        self._perturber = SimpleHillClimbPerturber()
        self._selector =  SimpleHillClimbSelector(maximize)
        return self
    
    def binary(self, k: int, p: float, maximize: bool=False):
        self._perturber = BinaryHillClimbPerturber(k, p, maximize)
        self._selector = SimpleHillClimbSelector(maximize)
        return self
    
    def gaussian(self, mean: float=-2, std: float=1, k: int=16, momentum: float=0.0, maximize: bool=False):
        self._perturber = GaussianHillClimbPerturber(mean, std, k, momentum, maximize)
        self._selector = GaussianHillClimbSelector(momentum, maximize)
        return self
    
    def __call__(self, machine):
        return HillClimbInputOptim(perturber=self._perturber, selector=self._selector)


class HillClimberThetaBuilder(ThetaOptimBuilder):

    def __init__(self, maximize: bool=False):
        super().__init__()
        self._perturber = None
        self._selector = None
        self.step(maximize)

    def binary(self, k: int, p: float, maximize: bool=False):
        self._perturber = BinaryHillClimbPerturber(k, p, maximize)
        self._selector = SimpleHillClimbSelector(maximize)
        return self

    def step(self, maximize: bool=False):
        self._perturber = SimpleHillClimbPerturber()
        self._selector = SimpleHillClimbSelector(maximize)
        return self
    
    def gaussian(self, mean: float=-2, std: float=1, k: int=16, momentum: float=0.0, maximize: bool=False):
        self._perturber = GaussianHillClimbPerturber(mean, std, k, momentum, maximize)
        self._selector = GaussianHillClimbSelector(momentum, maximize)
        return self

    def __call__(self, machine: ParameterizedMachine):
        if not isinstance(machine, machinery.TorchNN):
            raise RuntimeError('Can only use HillClimbThetaBuilder with Parameterized Machine')
        return HillClimbThetaOptim(machine, perturber=self._perturber, selector=self._selector)


class GradThetaBuilder(ThetaOptimBuilder):

    def __init__(self):
        super().__init__()
        self._optim_cls = None
        self._kwargs = {}
        self.grad()

    def grad(self, optim_cls=None, **kwargs):
        optim_cls = optim_cls or  torch.optim.Adam
        self._optim_cls = optim_cls
        self._kwargs = kwargs
        return self
    
    def __call__(self, machine: machinery.TorchNN):
        if not isinstance(machine, machinery.TorchNN):
            raise RuntimeError('Can only use grad theta builder with TorchNN')
        return GradThetaOptim(machine.module, False, partial(self._optim_cls, **self._kwargs))
    

class GradInputBuider(InputOptimBuilder):

    def __init__(self):
        super().__init__()
        self._optim_cls = None
        self._kwargs = {}
        self.grad()
    
    def grad(self, optim_cls=None, **kwargs):

        optim_cls = optim_cls or  torch.optim.Adam
        self._optim_cls = optim_cls
        self._kwargs = kwargs
        return self
    
    def __call__(self, machine):
        return GradInputOptim(False, skip_eval=True)


class SklearnThetaBuilder(ThetaOptimBuilder):

    def __init__(self):
        super().__init__()
        self.is_partial = True
    
    def full(self):
        self.is_partial = False
        return self
    
    def partial(self):
        self.is_partial = True
        return self
    
    def __call__(self, machine: machinery.SklearnMachine):
        if not isinstance(machine, machinery.SklearnMachine):
            raise RuntimeError('Can only use SklearnThetaBuilder with SklearnMachine')
        return SklearnThetaOptim(machine.module, self.is_partial)
