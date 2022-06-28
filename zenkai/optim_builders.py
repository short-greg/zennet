from functools import partial

import torch

from .modules import Skloss
from .optimizers import GradThetaOptim, NRepeatThetaOptim, NullThetaOptim, SklearnThetaOptim, ThetaOptim
from .hill_climbing import HillClimbInputOptim, HillClimbPerturber, HillClimbSelector, HillClimbThetaOptim, SimpleHillClimbPerturber, SimpleHillClimbSelector


class ThetaOptimBuilder(object):

    def __init__(self):
        
        self._optim = None
        self.grad(lr=1e-2)
        self.n_repetitions = 1
        self.is_null = False

    def grad(self, optim_cls=None, **kwargs):
        optim_cls = optim_cls or  torch.optim.Adam
        self._optim = partial (
            GradThetaOptim,
            optim=partial(optim_cls, **kwargs)
        )
        return self
    
    def step_hill_climber(self, perturber: HillClimbPerturber=None, selector: HillClimbSelector=None):
        perturber = perturber or SimpleHillClimbPerturber()
        selector = selector or SimpleHillClimbSelector()
        self._optim = partial(HillClimbThetaOptim, perturber=perturber, selector=selector )
        return self
    
    def null(self, is_null: bool=True):
        self.is_null = is_null
        return self
    
    def repeat(self, n_repetitions: int):

        if n_repetitions <= 0:
            raise RuntimeError("Repetitions must be greater than or equal to 1")
        self.n_repetitions = n_repetitions
        return self

    def __call__(self, net, loss) -> ThetaOptim:
        if self.is_null:
            return NullThetaOptim(net, loss)
        if self.n_repetitions > 1:
            return NRepeatThetaOptim(self._optim(net, loss), self.n_repetitions)
        else:
            return self._optim(net, loss)


class SklearnOptimBuilder(object):

    def __init__(self):
        
        self.n_repetitions = 1
        self.is_partial = False
        self.is_null = False
    
    def partial(self, is_partial: bool=False):
        self.is_partial = is_partial
        return self
    
    def null(self, is_null: bool=True):
        self.is_null = is_null
        return self

    def repeat(self, n_repetitions: int):

        if n_repetitions <= 0:
            raise RuntimeError("Repetitions must be greater than or equal to 1")
        self.n_repetitions = n_repetitions
        return self

    def __call__(self, net) -> SklearnThetaOptim:
        if self.is_null:
            return NullThetaOptim(net, Skloss(net))
        optim = SklearnThetaOptim(net, self.is_partial)
        if self.n_repetitions > 1:
            return NRepeatThetaOptim(optim, self.n_repetitions)
        return optim


class InputOptimBuilder(object):

    def __init__(self):
        
        self._optim = None
        self.grad(lr=1e-2)
        self.n_repetitions = 1

    def grad(self, optim_cls=None, **kwargs):

        if optim_cls is None:
            optim_cls = torch.optim.Adam

        self._optim = partial (
            GradThetaOptim,
            optim=partial(optim_cls, **kwargs)
        )
        return self
    
    def step_hill_climber(self, perturber: SimpleHillClimbPerturber=None, selector: SimpleHillClimbSelector=None):
        perturber = perturber or SimpleHillClimbPerturber()
        selector = selector or SimpleHillClimbSelector()
        self._optim = partial(HillClimbInputOptim, perturber=perturber, selector=selector )
        return self
    
    def repeat(self, n_repetitions: int):

        if n_repetitions <= 0:
            raise RuntimeError("Rsepetitions must be greater than or equal to 1")
        self.n_repetitions = n_repetitions
        return self

    def __call__(self, net, loss) -> ThetaOptim:
        optim = self._optim(net, loss)
        if self.n_repetitions > 1:
            return NRepeatThetaOptim(optim, self.n_repetitions)
        return optim
