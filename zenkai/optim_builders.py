from functools import partial
import torch
from .optimizers import GradThetaOptim, NRepeatInputOptim, NRepeatThetaOptim, NullThetaOptim, SklearnThetaOptim, ThetaOptim
from .hill_climbing import BinaryHillClimbPerturber, GaussianHillClimbPerturber, GaussianHillClimbSelector, HillClimbInputOptim, HillClimbPerturber, HillClimbSelector, HillClimbThetaOptim, SimpleHillClimbPerturber, SimpleHillClimbSelector


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
    
    def step_hill_climber(self, maximize: bool=False):
        perturber = SimpleHillClimbPerturber()
        selector = SimpleHillClimbSelector(maximize)
        self._optim = partial(HillClimbThetaOptim, perturber=perturber, selector=selector )
        return self
    
    def step_gaussian_hill_climber(self, mean: float=-2, std: float=1, k: int=16, momentum: float=0.0, maximize: bool=False):
        perturber = GaussianHillClimbPerturber(mean, std, k, momentum, maximize)
        selector = GaussianHillClimbSelector(momentum, maximize)
        self._optim = partial(HillClimbThetaOptim, perturber=perturber, selector=selector)
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
    
    def partial(self, is_partial: bool=True):
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

    def __call__(self, net, objective) -> SklearnThetaOptim:
        if self.is_null:
            return NullThetaOptim(net, objective)
        optim = SklearnThetaOptim(net, objective, self.is_partial)
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
    
    def step_hill_climber(self, maximize: bool=False):
        perturber = SimpleHillClimbPerturber()
        selector =  SimpleHillClimbSelector(maximize)
        self._optim = partial(
            HillClimbInputOptim, 
            perturber=perturber, 
            selector=selector
        )
        return self
    
    def step_binary_hill_climber(self, k: int, p: float, maximize: bool=False):
        perturber = BinaryHillClimbPerturber(k, p, maximize)
        selector = SimpleHillClimbSelector(maximize)
        self._optim = partial(HillClimbInputOptim, perturber=perturber, selector=selector)
        return self
    
    def step_gaussian_hill_climber(self, mean: float=-2, std: float=1, k: int=16, momentum: float=0.0, maximize: bool=False):
        perturber = GaussianHillClimbPerturber(mean, std, k, momentum, maximize)
        selector = GaussianHillClimbSelector(momentum, maximize)
        self._optim = partial(HillClimbInputOptim, perturber=perturber, selector=selector)
        return self

    def repeat(self, n_repetitions: int):

        if n_repetitions <= 0:
            raise RuntimeError("Rsepetitions must be greater than or equal to 1")
        self.n_repetitions = n_repetitions
        return self

    def __call__(self, net, loss) -> ThetaOptim:
        optim = self._optim(net, loss)
        if self.n_repetitions > 1:
            return NRepeatInputOptim(optim, self.n_repetitions)
        return optim


