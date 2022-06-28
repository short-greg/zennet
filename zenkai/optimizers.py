from functools import partial
import typing
import sklearn
import torch.nn as nn
from abc import ABC, abstractmethod, abstractproperty
import torch
from .modules import Objective, SklearnModule, Skloss
from torch.nn import utils as nn_utils
import numpy as np


def update_theta(module, theta):
    nn_utils.vector_to_parameters(theta, module.parameters())


def get_theta(module):
    return nn_utils.parameters_to_vector(module.parameters())


class Scorer(ABC):

    @abstractmethod
    def assess(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractproperty
    def maximize(self):
        pass


class ThetaOptim(ABC):

    def __init__(self):
        self._evaluations = None

    @abstractproperty
    def theta(self):
        pass

    @abstractmethod
    def step(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, scorer: Scorer=None):
        pass

    @property
    def evaluations(self):
        return self._evaluations


class InputOptim(ABC):

    def __init__(self):
        self._evaluations = None

    @abstractmethod
    def step(self, x: torch.Tensor, t: torch.Tensor, scorer: Scorer=None) -> torch.Tensor:
        pass

    @property
    def evaluations(self):
        return self._evaluations



class SklearnThetaOptim(ThetaOptim):

    def __init__(self, sklearn_machine, partial_fit: bool=False):
        self._evaluations = None
        self._partial_fit = partial_fit
        self._machine = sklearn_machine
    
    @property
    def theta(self):
        return self._machine

    def step(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, scorer: Scorer=None):
        if self._partial_fit:
            self._machine.partial_fit(x, t)
        else:
            self._machine.fit(x, t)
        self._evaluations = [self._machine.score(x, t)]



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
        self._evaluations = evaluations


class GradThetaOptim(ThetaOptim):

    def __init__(
        self, net: nn.Module, objective: Objective, 
        optim: typing.Type[torch.optim.Optimizer]=torch.optim.AdamW
    ):
        super().__init__()
        self._net = net
        self.objective = objective
        self.optim = optim(self._net.parameters())

    @property
    def theta(self):
        return get_theta(self._net)
    
    def step(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, scorer: Scorer=None):
        self.optim.zero_grad()
        if y is None:
            y = self._net(x)
        if scorer is None:
            evaluation = self.objective(y, t)
        else:
            evaluation = scorer.assess(y)
        evaluation.mean().backward()
        self.optim.step()
        self._evaluations = [evaluation.item()]


class InputUpdater(ABC):

    def updater(self, x: torch.Tensor):
        pass


class GradInputOptim(InputOptim):

    def __init__(self, net: nn.Module, objective: Objective, input_updater: InputUpdater=None, skip_eval: bool=False):
        super().__init__()
        self._net = net
        self.objective = objective
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
            x = x.detach()
            x.requires_grad_()
            x.retain_grad()
            y = self._net(x)
        else:
            y.retain_grad()
        if scorer is None:
            evaluation = self.objective(y, t)
        else:
            evaluation = scorer.assess(y)
        evaluation.mean().backward()
        x = self._update(x)
        self._evaluations = [evaluation.item()]
        return x


class NullThetaOptim(ThetaOptim):

    def __init__(self, f, loss: nn.Module):
        super().__init__()
        self.loss = loss
        self.f = f

    def step(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, scorer: Scorer=None):
        if scorer:
            self._evaluations = scorer.assess(self.f(x))
        else:
            self._evaluations = self.loss(self.f(x), t)


class Initializer(ABC):

    @abstractmethod
    def __call__(self, theta: torch.Tensor):
        pass


class RealInitializer(Initializer):

    def __init__(self, n_couples: int):
        self.n_couples = n_couples

    def __call__(self, theta: torch.Tensor):
        return torch.randn(self.n_couples, *theta.size()) / torch.sqrt(theta.nelement())


class DiscreteInitializer(Initializer):

    def __init__(self, n_couples: int):
        self.n_couples = n_couples

    def __call__(self, theta: torch.Tensor):
        return torch.round(torch.randn(self.n_couples, *theta.size()))


class Recombiner(ABC):

    @abstractmethod
    def __call__(self, couples, chromosomes, evaluations):
        pass


class SimpleRecombiner(Recombiner):

    def __call__(self, couples, chromosomes, evaluations):

        # TODO: may want to make in place
        first_parent = (torch.rand(couples[0].size()) > 0.5).float()
        second_parent = 1 - first_parent
        return couples[0] * first_parent + couples[1] * second_parent


class Selector(ABC):

    @abstractmethod
    def __call__(self, chromosomes: torch.Tensor, fitness: torch.Tensor):
        pass


class SimpleSelector(Selector):

    def __call__(self, chromosomes: torch.Tensor, fitness: torch.Tensor):

        p = (fitness / torch.sum(fitness, dim=fitness.dim() - 1)).detach()
        selection = torch.multinomial(
            p, 2 * len(fitness), True
        )
        selection.view(2, p.size(0))
        return chromosomes.view(1, -1)[selection]


class GeneProcessor(ABC):

    def __call__(self, chromosomes: torch.Tensor):
        pass


class GaussianMutator(GeneProcessor):

    def __init__(self, scale: float=1.):
        self.scale = scale

    def __call__(self, chromosomes: torch.Tensor):

        return chromosomes + torch.randn(
            chromosomes.size(), device=chromosomes.device
        ) * self.scale


class GeneticThetaOptim(ThetaOptim):

    def __init__(self, net: nn.Module, objective: Objective, initializer: Initializer, selector: Selector, recombiner: Recombiner, processor: GeneProcessor):
        
        super().__init__()
        self.net = net
        self._chromosomes = initializer(get_theta(net))
        self.objective = objective
        self.selector = selector
        self.recombiner = recombiner
        self.processor = processor
    
    def select(self):
        return self.selector(self._chromosomes, self._evaluations)

    def recombine(self, chromosome_pairs: torch.Tensor):
        return self.recombiner(chromosome_pairs, self._chromosomes, self._evaluations)

    def step(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, scorer: Scorer=None) -> torch.Tensor:
        chromosome_pairs = self.selector()
        chromosomes = self.recombiner(chromosome_pairs)
        chromosomes = self.processor(chromosomes)
        evaluations = []
        for chromosome in chromosomes:
            update_theta(self.net, chromosome)
            y = self.net.forward(x)
            if scorer:
                evaluations.append(scorer.assess(x))
            else:
                evaluations.append(self.objective(x, t))
        best = chromosomes[torch.argmax(evaluations)]
        self._chromosomes = chromosomes
        self._evaluations = evaluations
        update_theta(self.net, best)
