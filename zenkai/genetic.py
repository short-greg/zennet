from abc import ABC, abstractmethod
import typing
import torch
import torch.nn as nn

from zenkai.machinery import TorchNN
from .modules import Objective
# from .optimizers import , get_theta, Scorer, update_theta
from .base import Evaluation, ThetaOptim, get_best


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


class Breeder(ABC):

    @abstractmethod
    def __call__(self, couples):
        pass


class SimpleBreeder(Breeder):

    def __call__(self, couples):
        # TODO: may want to make in place
        first_parent = (torch.rand(couples[0].size()) > 0.5).float()
        second_parent = 1 - first_parent
        return couples[0] * first_parent + couples[1] * second_parent


class PairSelector(ABC):

    @abstractmethod
    def __call__(self, chromosomes: torch.Tensor, fitness: torch.Tensor):
        pass


class SimpleSelector(PairSelector):

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


class DiscreteMutator(GeneProcessor):

    def __init__(self, p: float):
        self.p: float = p

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p: float):
        if not (0 <= p <= 1):
            raise RuntimeError(f'Value p must be in range [0,1] not {p}')
        self._p = p

    def __call__(self, chromosomes: torch.Tensor):

        flip = (
            torch.rand(chromosomes.size(), device=chromosomes.device
        ) <= self.p)
        flipped = ~chromosomes.bool()
        return (flip * flipped + ~flip * chromosomes).type_as(chromosomes)


class Bind(GeneProcessor):

    def __init__(
        self, lower_bound: typing.Union[float, torch.Tensor], 
        upper_bound: typing.Union[float, torch.Tensor]
    ):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def __call__(self, chromosomes: torch.Tensor):
        return chromosomes.clamp(
            min=self._lower_bound, max=self._upper_bound
        )


class Elitism(GeneProcessor):
    
    def __init__(self, k=1):
        if k <= 1:
            raise RuntimeError(f"Argument k must be a value of 1 or greater.")
        
    def __call__(
        self, prev_chromosomes: torch.Tensor, cur_chromosomes: torch.Tensor, fitness: torch.Tensor
    ):
        
        k = min(self.k, len(fitness))
        _, elite_indices = torch.topk(fitness, k, dim=0)
        return torch.concat([cur_chromosomes, prev_chromosomes[elite_indices]], dim=0)


class MultiProcessor(GeneProcessor):

    def __init__(self, processors: typing.List[GeneProcessor]=None):

        self._processors = processors or []

    def __call__(self, chromosomes: torch.Tensor):

        for processor in self._processors:
            chromosomes = processor(chromosomes)
        return chromosomes


class GeneticThetaOptim(ThetaOptim):

    def __init__(
        self, machine: TorchNN, objective: Objective, 
        initializer: Initializer, pair_selector: PairSelector, 
        breeder: Breeder, processors: typing.List[GeneProcessor]=None, 
        elitism: Elitism=None
    ):
        
        super().__init__()
        self._machine = machine
        self._chromosomes = initializer(machine.theta)
        self.objective = objective
        self.pair_selector = pair_selector
        self.breeder = breeder
        self.processors = MultiProcessor(processors)
        self.elitism = elitism

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, y: torch.Tensor=None) -> Evaluation:
        chromosome_pairs = self.pair_selector()
        children = self.breeder(chromosome_pairs)
        chromosomes = self.processors(children)

        if self.elitism is not None:
            children = self.elitism()
        evaluations = []
        for child in children:
            self._machine.theta = child
            evaluations.append(objective(x, t))

        best, evaluation = get_best(chromosomes, Evaluation.from_list(evaluations), objective.maximize)
        self._chromosomes = children
        self._machine.theta = best
        return evaluation
