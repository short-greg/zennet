from abc import ABC, abstractmethod, abstractproperty
import torch
import torch.nn as nn
import torch as th



class SKLearnModule(nn.Module):

    def __init__(self, machine):
        super().__init__()
        self._machine = machine
    
    def forward(self, x: torch.Tensor):

        return torch.from_numpy(self._machine.predict(x.detach().cpu().numpy()))


class Blackbox(nn.Module):
    """
    Executes any function whether it uses tensors or not 
    """

    def __init__(self, f, preprocessor=None, postprocessor=None):
        super().__init__()
        self._f = f
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor

    def forward(self, x):

        return self._postprocessor(self._f(self._preprocessor(x)))

class Reduction(ABC):
    
    @abstractmethod
    def __call__(self, evaluation):
        raise NotImplementedError


class MeanReduction(ABC):
    
    def __call__(self, grade):
        
        return grade.view(grade.size(0), -1).mean(dim=1)


class Objective(nn.Module):

    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction
    
    def minimize(self) -> bool:
        return not self.maximize

    @abstractproperty
    def maximize(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def eval(self, x, t):
        raise NotImplementedError

    def reduce(self, grade):
        return self.reduction(grade)
    
    def forward_multi(self, x, t):
        t = t[None].repeat(x.size(0), *[1] * len(t.size()))

        n0 = x.size(0)
        n1 = x.size(1)
        t = t.view(t.shape[0] * t.shape[1], *t.shape[2:])
        x = x.view(x.shape[0] * x.shape[1], *x.shape[2:])

        evaluation = self.eval(x, t)
        evaluation = evaluation.view(n0, n1, *evaluation.shape[1:])
        reduced = self.reduction(evaluation)
        return reduced

    def forward(self, x, t):
        evaluation = self.eval(x, t)
        reduced = self.reduction(evaluation[None]).view([])
        return reduced
    
    def best(self, evaluations, x):
        if self.maximize:
            return x[th.argmax(evaluations)]
        return x[th.argmin(evaluations)]



class LossObjective(Objective):

    def __init__(self, th_loss, reduction: Reduction):
        super().__init__(reduction)
        loss = th_loss(reduction="none")
        loss.reduction = 'none'
        self.loss = loss

    @property
    def maximize(self) -> bool:
        return False

    def eval(self, x, t):
        return self.loss(x, t)
