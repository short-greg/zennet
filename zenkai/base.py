
from abc import ABC, abstractmethod, abstractproperty
import torch
import torch.nn as nn


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


class Reduction(ABC):
    
    @abstractmethod
    def __call__(self, evaluation):
        raise NotImplementedError


class Objective(nn.Module):

    def __init__(self, reduction: Reduction):
        super().__init__()
        self.reduction = reduction
    
    def minimize(self) -> bool:
        return not self.maximize

    @abstractproperty
    def maximize(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def eval(self, y: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError

    def reduce(self, evauation):
        return self.reduction(evauation)
    
    def forward_multi(self, y: torch.Tensor, t: torch.Tensor):

        n0 = y.size(0)
        # n1 = x.size(1)
        # t = t.view(t.shape[0] * t.shape[1], *t.shape[2:])
        # x = x.view(x.shape[0] * x.shape[1], *x.shape[2:])

        evaluation = self.eval(y, t)
        evaluation = evaluation.view(n0, *evaluation.shape[1:])
        return self.reduction(evaluation)

    def forward(self, y: torch.Tensor, t: torch.Tensor):
        evaluation = self.eval(y, t)
        evaluation = self.reduction(evaluation[None]).view([])
        return evaluation
    
    def best(self, evaluations, x):
        if self.maximize:
            return x[torch.argmax(evaluations)]
        return x[torch.argmin(evaluations)]


class SklearnModule(nn.Module):

    def __init__(self, in_features: int, out_features: int, out_dtype: torch.dtype=torch.float):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self.out_dtype = out_dtype

    @property
    def in_features(self):
        return self._in_features

    @property
    def out_features(self):
        return self._out_features
    
    @abstractmethod
    def fit(self, x: torch.Tensor, t: torch.Tensor):
        pass

    @abstractmethod
    def partial_fit(self, x: torch.Tensor, t: torch.Tensor):
        pass
    
    @abstractmethod
    def score(self, x: torch.Tensor, t: torch.Tensor):
        pass

    def predict(self, x: torch.Tensor):
        
        return self.forward(x)
    
    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass


class Recorder(ABC):

    @abstractmethod
    def adv(self):
        pass

    @abstractmethod
    def record_inputs(self, layer, prev_inputs, cur_inputs, evaluations):
        pass

    @abstractmethod
    def record_theta(self, layer, prev_theta, cur_theta, evaluations):
        pass
    
    @abstractproperty
    def pos(self):
        pass

    @abstractproperty
    def theta_df(self):
        pass

    @abstractproperty
    def input_df(self):
        pass
