from abc import ABC, abstractmethod, abstractproperty
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import torch
import torch.nn as nn
import torch as th
import sklearn
import sklearn.base
import numpy as np


class SklearnModule(nn.Module):

    def __init__(self, module, in_features: int, out_features: int, out_dtype: torch.dtype=torch.float):
        super().__init__()
        self.module = module
        self._in_features= in_features
        self._out_featurse = out_features
        self.out_dtype = out_dtype
        self._fitted = False
        self._base = nn.Linear(in_features, out_features)

    @property
    def in_features(self):
        return self._in_features

    @property
    def out_features(self):
        return self._out_featurse
    
    def fit(self, x: torch.Tensor, t: torch.Tensor):
        if self._fitted:
            self.module = MultiOutputRegressor(SVR())
        #    self.module = sklearn.base.clone(self.module)

        result = self.module.fit(
            x.detach().cpu().numpy(),
            t.detach().cpu().numpy()
        )
        
        self._fitted = True
        return result

    def partial_fit(self, x: torch.Tensor, t: torch.Tensor):
        return self.module.partial_fit(
            x.detach().cpu().numpy(),
            t.detach().cpu().numpy()
        )
    
    def score(self, x: torch.Tensor, t: torch.Tensor):
        if not self._fitted:
            return None
        return self.module.score(x.detach().cpu().numpy(), t.detach().cpu().numpy())
    
    def predict(self, x: torch.Tensor):
        
        return self.forward(x)
    
    def forward(self, x: torch.Tensor):
        if not self._fitted:
            return self._base(x).type(self.out_dtype)

        return torch.from_numpy(self.module.predict(x.detach().cpu().numpy())).type(self.out_dtype)


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
        
        return grade.reshape(grade.size(0), -1).mean(dim=1)


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


class Skloss(object):

    def __init__(self, sklearn_machine):
        super().__init__()
        self._machine = sklearn_machine

    @property
    def maximize(self) -> bool:
        return False
    
    def eval(self, x, t):
        return self._machine.score(x, t)
