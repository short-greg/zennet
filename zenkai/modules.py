from abc import ABC, abstractmethod, abstractproperty
import math
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import torch
import torch.nn as nn
import torch as th
import sklearn
import sklearn.base
import numpy as np


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


class SklearnWrapper(SklearnModule):

    def __init__(self, module, in_features: int, out_features: int, out_dtype: torch.dtype=torch.float):
        super().__init__(in_features, out_features, out_dtype)
        self.module = module
        self._fitted = False
        self._base = nn.Linear(in_features, out_features)
    
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


class MeanReduction(Reduction):
    
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
    
    def forward_multi(self, x: torch.Tensor, t: torch.Tensor):

        n0 = x.size(0)
        # n1 = x.size(1)
        # t = t.view(t.shape[0] * t.shape[1], *t.shape[2:])
        # x = x.view(x.shape[0] * x.shape[1], *x.shape[2:])

        evaluation = self.eval(x, t)
        evaluation = evaluation.view(n0, *evaluation.shape[1:])
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


# class Skloss(object):

#     def __init__(self, sklearn_module: SklearnModule, objective: Objective):
#         super().__init__()
#         self._machine = sklearn_module
#         self._objective = objective

#     @property
#     def maximize(self) -> bool:
#         return False

#     def forward_multi(self, x, t):
#         y = self._machine(x)
#         return self._objective.forward_multi(y, t)
    
#     def eval(self, x, t):
#         return self._machine.score(x, t)


class Perceptron(SklearnModule):

    def __init__(self, in_features: int, out_features: int, lr: float=1e-2):

        super().__init__( in_features, out_features)
        self._weight = torch.randn(in_features, out_features) / math.sqrt(out_features)
        self._bias = torch.randn(out_features) / math.sqrt(out_features)
        self._lr = lr

    def fit(self, x: torch.Tensor, t: torch.Tensor):
        # want to reset
        self.partial_fit(x, t)

    def partial_fit(self, x: torch.Tensor, t: torch.Tensor):
        # https://towardsdatascience.com/perceptron-algorithm-in-python-f3ac89d2e537
        # https://www.simplilearn.com/tutorials/deep-learning-tutorial/perceptron#:~:text=Perceptron%20Learning%20Rule%20states%20that,a%20neuron%20fires%20or%20not.
        
        y = self.forward(x)
        y = y * 2 - 1
        t = t * 2 - 1
        
        m = (y == t).float()
        # think this is right but need to confirm
        self._weight += self._lr * (x.T @ (t - m))

    def score(self, x: torch.Tensor, t: torch.Tensor):
        y = self.forward(x)
        return (y == t).type_as(x).mean().item()
    
    def forward(self, x: torch.Tensor):
        
        return ((x @ self._weight + self._bias) >= 0).float()
        
