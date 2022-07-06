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
from torch.nn import functional as nn_func

from .base import Scorer, Reduction, Objective, SklearnModule



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


class MeanReduction(Reduction):
    
    def __call__(self, grade):
        
        return grade.reshape(grade.size(0), -1).mean(dim=1)


class LossObjective(Objective):

    def __init__(self, th_loss, reduction: Reduction):
        super().__init__(reduction)
        loss = th_loss(reduction="none")
        loss.reduction = 'none'
        self.loss = loss

    @property
    def maximize(self) -> bool:
        return False

    def eval(self, y, t):
        return self.loss(y, t)


def score(x: torch.Tensor, t: torch.Tensor, objective: Objective, scorer: Scorer=None):

    if scorer is not None:
        return scorer.assess(x, t)
    return objective(x, t)


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

# 1) need a special type of 'objective'
# 2) 

class AdaptiveObjective(Objective):

    def __init__(self, reduction: str='mean'):
        pass

    @abstractmethod
    def update(self, x: th.Tensor, t: th.Tensor):
        pass
    
    @abstractproperty
    def evaluations(self):
        pass

    @property
    def maximize(self) -> bool:
        return False

    def eval(self, x, t):
        return self.loss(x, t)


# TODO: I think i want to use a machine
# rather than objective for this... The machine will be limiting
# I am on the right track but I don't think this will work well

# t = adapt(x, t)
# loss(x, t)
# calculate gradients
# how to deal with regularization

class AdaptiveRegression(AdaptiveObjective):

    def __init__(self, regression_loss: nn.Module, reduction: str='mean', w: float=0.1):
        super().__init__(reduction)
        self._regression_loss = regression_loss
        assert 0 < w < 1, f"Argument w must be between value of 0 and 1 not {w}"
        self._w = w
        self._w_comp = 1 - self._w

    def update(self, x: th.Tensor, t: th.Tensor):
        # output the true evaluations
        # just have it be like a regular 'objective'
        # no 'update' like this. Then if you decide to use gradient or something you can
        # and combine it with other losses
        # include "preprocess" method for targets
        sz = x.size()
        x = x.view(x.size(0), -1).detach().requires_grad()
        x.retains_grad()
        t = self._w_comp * x + self._w * t.view(t.size(0), -1)
        loss: torch.Tensor = self._loss(x, t)
        self._evaluations = self.reduction(loss)
        loss.backward()
        return (x - x.grad).view(*sz)

    @abstractproperty
    def evaluations(self):
        return self._evaluations


class AdaptiveClassificaton(AdaptiveObjective):

    def __init__(self, class_loss: nn.Module, reduction: str='mean', w: float=0.1):
        super().__init__(reduction)
        self._loss = class_loss
        assert 0 < w < 1, f"Argument w must be between value of 0 and 1 not {w}"
        self._w = w
        self._w_comp = 1 - self._w

    def update(self, x: th.Tensor, t: th.Tensor):
        sz = x.size()
        x = x.view(x.size(0), -1).detach().requires_grad()
        x.retains_grad()
        t: torch.FloatTensor = nn_func.one_hot(t.long()).float().view(t.size(0), -1)
        t = self._w_comp * x + self._w * t
        t = t / t.sum(dim=1, keepdim=True)
        loss: torch.Tensor = self._loss(x, t)
        self._evaluations = self.reduction(loss)
        loss.backward()
        return (x - x.grad).view(*sz)

    @abstractproperty
    def evaluations(self):
        return self._evaluations



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

