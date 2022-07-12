from abc import ABC, abstractmethod, abstractproperty
import math
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import torch
import torch.nn as nn
import torch as th
import sklearn.base
import numpy as np
from torch.nn import functional as nn_func

from .base import SklearnModule


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
