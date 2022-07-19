from abc import abstractmethod
import math
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import torch
import torch.nn as nn
from .base import SklearnModule
import torch.nn.functional as nn_func


class SklearnWrapper(SklearnModule):

    def __init__(self, module, in_features: int, out_features: int, out_dtype: torch.dtype=torch.float):
        super().__init__(in_features, out_features, out_dtype)
        self.module = module
        self._fitted = False
        self._base = nn.Linear(in_features, out_features)
    
    def fit(self, x: torch.Tensor, t: torch.Tensor):
        if self._fitted:
            self.module = MultiOutputRegressor(SVR())

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


class Lambda(nn.Module):
    """
    Executes any function whether it uses tensors or not 
    """

    def __init__(self, f):
        super().__init__()
        self._f = f

    def forward(self, *x: torch.Tensor):
        return self._f(*x)


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


class Invertable(nn.Module):

    @abstractmethod
    def invert(self, y) -> torch.Tensor:
        pass


class SigmoidInv(Invertable):

    def inverse(self, y: torch.Tensor):
        # x = ln(y/(1-y))
        return torch.log(
            y / (1 - y)
        )

    def forward(self, x: torch.Tensor):
        return nn_func.sigmoid(x)


class LReLUInv(Invertable):

    def __init__(self, negative_slope: float=None):
        super().__init__()
        self._negative_slope = negative_slope

    def inverse(self, y: torch.Tensor):
        return nn_func.leaky_relu(y, 1 / self._negative_slope)

    def forward(self, x: torch.Tensor):
        # forward = max(0,x)+negative_slope∗min(0,x)
        return nn_func.leaky_relu(x, self._negative_slope)
