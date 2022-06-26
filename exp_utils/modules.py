import math
import torch
import torch.nn as nn

from zenkai.modules import SKLearnModule



class Perceptron(SKLearnModule):

    def __init__(self, in_features: int, out_features: int, lr: float=1e-2):

        super().__init__()
        self._weight = torch.randn(in_features, out_features) / math.sqrt(out_features)
        self._bias = torch.randn(out_features) / math.sqrt(out_features)
        self._lr = lr

    def fit(self, x: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError

    def partial_fit(self, x: torch.Tensor, t: torch.Tensor):
        # https://towardsdatascience.com/perceptron-algorithm-in-python-f3ac89d2e537
        # https://www.simplilearn.com/tutorials/deep-learning-tutorial/perceptron#:~:text=Perceptron%20Learning%20Rule%20states%20that,a%20neuron%20fires%20or%20not.
        y = self.forward(x)
        m = (y == t).float()
        # think this is right but need to confirm
        self._weight += self._lr * (x.T @ (t - m))

    def score(self, x: torch.Tensor, t: torch.Tensor):
        y = self.forward(x)
        return (y == t).mean().item()

    def predict(self, x: torch.Tensor):
        
        return self.forward(x)
    
    def forward(self, x: torch.Tensor):
        
        return ((x @ self._weight + self._bias) >= 0).float() * 2 - 1
        
