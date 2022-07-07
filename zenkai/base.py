from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
import typing
import torch
import pandas as pd
import torch.nn as nn


class Reduction(ABC):
    
    @abstractmethod
    def __call__(self, evaluation):
        raise NotImplementedError


@dataclass
class Evaluation:
    regularized: torch.Tensor=None
    unregularized: torch.Tensor=None

    @classmethod
    def from_list(self, evaluations: typing.List):

        regularized = []
        unregularized = []
        for evaluation in evaluations:
            if evaluation.regularized.dim() > 0:
                regularized.extend(evaluation.regularized.tolist())
                unregularized.extend(evaluation.unregularized.tolist())
            else:
                regularized.append(evaluation.regularized.item())
                unregularized.append(evaluation.unregularized.item())
        
        return Evaluation(torch.tensor(regularized), torch.tensor(unregularized))


class Objective(nn.Module):

    # def __init__(self, reduction: Reduction):
    #     super().__init__()
    #     self.reduction = reduction
    
    @abstractproperty
    def maximize(self) -> bool:
        raise NotImplementedError

    @property
    def minimize(self) -> bool:
        return not self.maximize

    # @abstractmethod
    # def eval(self, y: torch.Tensor, t: torch.Tensor):
    #     raise NotImplementedError

    @abstractmethod
    def extend(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, multi: bool=False) -> Evaluation:
        pass
        # return self.reduction(self.eval(y, t), multi)
        # if multi:
        #     dim0 = y.size(0)
        #     evaluation = self.eval(y, t)
        #     evaluation = evaluation.view(dim0, *evaluation.shape[1:])
        #     reduced = self.reduction(evaluation)
        #     return Evaluation(reduced, reduced)
            
        # evaluation = self.eval(y, t)
        # reduced = self.reduction(evaluation[None]).view([])
        # return Evaluation(reduced, reduced)
    
    def best(self, evaluations, x):
        if self.maximize:
            return x[torch.argmax(evaluations)]
        return x[torch.argmin(evaluations)]


class ObjectivePair(Objective):

    def __init__(self, first: Objective, second: Objective):
        super().__init__()
        self.first = first
        self.second = second    
    
    def extend(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        y, regularization = self.first.extend(x)
        y, regularization2 = self.second.extend(y)
        if self.first.maximize is not self.second.maximize:
            regularization *= -1
        return y, regularization + regularization2

    def forward(self, x: torch.Tensor, t: torch.Tensor, multi: bool=False) -> Evaluation:
        
        y, regularization = self.first.extend(x)
        evaluation = self.second.forward(y, t)
        if self.first.maximize is not self.second.maximize:
            regularization *= -1

        evaluation.regularized += regularization
        return evaluation
        # if multi:
        #     dim0 = y.size(0)
        #     evaluation = self.eval(y, t)
        #     evaluation = evaluation.view(dim0, *evaluation.shape[1:])
        #     reduced = self.reduction(evaluation)
        #     return Evaluation(reduced, reduced)
            
        # evaluation = self.eval(y, t)
        # reduced = self.reduction(evaluation[None]).view([])
        # return Evaluation(reduced, reduced)
    
    def best(self, evaluations, x):
        if self.maximize:
            return x[torch.argmax(evaluations)]
        return x[torch.argmin(evaluations)]



class ThetaOptim(ABC):

    def __init__(self):
        self._evaluations = None

    @abstractproperty
    def theta(self):
        pass

    @abstractmethod
    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, y: torch.Tensor=None) -> Evaluation:
        pass


class InputOptim(ABC):

    def __init__(self):
        self._evaluations = None

    @abstractmethod
    def step(self, x: torch.Tensor, t: torch.Tensor,objective: Objective, y: torch.Tensor) -> typing.Tuple[torch.Tensor, Evaluation]:
        pass


def get_best(value: torch.Tensor, evaluation: Evaluation, maximize: bool=True):

    regularized = evaluation.regularized
    if regularized.ndim == 2:
        if maximize:
            idx = torch.argmax(regularized, 0, True)
        else:
            idx = torch.argmin(regularized, 0, True)
        idx.unsqueeze_(2)
        idx = idx.repeat(1, 1, value.shape[2])
        result = value.gather(0, idx)
        return result[0], Evaluation(evaluation.regularized[idx], evaluation.unregularized[idx])

    if maximize:
        idx = torch.argmax(evaluation.regularized)
    else:
        idx = torch.argmin(evaluation.regularized)
    return value[idx], Evaluation(evaluation.regularized[idx], evaluation.unregularize[idx])


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


class Recording(object):

    def __init__(self):

        self._df = pd.DataFrame([])

    def record_inputs(self, name, data: typing.Dict):
        self.df.loc[len(self._df)] = {
            'Recorder': name,
            'Type': 'Inputs',
            **data
        }

    def record_theta(self, name, data: typing.Dict):
        self.df.loc[len(self._df)] = {
            'Recorder': name,
            'Type': 'Theta',
            **data
        }

    @property
    def df(self):
        return self._df


class InputRecorder(InputOptim):

    def __init__(self, name: str, optim: InputOptim, recording: Recording=None):

        self._name = name
        self._optim = optim
        self._recording = recording or Recording()
    
    @abstractmethod
    def record(self, x: torch.Tensor, x_prime: torch.Tensor, evaluation: Evaluation):
        pass

    def step(self, x: torch.Tensor, t: torch.Tensor,objective: Objective, y: torch.Tensor) -> typing.Tuple[torch.Tensor, Evaluation]:
        x_prime, evaluation = self._optim.step(
            x, t, objective, y
        )
        self.record(x, x_prime, evaluation)
        return x_prime, evaluation

    @property
    def recording(self):
        return self._recording


class ThetaRecorder(InputOptim):

    def __init__(self, name: str, optim: ThetaOptim, recording: Recording=None):

        self._name = name
        self._optim = optim
        self._recording = recording or Recording()
    
    @abstractmethod
    def record(self, theta: torch.Tensor, theta_prime: torch.Tensor, evaluation: Evaluation):
        pass

    def step(self, x: torch.Tensor, t: torch.Tensor,objective: Objective, y: torch.Tensor) -> typing.Tuple[torch.Tensor, Evaluation]:
        theta = self._optim.theta
        x_prime, evaluation = self._optim.step(
            x, t, objective, y
        )
        self.record(theta, self._optim.theta, evaluation)
        return x_prime, evaluation

    @property
    def recording(self):
        return self._recording


class Machine(ABC):

    @abstractmethod
    def assess(self, y, t):
        pass

    @abstractmethod
    def output_ys(self, x) -> typing.Tuple[typing.Any, typing.Any]:
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def forward_update(self, x, t, objective: Objective=None, update_theta: bool=True):
        pass

    @abstractmethod
    def backward_update(self, x, t, outs=None, update_theta: bool=True, update_inputs: bool= True):
        pass

    @abstractmethod
    def get_y_out(self, outs):
        pass
    
    def __call__(self, x):
        return self.forward(x)


class Processor(ABC):

    @abstractmethod
    def output_ys(self, x):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x, t, outs=None):
        pass

    @abstractmethod
    def get_y_out(self, outs):
        pass
