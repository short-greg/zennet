from abc import ABC, abstractmethod, abstractproperty
import typing
import torch
import pandas as pd
import torch.nn as nn
from .utils import batch_flatten

import scipy.optimize as sciopt


class Assessment:

    unregularized: typing.Optional[torch.Tensor]=None
    regularized: typing.Optional[torch.Tensor]=None
    maximize: bool=False

    def __init__(
        self, unregularized: torch.Tensor, 
        regularized: typing.Optional[torch.Tensor]=None, maximize: bool=False
    ):
        self.regularized = regularized
        self.unregularized = unregularized
        self.maximize = maximize
        if self.regularized is None:
            self.regularized = self.unregularized
        
        assert (self.regularized is None and self.unregularized is None) or (self.regularized.size() == self.unregularized.size())
    
    def is_null(self):
        return self.unregularized is None

    def to_maximize(self, maximize: bool=True):
        if self.maximize and maximize or (not self.maximize and not maximize):
            return self.__class__(self.unregularized, self.regularized, True)
        return self.__class__(-self.unregularized, -self.regularized, False)

    def to_minimize(self, minimize: bool=True):
        return self.to_maximize(not minimize)

    def __add__(self, other):

        if other is None:
            return self

        other = other.to_maximize(self.maximize)
        return self.__class__(
            self.unregularized + other.unregularized,
            self.regularized + other.regularized,
            self.maximize
        )

    def __mul__(self, value):
        
        return self.__class__(
            self.unregularized * value,
            self.regularized * value,
            self.maximize
        )


class ScalarAssessment(Assessment):

    def backward(self, for_unreg: bool=False):
        
        if self.unregularized is not None and for_unreg:
            self.unregularized.backward()
        elif self.regularized is not None:
            self.regularized.backward()
    
    def item(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        if self.unregularized is not None:
            return self.unregularized.item(), self.regularized.item()
        return None, None


class ScalarNullAssessment(Assessment):

    def __init__(self, dtype: torch.dtype, device: torch.device, maximize: bool=False):
        evaluation = torch.tensor(0.0, dtype=dtype, device=device)
        super().__init__(evaluation, evaluation)


class BatchAssessment(Assessment):    
    
    def mean(self):
    
        return ScalarAssessment(
            self.unregularized.mean(),
            self.regularized.mean()
        )

    @property
    def shape(self) -> int:
        return self.unregularized.shape

    @property
    def batch_size(self) -> int:
        return self.regularized.shape[0]
    
    def sum(self):
        return ScalarAssessment(
            self.unregularized.mean(),
            self.regularized.mean()
        )

    def __add__(self, other):

        if isinstance(other, BatchNullAssessment):
            return self
        
        return super().__add__(other)


class BatchNullAssessment(Assessment):

    def __init__(self, dtype: torch.dtype, device, maximize: bool=False):
        super().__init__(None, maximize=maximize)
        self.dtype = dtype
        self.device = device
    
    def mean(self):
        return ScalarAssessment(
            torch.tensor(0.0, dtype=self.dtype, device=self.device)
        )
    
    def sum(self):
        return ScalarAssessment(
            torch.tensor(0.0, dtype=self.dtype, device=self.device)
        )

    @property
    def shape(self) -> int:
        return torch.Size([0])

    @property
    def batch_size(self) -> int:
        return 0

    def __add__(self, other):

        if isinstance(other, BatchAssessment):
            return other
        
        return BatchNullAssessment(self.dtype, self.device)

    def __mul__(self, value):
        
        return BatchNullAssessment(self.dtype, self.device)


class PopulationAssessment(Assessment):
    
    def best(self, for_reg: bool=True) -> typing.Tuple[torch.LongTensor, torch.FloatTensor]:

        if for_reg:
            x = self.regularized
        else:
            x = self.unregularized

        if self.maximize:
            result = torch.max(x, dim=0)
        else: result = torch.min(x, dim=0)
        return result[1], result[0] 
    
    def append(self, assessment: ScalarAssessment):
        assessment = assessment.to_maximize(self.maximize)
        unregularized = assessment.unregularized.unsqueeze(0)
        regularized = assessment.regularized.unsqueeze(0)
        
        return PopulationAssessment(
            torch.cat([self.unregularized, unregularized], dim=0),
            torch.cat([self.regularized, regularized], dim=0),
            self.maximize
        )
    
    def __len__(self) -> int:
        return len(self.regularized)

    @property
    def shape(self) -> int:
        return self.unregularized.shape
    
    def __getitem__(self, idx: int):
        return ScalarAssessment(
            self.unregularized[idx], self.regularized[idx], self.maximize
        )

    def backward(self, for_unreg: bool=False):
        if self.is_null:
            return

        x = self.unregularized if for_unreg else self.regularized
        x.sum().backward()

    @classmethod
    def concat(cls, *assessments: ScalarAssessment):
        assert len(assessments) > 0, f"Number of assessments must be greater than 0."
        unregularized = [a.unregularized.unsqueeze(0) for a in assessments]
        regularized = [a.regularized.unsqueeze(0) for a in assessments]
        device = unregularized[0].device
        return PopulationAssessment(
            torch.tensor(unregularized, device=device),
            torch.tensor(regularized, device=device)
        )


class PopulationBatchAssessment(Assessment):
    
    def mean(self):
        return PopulationAssessment(
            batch_flatten(self.unregularized).mean(1),
            batch_flatten(self.regularized).mean(1)
        )
    
    def sum(self):
        return PopulationAssessment(
            batch_flatten(self.unregularized).sum(1),
            batch_flatten(self.regularized).sum(1)
        )

    def to_batch(self):
        
        new_size = torch.Size([self.batch_size * self.population_size, *self.regularized.size[1:]])
        return BatchAssessment(
            self.unregularized.view(new_size),
            self.regularized.view(new_size),
            self.maximize
        )

    @property
    def batch_size(self) -> int:
        return self.unregularized.shape[1]

    @property
    def population_size(self) -> int:
        return self.unregularized.shape[0]

    @property
    def shape(self) -> int:
        return self.unregularized.shape

    def append(self, assessment: BatchAssessment):

        assessment = assessment.to_maximize(self.maximize)
        unregularized = assessment.unregularized.unsqueeze(0)
        regularized = assessment.regularized.unsqueeze(0)
        return PopulationBatchAssessment(
            torch.cat([self.unregularized, unregularized], dim=0),
            torch.cat([self.regularized, regularized], dim=0),
            self.maximize
        )

    @classmethod
    def concat(cls, *assessments: BatchAssessment, maximize: bool=None):
        assert len(assessments) > 0, f"Number of assessments must be greater than 0."
        if maximize is None:
            maximize = assessments[0].maximize
        assessments = [a.to_maximize(maximize) for a in assessments]
        unregularized = [a.unregularized for a in assessments]
        regularized = [a.regularized for a in assessments]
        device = unregularized[0].device
        return PopulationBatchAssessment(
            torch.tensor(unregularized, device=device),
            torch.tensor(regularized, device=device),
            maximize=maximize
        )

    @classmethod
    def from_batch(cls, assessment: BatchAssessment, population_size: int=1):
        if population_size <= 0:
            raise RuntimeError(f"Size of population must be greater than 0 not {population_size}")
        
        new_size = torch.Size([population_size, assessment.batch_size // population_size, *assessment.shape[1:]])
        return PopulationBatchAssessment(
            assessment.unregularized.view(new_size),
            assessment.regularized.view(new_size),
            assessment.maximize
        )


def add_prev(cur, prev=None):

    if cur is not None and prev is not None:
        return cur + prev
    if cur is not None:
        return cur
    
    return prev

Result = typing.TypeVar('Result')

class Result(object):

    def __init__(self, x: torch.Tensor, maximize: bool):

        self.x = x

        self._outputs = []
        self._reg = BatchNullAssessment(x.dtype, x.device, maximize)
        self._y = x

    def update(self, y, result: Result=None):

        if result:
            self._outputs.append((y, result))
            self._reg = add_prev(result._reg, self._reg)
        else:
            self._outputs.append((y, None))

        self._y = y
        return self
    
    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, typing.Union[Result, None]]:
        return self._outputs[idx][0], self._outputs[idx][1]

    @property
    def y(self):
        return self._y
    
    # TODO: Depracate
    @property
    def regularization(self):
        return self._reg

    def add_reg(self, reg: BatchAssessment):
        self._reg += reg

    @property
    def reg(self):
        return self._reg

    @property
    def outputs(self):
        return self._outputs


# class Objective(ABC):
    
#     def minimize(self) -> bool:
#         return not self.maximize

#     @abstractproperty
#     def maximize(self) -> bool:
#         raise NotImplementedError

#     def assess(
#         self, x: torch.Tensor, t: torch.Tensor, full_output: bool=False
#     ) -> typing.Union[BatchAssessment, typing.Tuple[BatchAssessment, Result]]:
#         y, result = self.extend(x, t)
#         assessment = self.assess_output(y, t) + result
#         if full_output:
#             return assessment, result
#         return assessment

#     @abstractmethod
#     def assess_output(
#         self, y: torch.Tensor, t: torch.Tensor
#     ) -> BatchAssessment:
#         pass

#     @abstractmethod
#     def forward(
#         self, x: torch.Tensor, full_output: bool=False
#     ) -> (
#         typing.Tuple[torch.Tensor, Result]
#     ):
#         pass


class Recording(object):

    def __init__(self):

        self._df = pd.DataFrame()

    def record_inputs(self, name, data: typing.Dict):
        self._df.loc[len(self._df), ['Recorder', 'Type', *list(data.keys())]] = [name, 'Inputs', *list(data.values())]

    def record_theta(self, name, data: typing.Dict):
        self._df.loc[len(self._df), ['Recorder', 'Type', *list(data.keys())]] = [name, 'Theta', *list(data.values())]

    @property
    def df(self):
        return self._df


class Objective(ABC):

    @abstractproperty
    def differentiable(self) -> bool:
        pass

    @abstractproperty
    def maximize(self) -> bool:
        pass

    def minimize(self) -> bool:
        return not self.maximize

    def assess(self, x: torch.Tensor, t: torch.Tensor, regularize: bool=True) -> BatchAssessment:
        
        if regularize:
            y, result = self.forward(x, full_output=True)
            assessment = self.assess_output(y, t)
            return assessment + result.regularization

        y = self.forward(x)
        return self.assess_output(y, t)

    @abstractmethod
    def assess_output(self, y: torch.Tensor, t: torch.Tensor)-> BatchAssessment:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, full_output: bool=False) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, Result]]:
        pass


class Machine(Objective):

    def __init__(self, update_theta: bool=True):
        self._update_theta = update_theta

    def update_theta(self, to_update: bool=True):
        self._update_theta = to_update
        return self

    @abstractmethod
    def forward_update(self, x, t, outer: Objective=None):
        pass

    @abstractmethod
    def backward_update(self, x, t, result: Result=None, update_inputs: bool= True) -> torch.Tensor:
        pass
    
    def __call__(self, x):
        return self.forward(x)


class Score(object):

    @abstractproperty
    def maximize(self) -> bool:
        pass

    @property
    def minimize(self) -> bool:
        return not self.maximize
    
    @abstractmethod
    def score(self, x: torch.Tensor, t: torch.Tensor, reduce: bool= False) -> typing.Union[ScalarAssessment, BatchAssessment]:
        pass

    def __call__(self, x: torch.Tensor, t: torch.Tensor, reduce: bool=False) -> typing.Union[ScalarAssessment, BatchAssessment]:
        return self.score(x, t, reduce)


class LossReverse(Score):

    def __init__(self, loss: typing.Type[nn.Module]):
        self._loss = loss(reduction='none')
        self._score = TorchScore(loss)

    def maximize(self) -> bool:
        return False

    @abstractmethod
    def reverse(self, x: torch.Tensor, t: torch.Tensor, lr: float=1e-2) -> torch.Tensor:
        pass

    def score(self, x: torch.Tensor, t: torch.Tensor, reduce: bool= False) -> typing.Union[ScalarAssessment, BatchAssessment]:
        return self._score(x, t, reduce)


class MSELossReverse(LossReverse):

    def __init__(self):
        super().__init__(nn.MSELoss)
    
    def reverse(self, x: torch.Tensor, t: torch.Tensor, lr: float=1e-2) -> torch.Tensor:
        # if x - t == positive, dx = negative. vice versa
        return x + torch.sqrt(lr * self._loss(x, t)) * -torch.sign(x - t)


class GeneralLossReverse(LossReverse):

    def __init__(self, loss: typing.Type[nn.Module], maxiter: int=20):
        super().__init__(loss)

        # TODO: Use this
        self._maxiter = maxiter

    def reverse(self, x: torch.Tensor, t: torch.Tensor, lr: float=1e-2) -> torch.Tensor:
        # if x - t == positive, dx = negative. vice versa
        # 1) convert to numpy
        # 2) 

        # set up bounds
        # if x > t.. make t a lowerbound and x an upper bound
        # otherwise vice versa
        # ensure that 0 <= lr <= 1
        # use nelder mead
        # maxiter = 5pip in

        # This is the basics of how it should work... Need to test

        shape = x.shape
        x = x.flatten()
        t = t.flatten()
        t = t.detach().cpu()

        target_loss = (1 - lr) * self._loss(x, t)

        def objective(pt):
            # print('Shapes: ',  pt.shape,t.shape )
            result = ((target_loss - self._loss(torch.tensor(pt), t)) ** 2)
            print(result)
            return result.mean().item()
            
            # may need to compute the jacobian
            # pt = torch.tensor(pt)
            # pt = freshen(pt)
            # self._loss(pt, t).mean().backward()
            # grad = pt.grad.detach().cpu().numpy()
            # print(grad)
            # return grad

        lb = torch.min(x, t).detach().cpu().numpy()
        ub = torch.max(x, t).detach().cpu().numpy()
        
        bounds = sciopt.Bounds(lb, ub) 
        # find out how to set the maximium number of iterations
        x_prime = sciopt.minimize(objective, x.detach().cpu().numpy(), method='Powell', bounds=bounds).x
        
        return torch.tensor(x_prime, dtype=x.dtype, device=x.device).view(shape)


class Regularize(object):

    @abstractproperty
    def maximize(self) -> bool:
        pass

    @property
    def minimize(self) -> bool:
        return not self.maximize

    def __call__(self, x: torch.Tensor, reduce: bool=False):
        return self.score(x, reduce)

    @abstractmethod
    def score(self, x: torch.Tensor, reduce: bool=False):
        pass


class TorchScore(Score):

    def __init__(
        self, torch_scorer: typing.Type[nn.Module], 
        reduction: str='mean', maximize: bool=False
    ):
        assert reduction == 'mean' or reduction == 'sum'
        self.torch_loss = torch_scorer(reduction='none')
        self.reduction = reduction
        self._maximize = maximize
    
    @property
    def maximize(self):
        return self._maximize
    
    def score(self, x: torch.Tensor, t: torch.Tensor, reduce: bool=False):
        output = self.torch_loss(x, t)
        
        if self.reduction == 'mean' and not reduce:
            return output.view(x.size(0), -1).mean(1)
        elif self.reduction == 'mean':
            return output.mean()
        elif self.reduction == 'sum' and not reduce:
            return output.view(x.size(0), -1).sum(1)
        return output.sum()


class ThetaOptim(ABC):

    @abstractproperty
    def theta(self):
        pass

    @abstractmethod
    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective) -> ScalarAssessment:
        pass


class InputOptim(ABC):

    @abstractmethod
    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective) -> typing.Tuple[torch.Tensor, ScalarAssessment]:
        pass


class ParameterizedMachine(Machine):
    """Machine used for torch-based neural networks
    """

    @abstractproperty
    def theta(self):
        pass
    
    @theta.setter
    def theta(self, theta: torch.Tensor):
        pass


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


class ThetaOptimBuilder(ABC):

    @abstractmethod
    def __call__(self, net) -> ThetaOptim:
        pass


class InputOptimBuilder(ABC):

    @abstractmethod
    def __call__(self, net) -> InputOptim:
        pass


class SklearnThetaOptim(ThetaOptim):

    def __init__(self, module: SklearnModule, partial_fit: bool=False):
        self._partial_fit = partial_fit
        self._module = module
    
    @property
    def theta(self):
        return self._module

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, result: Result=None) -> ScalarAssessment:
        if self._partial_fit:
            self._module.partial_fit(x, t)
        else:
            self._module.fit(x, t)
        return objective.assess(x, t, True)


class SklearnOptimBuilder(ABC):

    @abstractmethod
    def __call__(self, net) -> SklearnThetaOptim:
        pass
