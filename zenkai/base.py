from abc import ABC, abstractmethod, abstractproperty
import typing
import torch
import pandas as pd
import torch.nn as nn
from .utils import batch_flatten


class Assessment:

    unregularized: typing.Optional[torch.Tensor] = None
    regularized: typing.Optional[torch.Tensor]=None
    maximize: bool=False

    def __init__(self, unregularized: torch.Tensor, regularized: typing.Optional[torch.Tensor]=None, maximize: bool=False):

        self.regularized = regularized
        self.unregularized = unregularized
        self.maximize = maximize
        if self.regularized is None:
            self.regularized = self.unregularized
        
        assert (self.regularized is None and self.unregularized is None) or (self.regularized.size() == self.unregularized.size())
    
    def is_null(self):
        return self.unregularized is None

    def to_maximize(self, maximize: bool=False):
        if self.maximize and maximize or (not self.maximize and not maximize):
            return self.__class__(self.unregularized, self.regularized, True)
        return self.__class__(-self.unregularized, -self.regularized, False)

    def __add__(self, other):
        
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
    
    def item(self):
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
    
    def sum(self):
        return ScalarAssessment(
            self.unregularized.mean(),
            self.regularized.mean()
        )

    def __add__(self, other):

        if isinstance(other, BatchNullAssessment):
            return self
        
        return super().__add__(self, other)


class BatchNullAssessment(Assessment):

    def __init__(self, dtype: torch.dtype, device):
        super().__init__(None)
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

    def __add__(self, other):

        if isinstance(other, BatchAssessment):
            return other
        
        return BatchNullAssessment(self.dtype, self.device)

    def __mul__(self, value):
        
        return BatchNullAssessment(self.dtype, self.device)


class PopulationAssessment(Assessment):
    
    def best(self, for_reg: bool=True):

        if for_reg:
            x = self.regularized
        else:
            x = self.unregularized

        if self.maximize:
            result = torch.max(x, dim=0)
        else: result = torch.min(x, dim=0)
        return x[result[1]], result[0] 
    
    def append(self, assessment: ScalarAssessment):
        assessment = assessment.to_maximize(self.maximize)
        unregularized = assessment.unregularized.unsqueeze(0)
        regularized = assessment.regularized.unsqueeze(0)
        
        return PopulationBatchAssessment(
            torch.concat([self.unregularized, unregularized], dim=0),
            torch.concat([self.regularized, regularized], dim=0),
            self.maximize
        )
    
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
        unregularized = [a.unregularized for a in assessments]
        regularized = [a.regularized for a in assessments]
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

    def append(self, assessment: BatchAssessment):

        assessment = assessment.to_maximize(self.maximize)
        unregularized = assessment.unregularized.unsqueeze(0)
        regularized = assessment.regularized.unsqueeze(0)
        return PopulationBatchAssessment(
            torch.concat([self.unregularized, unregularized], dim=0),
            torch.concat([self.regularized, regularized], dim=0),
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
        
        new_size = torch.Size([population_size, assessment.batch_size // population_size, *assessment.regularized.size[1:]])
        return PopulationBatchAssessment(
            assessment.unregularized.view(new_size),
            assessment.regularized.view(new_size),
            assessment.maximize
        )


class Objective(ABC):
    
    def minimize(self) -> bool:
        return not self.maximize

    @abstractproperty
    def maximize(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def assess(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, 
        batch_assess: bool=True
    ) -> typing.Union[ScalarAssessment, BatchAssessment]:
        pass

    @abstractmethod
    def extend(
        self, x: torch.Tensor, t: torch.Tensor, 
        y: torch.Tensor=None, batch_assess: bool=True
    ) -> (
        typing.Tuple[torch.Tensor, typing.Union[ScalarAssessment, BatchAssessment]]
    ):
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


class Machine(ABC):

    def __init__(self):
        self._fixed = False

    @abstractmethod
    def assess(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, reg_excitations: bool=False, get_ys: bool=False):
        pass

    @abstractmethod
    def forward_update(self, x, t, objective: Objective=None, update_theta: bool=True):
        pass

    @abstractmethod
    def backward_update(self, x, t, ys=None, update_theta: bool=True, update_inputs: bool= True) -> torch.Tensor:
        pass

    def fix(self, fixed: bool=True):
        self._fixed = fixed

    @abstractmethod
    def get_y_out(self, outs):
        pass
    
    def __call__(self, x):
        return self.forward(x)


class MachineObjective(Objective):

    def __init__(self, machine: Machine):
        self._machine = machine
    
    def assess(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, 
        batch_assess: bool=True
    ) -> typing.Union[ScalarAssessment, BatchAssessment]:
        return self._machine.assess(
            x, t, y, batch_assess, **self._assess_args
        )

    def extend(
        self, x: torch.Tensor, t: torch.Tensor, 
        y: torch.Tensor=None, batch_assess: bool=True
    ) -> (
        typing.Tuple[torch.Tensor, typing.Union[ScalarAssessment, BatchAssessment]]
    ):
        if y is None:
            y = self._machine.forward(
                x
            )
            
        return y, BatchNullAssessment(x.dtype, x.device)


class ObjectivePair(Objective):

    def __init__(self, first: Objective, second: Objective):
        self.first = first
        self.second = second
    
    def assess(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, 
        batch_assess: bool=True
    ) -> typing.Union[ScalarAssessment, BatchAssessment]:
        y, assessment = self.first.extend(
            x, t, y, batch_assess
        )
        return self.second.assess(
            y, t, batch_assess=batch_assess
        ) + assessment

    def extend(
        self, x: torch.Tensor, t: torch.Tensor, 
        y: torch.Tensor=None, batch_assess: bool=True
    ) -> (
        typing.Tuple[torch.Tensor, typing.Union[ScalarAssessment, BatchAssessment]]
    ):        
        y, assessment1 = self.first.extend(
            x, t, y, batch_assess
        )
        y, assessment2 = self.second.assess(
            y, t, batch_assess=batch_assess
        )
        return y, assessment1 + assessment2


class Result(object):

    def __init__(self, x, get_ys, get_reg):

        self._get_ys = get_ys
        self._get_reg = get_reg
        if get_ys:
            self._ys = [x]
        else:
            self._ys = None
        self._reg = None
        self._y = x
    
    def update(self, output):

        if self._get_ys:
            self._ys.append(output[1])
        if self._get_reg and self._get_ys:
            reg = output[2]
        elif self._get_reg:
            reg = output[1]
        
        if reg and self._reg:
            self._reg += reg
        elif reg:
            self._reg = reg
        self._y = output[0]

    @property
    def y(self):
        return self._y
    
    @property
    def output(self):
        if self._get_reg and self._get_ys:
            return self._y, self._ys, self._reg
        elif self._get_reg:
            return self._y, self._reg
        elif self._get_ys:
            return self._y, self._ys
        return self._y


class Score(object):

    @abstractproperty
    def maximize(self) -> bool:
        pass

    @property
    def minimize(self) -> bool:
        return not self.maximize

    @abstractmethod
    def __call__(self, x: torch.Tensor, t: torch.Tensor):
        pass


class Regularize(object):

    @abstractproperty
    def maximize(self) -> bool:
        pass

    @property
    def minimize(self) -> bool:
        return not self.maximize

    @abstractmethod
    def __call__(self, x: torch.Tensor):
        pass



class TorchScore(Score):

    def __init__(self, torch_scorer: typing.Type[nn.Module], batch_assess: bool, reduction: str='mean', maximize: bool=False):

        assert reduction == 'mean' or reduction == 'sum'
        self.torch_loss = torch_scorer(reduction='none')
        self.batch_assess = batch_assess
        self.reduction = reduction
        self._maximize = maximize
    
    @property
    def maximize(self):
        return self._maximize
    
    def __call__(self, x: torch.Tensor, t: torch.Tensor):
        output = self.torch_loss(x, t)
        
        if self.reduction == 'mean' and self.batch_assess:
            return output.view(x.size(0), -1).mean(1)
        elif self.reduction == 'mean':
            return output.mean()
        elif self.reduction == 'sum' and self.batch_assess:
            return output.view(x.size(0), -1).sum(1)
        return output.sum()



class ThetaOptim(ABC):

    @abstractproperty
    def theta(self):
        pass

    @abstractmethod
    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, y: torch.Tensor=None) -> ScalarAssessment:
        pass


class InputOptim(ABC):

    @abstractmethod
    def step(self, x: torch.Tensor, t: torch.Tensor,objective: Objective, y: torch.Tensor=None) -> typing.Tuple[torch.Tensor, ScalarAssessment]:
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


class Processor(ABC):

    @abstractmethod
    def forward(self, x, get_ys: bool=False):
        pass

    @abstractmethod
    def backward(self, x, t, ys=None):
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


class InputRecorder(InputOptim):

    def __init__(self, name: str, optim: InputOptim, recording: Recording=None):

        self._name = name
        self._optim = optim
        self._recording = recording or Recording()
    
    @abstractmethod
    def record(self, x: torch.Tensor, x_prime: torch.Tensor, evaluation: Assessment):
        pass

    def step(self, x: torch.Tensor, t: torch.Tensor,objective: Objective, y: torch.Tensor=None) -> typing.Tuple[torch.Tensor, ScalarAssessment]:
        x_prime, assessment = self._optim.step(
            x, t, objective, y
        )
        self.record(x, x_prime, assessment)
        return x_prime, assessment

    @property
    def recording(self):
        return self._recording


class ThetaRecorder(InputOptim):

    def __init__(self, name: str, optim: ThetaOptim, recording: Recording=None):

        self._name = name
        self._optim = optim
        self._recording = recording or Recording()
    
    @abstractmethod
    def record(self, theta: torch.Tensor, theta_prime: torch.Tensor, assessment: Assessment):
        pass

    def step(self, x: torch.Tensor, t: torch.Tensor, objective: Objective, y: torch.Tensor=None) -> ScalarAssessment:
        theta = self._optim.theta
        x_prime, asessment = self._optim.step(
            x, t, objective, y
        )
        self.record(theta, self._optim.theta, asessment)
        return x_prime, asessment

    @property
    def recording(self):
        return self._recording
