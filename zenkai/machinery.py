

from abc import ABC, abstractclassmethod, abstractmethod
from functools import partial, singledispatchmethod
import statistics
import typing
import sklearn
import torch as th
import torch.nn as nn

from abc import ABC, abstractproperty
from dataclasses import dataclass
import torch as th
import numpy as np
import torch.nn.utils as nn_utils
from copy import deepcopy
import pandas as pd


torch_id = "torch"
numpy_id = "numpy"
null_id = "null"


class Reduction(ABC):
    
    @abstractmethod
    def __call__(self, evaluation):
        raise NotImplementedError


class MeanReduction(ABC):
    
    def __call__(self, grade):
        
        return grade.view(grade.size(0), -1).mean(dim=1)


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
    

class DType(object):

    @abstractproperty
    def dtype(self):
        raise NotImplementedError


class THDType(DType):

    def __init__(self, dtype: th.dtype):
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def id(self):
        return torch_id


class NPDType(DType):

    def __init__(self, dtype: np.dtype):
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def id(self):
        return "numpy"


class Shape(object):

    def __init__(self, *args: int):
        self._shape = args

    def dim(self):
        return len(self._shape)

    @classmethod
    def from_torch(self, size: th.Size):
        return Shape(*size)

    @classmethod
    def from_numpy(self, size: np.shape):
        return Shape(*size)


class Device(ABC):

    @abstractproperty
    def id(self) -> str:
        raise NotImplementedError


class THDevice(Device):

    def __init__(self, device):
        self._device = device

    @property
    def device(self):
        return self._device

    @property
    def id(self) -> str:
        return torch_id


class NullDevice(Device):

    @property
    def id(self) -> str:
        return null_id


@dataclass
class Port(object):

    dtype: DType
    shape: Shape
    device: Device


class Scorer(ABC):

    @abstractmethod
    def assess(self, x: th.Tensor) -> th.Tensor:
        pass

    @abstractproperty
    def maximize(self):
        pass


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
    def forward_update(self, x, t, scorer: Scorer=None, update_theta: bool=True, recorder: Recorder=None):
        pass

    @abstractmethod
    def backward_update(self, x, t, outs=None, update_theta: bool=True, update_inputs: bool= True, recorder: Recorder=None):
        pass

    @abstractmethod
    def get_y_out(self, outs):
        pass

    # def update(self, x, t, outs=None):
    #     self.backward(x, t, outs, True)    
        
    # def update_x(self, x, t, outs=None):
    #     return self.backward_update(x, t, outs, False) 
    
    def __call__(self, x):
        return self.forward(x)



class Optimizer(ABC):

    @abstractproperty
    def reset_inputs(self, inputs):
        pass

    @abstractproperty
    def reset_theta(self, theta):
        pass

    @abstractmethod
    def update_theta(self, t, inputs=None, theta=None, y=None, scorer: Scorer=None):
        pass
    
    @abstractmethod
    def update_inputs(self, t, inputs=None, theta=None, y=None, scorer: Scorer=None):
        pass

    @abstractproperty
    def inputs(self):
        pass
    
    @abstractproperty
    def theta(self):
        pass

    @abstractproperty
    def evaluations(self):
        pass


class TorchOptimizer(Optimizer):

    def __init__(self, net: nn.Module, objective: Objective):
        super().__init__()
        self._objective = objective
        self._net = net

    def update_theta(self, t, inputs=None, theta=None, y=None, scorer: Scorer=None):
        if inputs is not None:
            self.reset_inputs(inputs)
        if theta is not None:
            self.reset_theta(theta)
        
        self._evaluations = self._update_theta(t, y, scorer)

    def update_inputs(self, t, inputs=None, theta=None, y=None, scorer: Scorer=None):
        if inputs is not None:
            self.reset_inputs(inputs)
        if theta is not None:
            self.reset_theta(theta)
        
        self._evaluations = self._update_inputs(t, y, scorer=None)
    
    @abstractmethod
    def _update_inputs(self, t, y=None, scorer: Scorer=None):
        pass

    @abstractmethod
    def _update_theta(self, t, y=None, scorer: Scorer=None):
        pass

    @property
    def evaluations(self):
        return self._evaluations


class GradOptimizer(TorchOptimizer):

    def __init__(self, net: nn.Module, objective: Objective, optim):
        super().__init__(net, objective)
        self._net = net
        self._optim: th.optim.Optimizer = optim(self._net.parameters())

    def reset_theta(self, theta=None):
        self._theta = theta
        nn_utils.vector_to_parameters(theta, self._net.parameters())
        self._evaluation = None

    def reset_inputs(self, inputs):
        self._inputs = inputs
        self._evaluation = None

    def _update_theta(self, t, y=None, scorer: Scorer=None):
        self._optim.zero_grad()
        if y is None:
            y = self._net(self._inputs)
        if scorer is None:
            evaluation = self._objective(y, t)
        else:
            evaluation = scorer.assess(y)
        evaluation.mean().backward()
        self._optim.step()
        return evaluation.view(1, -1)

    def _update_inputs(self, t, y=None, scorer: Scorer=None):
        
        if y is None:
            y = self._net(self._inputs)

        if scorer is None:
            evaluation = self._objective(y, t)
        else:
            evaluation = scorer.assess(y)
        if self._inputs.grad is None:
            evaluation.sum().backward()
        if self._inputs.grad is None:
            raise RuntimeError("Input is not a leaf node or does not retain grad")
        self._inputs = self._inputs - self._inputs.grad
        return evaluation.view(1, -1)

    @property
    def inputs(self):
        return self._inputs
    
    @property
    def theta(self):
        return nn.utils.parameters_to_vector(self._net.parameters())
    

class XPOptimizer(Optimizer):

    def __init__(self, x_updater: Optimizer, p_updater: Optimizer):
        super().__init__()
        self._x_updater: Optimizer = x_updater
        self._p_updater: Optimizer = p_updater
        self._updated_inputs = False
        self._updated_theta = False

    def reset_theta(self, theta=None):
        self._x_updater.reset_theta(theta)
        self._p_updater.reset_theta(theta)

    def reset_inputs(self, inputs):
        self._x_updater.reset_inputs(inputs)
        self._p_updater.reset_inputs(inputs)

    def update_theta(self, t, inputs=None, theta=None, y=None, scorer: Scorer=None):
        if inputs is not None:
            self._x_updater.reset_inputs(inputs)
        if theta is not None:
            self._x_updater.reset_theta(theta)
        self._p_updater.update_theta(t, theta=theta, inputs=inputs, y=y, scorer=scorer)
        self._x_updater.reset_theta(self._p_updater.theta)
        self._updated_inputs = False
        self._updated_theta = True

    def update_inputs(self, t, inputs: th.Tensor=None, theta=None, y=None, scorer: Scorer=None):
        if inputs is not None:
            self._p_updater.reset_inputs(inputs)
        if theta is not None:
            self._p_updater.reset_theta(theta)
        self._x_updater.update_inputs(t, theta=theta, inputs=inputs, y=y, scorer=scorer)
        self._p_updater.reset_inputs(self._x_updater.inputs)
        self._updated_inputs = True
        self._updated_theta = False

    @property
    def inputs(self):
        return self._x_updater.inputs
    
    @property
    def theta(self):
        return self._p_updater.theta
    
    @property
    def evaluations(self):
        if self._updated_inputs:
            return self._x_updater.evaluations
        if self._updated_theta:
            return self._p_updater.evaluations


class HillClimberProcessor(ABC):

    @abstractmethod
    def __call__(self, x):
        pass
    
    @abstractmethod
    def spawn(self):
        pass
    
    @abstractmethod
    def report(self, evaluations):
        pass


def get_best(x: th.Tensor, evaluations, maximize: bool=True):
    if maximize:
        idx = th.argmax(evaluations)
    else:
        idx = th.argmin(evaluations)
    return x[idx]


class StepHillClimberProcessor(HillClimberProcessor):

    def __init__(self, s_mean: int=-2, s_std=1, k: int=1, momentum: float=None, maximize: bool=True, update_after: int=400):
        self.s_mean = s_mean
        self.s_std = s_std

        self.k = k
        self.maximize = maximize
        self._momentum = momentum
        self._diff = None
        self._x_updated = None
        self._out = None
        self._s = None
        self._step_evaluations = {}
        self._keep_s = 100
        self._i = 0
        self._update_after = update_after
        self._mean_offset = 0.
    
    def reset_state(self):
        self._diff = None
        self._x_updated = None
        self._out = None
        self._s = None

    def __call__(self, x: th.Tensor):
        
        self._x = x
        self._s = th.randn(self.k)  * self.s_std + self.s_mean
        s = (10 ** self._s).view(-1, *([1] * x.dim())).repeat(1, *x.size())
        y = th.randn(self.k, *x.size()) * s + x
        self._out = th.cat([x.unsqueeze(0), y])
        return self._out
    
    def report(self, evaluations: th.Tensor):

        self._step_evaluations = {
            k: v for i, (k, v) in enumerate(
                sorted(self._step_evaluations.items(), key=lambda item: item[1], reverse=self.maximize)
            ) if i < self._keep_s
        }
        self._step_evaluations.update(dict(zip(self._s.tolist(), evaluations[1:].tolist())))

        self.s_mean = statistics.mean(self._step_evaluations.keys())
        # print(self.s_mean)
        if len(self._step_evaluations) > 2:
           self.s_std = statistics.stdev(self._step_evaluations.keys())
        
        self.s_mean += self._mean_offset

        if self._out is None:
            raise RuntimeError('Have not processed any inputs yet (i.e. use __call__)')
        best = get_best(self._out, evaluations, self.maximize)
        if self._diff is not None and self._momentum is not None:
            self._diff = (1 - self._momentum) * (best - self._x) + self._momentum * self._diff
            x_updated = self._x + self._diff
        elif self._momentum is not None:
            self._diff = (best - self._x)
            x_updated = self._x + self._diff
        else:
            x_updated = best
        self._i += 1
        # if self._i % self._update_after == 0:
        #     self._mean_offset -= 0.5
        return x_updated

    def spawn(self, maximize: bool=None):
        return StepHillClimberProcessor(
            self.s_mean, self.s_std, self.k, self._momentum, self.maximize if maximize is None else maximize
        )


class HillClimberOptimizer(TorchOptimizer):

    def __init__(self, net: nn.Module, objective: Objective, processor: typing.Union[str, StepHillClimberProcessor]="gaussian"):
        super().__init__(net, objective)
        if processor == "gaussian":
            self.input_processor = StepHillClimberProcessor(maximize=objective.maximize)
            self.input_processor = StepHillClimberProcessor(maximize=objective.maximize)
        else:
            self.input_processor = processor.spawn(maximize=objective.maximize)
            self.theta_processor = processor.spawn(maximize=objective.maximize)
        self._inputs_diff = None
        self._theta_diff = None
        self._x = None
        self._theta = nn_utils.parameters_to_vector(self._net.parameters())

    def reset_inputs(self, inputs):
        self._inputs = inputs
        self._evaluation = None
        self._inputs_diff = None
        self.input_processor.reset_state()

    def reset_theta(self, theta):
        self._theta = theta
        nn_utils.vector_to_parameters(theta, self._net.parameters())
        self._evaluation = None
        self._theta_diff = None
        self.theta_processor.reset_state()

    def _update_theta(self, t, y=None, scorer: Scorer=None):
        theta = self.theta_processor(self._theta)
        evaluations = []
        for theta_i in theta:
            nn_utils.vector_to_parameters(theta_i, self._net.parameters())
            y = self._net(self._inputs)
            if scorer is None:
                evaluations.append(self._objective.forward(y, t))
            else:
                evaluations.append(scorer.assess(y))
        evaluations = th.stack(evaluations)
        self._theta = self.theta_processor.report(evaluations)
        return evaluations
    
    def _update_inputs(self, t, y=None, scorer: Scorer=None):
        inputs = self.input_processor(self._inputs)
        y = self._net(inputs)
        if scorer is None:
            evaluations = self._objective.forward_multi(y, t)
        else:
            evaluations = scorer.assess_multi(y, t)
        self._inputs = self.input_processor.report(evaluations)
        return evaluations

    @property
    def inputs(self):
        return self._inputs
    
    @property
    def theta(self):
        return nn_utils.parameters_to_vector(self._net.parameters())

    @property
    def evaluations(self):
        return self._evaluations


class RepeatOptimizer(Optimizer):

    def __init__(self, sub_optimizer: TorchOptimizer):
        super().__init__()
        self._sub_optimizer = sub_optimizer
        self._evaluations = None
    
    def reset_inputs(self, inputs):
        self._sub_optimizer.reset_inputs(inputs)

    def reset_theta(self, theta):
        self._sub_optimizer.reset_theta(theta)

    @abstractmethod
    def update_theta(self, t, inputs=None, theta=None, y=None):
        pass

    @abstractmethod
    def update_inputs(self, t, inputs=None, theta=None, y=None):
        pass

    @property
    def inputs(self):
        return self._sub_optimizer.inputs
    
    @property
    def theta(self):
        return self._sub_optimizer.theta

    @property
    def evaluations(self):
        return self._evaluations


class NRepeatOptimizer(RepeatOptimizer):

    def __init__(self, sub_optimizer: TorchOptimizer, n: int):
        super().__init__(sub_optimizer)
        self._n = n
    
    def update_theta(self, t, inputs=None, theta=None, y=None, scorer: Scorer=None):
        evaluations = []
        for _ in range(self._n):
            self._sub_optimizer.update_theta(t, inputs, theta, y, scorer=scorer)
            evaluations.append(self._sub_optimizer.evaluations)
        self._evaluations = evaluations

    def update_inputs(self, t, inputs=None, theta=None, y=None, scorer: Scorer=None):
        evaluations = []
        for _ in range(self._n):
            self._sub_optimizer.update_inputs(t, inputs, theta, y, scorer=scorer)
            evaluations.append(self._sub_optimizer.evaluations)
        self._evaluations = evaluations


class SKOptimBuilder(object):

    def __call__(self, machines) -> Optimizer:
        pass


class THOptimBuilder(ABC):

    @abstractmethod
    def __call__(self, net, loss) -> Optimizer:
        pass


class SingleOptimBuilder(object):

    def __init__(self):
        
        self._optim = None
        self.grad(lr=1e-2)
        self.n_repetitions = 1

    def grad(self, optim_cls=None, **kwargs):

        if optim_cls is None:
            optim_cls = th.optim.Adam

        self._optim = partial(
            GradOptimizer,
            optim=partial(optim_cls, **kwargs)
        )
        return self
    
    def step_hill_climber(self, momentum=0.5, mean: float=-1e-2, std: float=1, update_after: int=400):

        processor = StepHillClimberProcessor(k=16, momentum=momentum, s_mean=mean, s_std=std, update_after=update_after)
        self._optim = partial(HillClimberOptimizer, processor=processor)
        return self
    
    def repeat(self, n_repetitions: int):

        if n_repetitions <= 0:
            raise RuntimeError("Repetitions must be greater than or equal to 1")
        self.n_repetitions = n_repetitions
        return self

    def __call__(self, net, loss) -> Optimizer:
        if self.n_repetitions > 1:
            return NRepeatOptimizer(self._optim(net, loss), self.n_repetitions)
        else:
            return self._optim(net, loss)


class THXPBuilder(object):

    def __init__(self, x_builder: THOptimBuilder, p_builder: THOptimBuilder):
        
        self._x_builder = x_builder
        self._p_builder = p_builder

    def __call__(self, net, loss):

        x_updater = self._x_builder(net, loss)
        p_updater = self._p_builder(net, loss)
        return XPOptimizer(x_updater, p_updater)

def to_float(x: typing.List[th.Tensor]):
    return list(map(lambda xi: xi.mean().item(), x))


class TorchNN(Machine):

    def __init__(self, module: nn.Module, objective: Objective, updater: THOptimBuilder=None, fixed: bool=False):
        
        self._module = module
        self._objective = objective
        self._updater = updater(module, objective) or GradOptimizer(module, objective)
        self._fixed = fixed

    def assess(self, y, t):
        return self._objective(y, t)

    def fix(self, fixed: bool=True):
        self._fixed = fixed

    def output_ys(self, x):
        x = x.detach().requires_grad_()
        x.retain_grad()
        y = self.forward(x)
        return y, [x, y]

    def get_y_out(self, outs):
        return outs[1]

    def get_in(self, outs):
        return outs[0]

    def forward(self, x):
        # x = x.detach().requires_grad_()
        # x.retain_grad()
        return self._module.forward(x)
    
    def forward_update(self, x, t, scorer: Scorer=None, update_theta: bool=True, recorder: Recorder=None):

        if update_theta and not self._fixed:
            self._updater.update_theta(t, x, scorer=scorer)
            if recorder:
                recorder.record_theta(
                    id(self), nn_utils.parameters_to_vector(self._module.parameters()), self._updater.theta,
                    to_float(self._updater.evaluations)
                )
            nn_utils.vector_to_parameters(self._updater.theta, self._module.parameters())
        
        y = self.forward(x)
        return y

    def backward_update(self, x, t, outs=None, update_theta: bool=True, update_inputs: bool= True, recorder: Recorder=None):
        if outs is not None:
            y = self.get_y_out(outs)
            x = self.get_in(outs)
        else:
            x = x.detach().requires_grad_()
            x.retain_grad()
            y = self._module(x)

        if update_theta and not self._fixed:
            self._updater.update_theta(t, y=y, inputs=x)
        
            if recorder:
                recorder.record_theta(
                    id(self), nn_utils.parameters_to_vector(self._module.parameters()), 
                    self._updater.theta,
                    evaluations=to_float(self._updater.evaluations)
                )
            nn_utils.vector_to_parameters(self._updater.theta, self._module.parameters())
        
        if update_inputs:
            self._updater.update_inputs(t, y=y, inputs=x)
            
            if recorder:
                recorder.record_inputs(
                    id(self), x, self._updater.inputs,
                    evaluations=to_float(self._updater.evaluations)
                )
            return self._updater.inputs

    @property
    def module(self):
        return self._module


class SklearnMachine(Machine):

    def __init__(self, machines, loss, updater: SKOptimBuilder=None, fixed: bool=False):
        super().__init__()
        self._machines = machines
        self._loss = loss
        self._updater = updater(machines)
        self._fixed = fixed

    def fix(self, fixed: bool=True):
        self._fixed = fixed

    def assess(self, y, t):
        # # TODO: Check
        # if y is None:
        #     y = self.forward(x)
        # else:
        #     outs = outs
        return self._loss(y, t)

    def output_ys(self, x):
        y = self.forward(x)
        return y, [x, y]

    def forward(self, x: th.Tensor):
        device = x.device
        x_np = x.detach().cpu().numpy()
        y_np = np.stack([machine.predict(x_np) for machine in self._machines])
        return th.tensor(y_np, device=device)

    def forward_update(self, x, t, scorer: Scorer=None, update_theta: bool=True):

        if update_theta:
            self._updater.update_theta(t, x, scorer=scorer)
        y = self.forward(x)
        return y
        
    def get_y_out(self, outs):
        return outs[1]

    def get_in(self, outs):
        return outs[0]
    
    def backward_update(self, x, t, outs=None, update_theta: bool=True, update_inputs: bool= True, recorder: Recorder=None):

        if update_theta and not self._fixed:
            self._updater.update_theta(t, inputs=x)
        
        if update_inputs:
            self._updater.update_inputs(t, inputs=x)
            return self._updater.inputs


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


class BlackboxOptimBuilder:

    def __init__(self):
        
        self._optim = None
        self.step_hill_climber()
        self.n_repetitions = 1

    def step_hill_climber(self, momentum=0.5, mean: float=-1e-2, std: float=1, update_after: int=400):

        processor = StepHillClimberProcessor(k=16, momentum=momentum, s_mean=mean, s_std=std, update_after=update_after)
        self._optim = partial(HillClimberOptimizer, processor=processor)
        return self
    
    def repeat(self, n_repetitions: int):

        if n_repetitions <= 0:
            raise RuntimeError("Repetitions must be greater than or equal to 1")
        self.n_repetitions = n_repetitions
        return self

    def __call__(self, net: Blackbox, loss) -> Optimizer:
        if self.n_repetitions > 1:
            return NRepeatOptimizer(self._optim(net, loss), self.n_repetitions)
        else:
            return self._optim(net, loss)


class BlackboxMachine(Machine):

    def __init__(self, f, loss, input_updater: BlackboxOptimBuilder=None):
        super().__init__()
        self._f = f
        self._input_updater = input_updater(f, loss)
        self._loss = loss

    def assess(self, x, t, outs=None):
        if y is None:
            y = self.forward(x)
        else:
            outs = outs
        return self._loss(y, t)

    def output_ys(self, x):
        y = self.forward(x)
        return y, [x, y]

    def forward(self, x: th.Tensor):
        return self._f(x)

    def forward_update(self, x, t, scorer: Scorer=None, update_theta: bool=True, recorder: Recorder=None):
        y = self.forward(x)
        return y

    def get_y_out(self, outs):
        return outs[1]

    def get_in(self, outs):
        return outs[0]
    
    def backward_update(self, x, t, outs=None, update_theta: bool=True, update_inputs: bool= True, recorder: Recorder=None):
        
        if update_inputs:
            self._input_updater.update_inputs(t, inputs=x)
            return self._input_updater.inputs


class SequenceScorer(Scorer):

    def __init__(self, machines: typing.List[Machine], t: th.Tensor, outer: Scorer=None):

        if len(machines) == 0:
            raise RuntimeError("The number of machines must be greater than 0.")
        self._machines = machines
        self._outer = outer
        self._t = t

    def assess(self, x):
        for machine in self._machines:
            x = machine.forward(x)
        if self._outer:
            return self._outer.assess(x)
        return self._machines[-1].assess(x, self._t)

    @property
    def maximize(self):

        return self._machines[-1].maximize


class Sequence(Machine):

    def __init__(self, machines: typing.List[Machine]):
        if len(machines) == 0:
            raise ValueError(f'Length of sequence must be greater than 0')
        self.machines = machines
    
    def assess(self, y, t):
        
        return self.machines[-1].assess(
            y, t
        )

    def get_y_out(self, outs):
        return self.machines[-1].get_y_out(outs[-1]) 

    def get_in(self, outs):
        return outs[0]

    def output_ys(self, x):
        outs = [x]
        y = x
        for machine in self.machines:
            y, outs_i = machine.output_ys(y)
            outs.append(outs_i)
        return y, outs

    def forward(self, x):
        y = x
        for layer in self.machines:
            y = layer.forward(y)
        return y

    def forward_update(self, x, t, scorer: Scorer=None, update_theta: bool=True, recorder: Recorder=None):
        if not update_theta:
            y = self.forward(x)
            return y

        y = x
        for i, machine in enumerate(self.machines):
            if i < len(self.machines) - 1:
                cur_scorer = SequenceScorer(self.machines[i + 1:], t, scorer)
            else:
                cur_scorer = scorer
            y = machine.forward_update(y, t, cur_scorer, update_theta, recorder)

        return y

    def backward_update(self, x, t, outs=None, update_theta: bool=True, update_inputs: bool= True, recorder: Recorder=None):
        if outs is None:
            _, outs = self.output_ys(x)
        
        xs = [x]
        for y_i, machine in zip(outs[1:-1], self.machines[:-1]):
            xs.append(machine.get_y_out(y_i))

        for i, (x_i, y_i, machine) in enumerate(zip(reversed(xs), reversed(outs[1:]), reversed(self.machines))):
            _update_inputs = i < len(xs) or update_inputs
            t = machine.backward_update(x_i, t, y_i, update_theta, _update_inputs, recorder)
        return t


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


class TH2NP(Processor):

    def output_ys(self, x):
        y = self.forward(x)
        return y, y

    def forward(self, x):
        return x.detach().cpu().numpy()

    def backward(self, x, t, outs=None):
        return th.from_numpy(t)

    def get_y_out(self, outs):
        return outs


class NP2TH(Processor):

    def __init__(self, dtype: th.dtype=th.float32):
        self.dtype = dtype

    def output_ys(self, x):
        y = self.forward(x)
        return y, y

    def forward(self, x):
        return th.tensor(x, dtype=self.dtype)
        # return th.from_numpy(x)

    def backward(self, x, t, outs=None):
        return t.detach().cpu().numpy()

    def get_y_out(self, outs):
        return outs


class CompositeProcessor(Processor):

    def __init__(self, processors: typing.List[Processor]):

        self.processors = processors

    def output_ys(self, x):
        outs = [x]
        y = x
        for processor in self.processors:
            y, outs_i = processor.output_ys(y)
            outs.append(outs_i)
        return y, outs

    def forward(self, x):
        y = x
        for processor in self.processors:
            y = processor.forward(y)
        return y

    def backward(self, x, t, outs=None):

        if outs is None:
            _, outs = self.output_ys(x)
        
        xs = outs[:-1]
        for x_i, y_i, processor in zip(
            reversed(xs), reversed(outs[1:-1]), reversed(self.processors)):
            t = processor.backward(x_i, t, y_i)
        return t

    def get_y_out(self, outs):
        if len(self.processors) == 0:
            return outs[0]
        return self.processors[-1].get_y_out(outs[-1])


class Processed(Machine):
    
    def __init__(
        self, processors: typing.List[Processor], 
        machine: Machine
    ):
        self.processors = CompositeProcessor(processors)
        self.machine = machine
    
    def assess(self, y, t):
        return self.machine.assess(y, t)
    
    def output_ys(self, x):
        y, outs_i = self.processors.output_ys(x)
        # y = outs[-1] if outs[-1] is not None else x
        y, outs_j = self.machine.output_ys(y)
        return y, [outs_i, outs_j]

    def forward(self, x):
        y = self.processors.forward(x)
        return self.machine.forward(y)
    
    def forward_update(self, x, t, scorer: Scorer=None, update_theta: bool=True, recorder: Recorder=None):
        x = self.processors.forward(x)
        return self.machine.forward_update(x, t, scorer, update_theta, recorder)

    def backward_update(self, x, t, outs=None, update_theta: bool=True, update_inputs: bool= True, recorder: Recorder=None):
        if outs is None:
            _, outs = self.output_ys(x)

        y_in = self.processors.get_y_out(outs[0])
        t = self.machine.backward_update(y_in, t, outs[1], update_theta, update_inputs, recorder)
        if update_inputs:
            return self.processors.backward(x, t, outs[0])

    def get_y_out(self, outs):
        return self.machine.get_y_out(outs[-1])


class WeightedLoss(nn.Module):
    
    def __init__(self, nn_loss: nn.Module, weight: float):

        super().__init__()
        self.nn_loss = nn_loss
        self.weight = weight

    def forward(self, x, t):

        return self.weight * self.nn_loss(x, t)



class EuclidRecorder(Recorder):

    def __init__(self):
        
        self._layer_results = {}
        self._cur_index = 0
        self._layer_map = {}
        self._layer_count = 0

    def adv(self):
        self._cur_index += 1
    
    def get_or_set_layer_id(self, layer):
        if layer not in self._layer_map:
            self._layer_map[layer] = self._layer_count
            self._layer_count += 1
        return self._layer_map[layer]

    def record_inputs(self, layer, prev_inputs, cur_inputs, evaluations):

        layer = self.get_or_set_layer_id(layer)
        if layer not in self._layer_results:
            self._layer_results[layer] = []
        if self._cur_index not in self._layer_results[layer]:
            self._layer_results[layer] += [{}] * (self._cur_index - len(self._layer_results[layer]) + 1)
        deviation = th.sqrt(th.sum((prev_inputs - cur_inputs) ** 2)).item()
        self._layer_results[layer][self._cur_index]['inputs'] = dict(
            deviation=deviation,
            evaluations=evaluations
        )

    def record_theta(self, layer, prev_theta, cur_theta, evaluations):

        layer = self.get_or_set_layer_id(layer)
        if layer not in self._layer_results:
            self._layer_results[layer] = []
        if len(self._layer_results[layer]) <= self._cur_index:

            self._layer_results[layer] += [{}] * (self._cur_index - len(self._layer_results[layer]) + 1)
        deviation = th.sqrt(th.sum((prev_theta - cur_theta) ** 2)).item()
        self._layer_results[layer][self._cur_index]['theta'] = dict(
            deviation=deviation,
            evaluations=evaluations
        )

    @property
    def pos(self):
        return self._cur_index

    @property
    def theta_df(self):
        df_results = []
        for name, layer in self._layer_results.items():
            for i, results in enumerate(layer):
                if 'theta' not in results:
                    continue
                evaluations = {
                    f'Theta Evaluation {i}':k
                    for i, k in enumerate(results['theta']['evaluations'])
                }

                df_results.append({
                    'Layer': name,
                    'Step': i,
                    'Theta Deviation': results['theta']['deviation'],
                    **evaluations
                })
        return pd.DataFrame(df_results)

    @property
    def input_df(self):
        df_results = []
        for name, layer in self._layer_results.items():
            for i, results in enumerate(layer):
                if 'inputs' not in results:
                    continue
                evaluations = {
                    f'Theta Evaluation {i}':k
                    for i, k in enumerate(results['inputs']['evaluations'])
                }

                df_results.append({
                    'Layer': name,
                    'Step': i,
                    'Theta Deviation': results['inputs']['deviation'],
                    **evaluations
                })
        return pd.DataFrame(df_results)


# need a way to "prepend the LearnF"


# if you want to use decision tree
# probably need to have a mixture of experts
# randomly choose what expert to update for each 
# sample... 
#
# output = 1 2 1 2 2 2
# update = 0 0 0 0 0 1 <- some small probability of updating
# otherwise the changes will be too great
# 


