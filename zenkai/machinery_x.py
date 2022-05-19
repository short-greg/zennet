

from abc import ABC, abstractclassmethod, abstractmethod
from audioop import reverse
from functools import partial, singledispatchmethod
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


class Machine(ABC):

    @abstractmethod
    def assess(self, x, t, y=None):
        pass

    @abstractmethod
    def update_ys(self, x) -> typing.Tuple[typing.Any, typing.Any]:
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x, t, ys=None, param_update: bool=True):
        pass

    @abstractmethod
    def get_y_out(self, ys):
        pass

    def update(self, x, t, ys=None):
        self.backward(x, t, ys, True)    
        
    def update_x(self, x, t, ys=None):
        return self.backward(x, t, ys, False) 


class Optimizer(ABC):

    @abstractproperty
    def reset_inputs(self, inputs):
        pass

    @abstractproperty
    def reset_theta(self, theta):
        pass

    @abstractmethod
    def update_theta(self, t, inputs=None, theta=None, y=None):
        pass
    
    @abstractmethod
    def update_inputs(self):
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


class GradOptimizer(TorchOptimizer):

    def __init__(self, net: nn.Module, objective: Objective, optim):
        super().__init__(net, objective)
        self._net = net
        self._optim: th.optim.Optimizer = optim(self._net.parameters())
        self._evaluation = None

    def reset_theta(self, theta=None):
        self._theta = theta
        nn_utils.vector_to_parameters(theta, self._net.parameters())
        self._evaluation = None

    def reset_inputs(self, inputs):
        
        self._inputs = inputs
        self._evaluation = None

    def update_theta(self, t, inputs=None, theta=None, y=None):
        if inputs is not None:
            self.reset_inputs(inputs)
        if theta is not None:
            self.reset_theta(theta)
        self._optim.zero_grad()
        if y is None:
            y = self._net(self._inputs)
        self._evaluation = self._objective(y, t)
        self._evaluation.mean().backward()
        self._optim.step()

    def update_inputs(self, t, inputs: th.Tensor=None, theta=None, y=None):
        if inputs is not None:
            self.reset_inputs(inputs)
        if theta is not None:
            self.reset_theta(theta)
        
        if y is None:
            y = self._net(self._inputs)
        self._evaluation = self._objective(y, t)
        self._evaluation.sum().backward()
        if self._inputs.grad is None:
            raise RuntimeError("Input is not a leaf node or does not retain grad")
        self._inputs = self._inputs - self._inputs.grad

    @property
    def inputs(self):
        return self._inputs
    
    @property
    def theta(self):
        return nn.utils.parameters_to_vector(self._net.parameters())
    
    @property
    def evaluations(self):
        return [self._evaluation]


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

    def update_theta(self, t, inputs=None, theta=None, y=None):
        if inputs is not None:
            self._x_updater.reset_inputs(inputs)
        if theta is not None:
            self._x_updater.reset_theta(theta)
        self._p_updater.update_theta(t, theta=theta, inputs=inputs, y=y)
        self._x_updater.reset_theta(self._p_updater.theta)
        self._updated_inputs = False
        self._updated_theta = True

    def update_inputs(self, t, inputs: th.Tensor=None, theta=None, y=None):
        if inputs is not None:
            self._p_updater.reset_inputs(inputs)
        if theta is not None:
            self._p_updater.reset_theta(theta)
        self._x_updater.update_inputs(t, theta=theta, inputs=inputs, y=y)
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


class GaussianHillClimberProcessor(HillClimberProcessor):

    def __init__(self, s: float=1e-2, k: int=1, momentum: float=None, maximize: bool=True):
        self.s = s
        self.k = k
        self.maximize = maximize
        self._momentum = momentum
        self._diff = None
        self._x_updated = None

    def __call__(self, x: th.Tensor):
        y = th.randn(self.k, *x.size()) * self.s + x
        x = x.unsqueeze(0)
        self._out = th.cat([x, y])
        return self._out
    
    def report(self, evaluations):
        best = get_best(self._out, evaluations, self.maximize)
        if self._diff is not None and self._momentum is not None:
            self._diff = (1 - self._momentum) * (best - self._x_updated) + self._momentum * self._diff
            self._x_updated = self._x_updated + self._diff
        elif self._momentum is not None:
            self._diff = (best - self._x_updated)
            self._x_updated = self._x_updated + self._diff
        elif self._x_updated is not None:
            self._x_updated = best
        return self._x_updated

    def spawn(self):
        return GaussianHillClimberProcessor(
            self.s, self.k, self._momentum, self.maximize
        )


class HillClimberOptimizer(TorchOptimizer):

    def __init__(self, net: nn.Module, objective: Objective, processor: typing.Union[str, GaussianHillClimberProcessor]="gaussian"):
        super().__init__(net, objective)
        if processor == "gaussian":
            self.input_processor = GaussianHillClimberProcessor(maximize=objective.maximize)
            self.input_processor = GaussianHillClimberProcessor(maximize=objective.maximize)
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

    def reset_theta(self, theta):
        self._theta = theta
        nn_utils.vector_to_parameters(theta, self._net.parameters())
        self._evaluation = None
        self._theta_diff = None

    def update_theta(self, t, inputs=None, theta=None, y=None):
        if inputs is not None:
            self.reset_inputs(inputs)
        if theta is not None:
            self.reset_theta(theta)
        
        self._evaluations = self._update_theta(t, y)

    def update_theta(self, t, inputs=None, theta=None, y=None):
        if inputs is not None:
            self.reset_inputs(inputs)
        if theta is not None:
            self.reset_theta(theta)
        
        self._evaluations = self._update_inputs(t, y)

    def _update_theta(self, t, y=None):
        theta = self.theta_processor(self._theta)
        evaluations = []
        for theta_i in theta:
            nn_utils.vector_to_parameters(theta_i, self._net.parameters())
            evaluations.append(self._objective.forward(self._net(self._inputs), t))
        evaluations = th.stack(evaluations)
        self._theta = self.theta_processor.report(evaluations)
        return evaluations
    
    def _update_inputs(self, t, y=None):
        inputs = self.input_processor(self._inputs)
        evaluations = self._objective.forward_multi(self._net(inputs), t)
        self._inputs = self.theta_processor.report(evaluations)
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


class SKOptimBuilder(object):

    def __call__(self, machines) -> Optimizer:
        pass


class THOptimBuilder(object):

    def __init__(self):
        
        self._optim = None
        self.grad()

    def grad(self, optim_cls=None, **kwargs):

        if optim_cls is None:
            optim_cls = th.optim.Adam

        self._optim = partial(
            GradOptimizer,
            optim=partial(optim_cls, **kwargs)
        )
        return self
    
    def hill_climber(self, momentum=0.5, processor=None):

        if processor is None:
            processor = GaussianHillClimberProcessor(s=1e-3, k=16)
        self._optim = partial(HillClimberOptimizer, momentum=momentum, perturber=processor)
        return self

    def __call__(self, net, loss) -> Optimizer:
        return self._optim(net, loss)


class TorchNN(Machine):

    def __init__(self, module: nn.Module, objective: Objective, updater: THOptimBuilder=None):
        
        self._module = module
        self._objective = objective
        self._updater = updater(module, objective) or GradOptimizer(module, objective)

    def assess(self, x, t, ys=None):
        if ys is None:
            y = self._objective(x)
        else:
            y = ys[1]
        return self._objective(y, t)

    def update_ys(self, x):
        x = x.detach().requires_grad_()
        x.retain_grad()
        y = self.forward(x)
        return y, [x, y]
    
    def get_y_out(self, ys):
        return ys[1]

    def get_in(self, ys):
        return ys[0]

    def forward(self, x):
        # x = x.detach().requires_grad_()
        # x.retain_grad()
        return self._module.forward(x)

    def backward(self, x, t, ys=None, update: bool=True):
        if ys is not None:
            y = self.get_y_out(ys)
            x = self.get_in(ys)
        else:
            x = x.detach().requires_grad_()
            x.retain_grad()
            y = self._module(x)

        if update:
            self._updater.update_theta(t, y=y, inputs=x)
            nn_utils.vector_to_parameters(self._updater.theta, self._module.parameters())
        
        self._updater.update_inputs(t, y=y, inputs=x)
        return self._updater.inputs

    @property
    def module(self):
        return self._module


class SklearnMachine(Machine):

    def __init__(self, machines, updater: SKOptimBuilder=None):
        super().__init__()
        self._machines = machines
        self._updater = updater(machines)

    def assess(self, x, t, ys=None):
        if y is None:
            y = self.forward(x)
        else:
            ys = ys
        return self._loss(y, t)

    def update_ys(self, x):
        y = self.forward(x)
        return y, [x, y]

    def forward(self, x: th.Tensor):
        device = x.device
        x_np = x.detach().cpu().numpy()
        y_np = np.stack([machine.predict(x_np) for machine in self._machines])
        return th.tensor(y_np, device=device)
        
    def get_y_out(self, ys):
        return ys[1]

    def get_in(self, ys):
        return ys[0]
    
    def backward(self, x, t, ys=None, update: bool=True):
        # if ys is not None:
        #     y = self.get_y_out(ys)
        #     x = self.get_in(ys)
        # else:
        #     y = self.forward(x)

        if update:
            # self._updater.update_theta(t, y=y, inputs=x)
            # nn_utils.vector_to_parameters(self._p_updater.theta, self._module.parameters()
            # x_np = x.detach().cpu().numpy()
            # for machine in self._machines:
            #     machine.partial_fit(x_np, t)
            self._updater.update_theta(t, inputs=x)
            
        self._updater.update_inputs(t, inputs=x)
        return self._updater.inputs


class Sequence(Machine):

    def __init__(self, machines: typing.List[Machine]):
        if len(machines) == 0:
            raise ValueError(f'Length of sequence must be greater than 0')
        self.machines = machines
    
    def assess(self, x, t, ys=None):
        if ys is None:
            _, ys = self.update_ys(x)
        
        return self.machines[-1].assess(
            x, t, ys[-1]
        )

    def get_y_out(self, ys):
        return self.machines[-1].get_y_out(ys[-1]) 

    def get_in(self, ys):
        return ys[0]

    def update_ys(self, x):
        ys = [x]
        y = x
        for machine in self.machines:
            y, ys_i = machine.update_ys(y)
            ys.append(ys_i)
        return y, ys

    def forward(self, x):
        y = x
        for layer in self.machines:
            y = layer.forward(y)
        return y

    def backward(self, x, t, ys=None, update: bool=True):
        if ys is None:
            _, ys = self.update_ys(x)
        
        xs = [x]
        for y_i, machine in zip(ys[1:-1], self.machines[:-1]):
            xs.append(machine.get_y_out(y_i))

        for x_i, y_i, machine in zip(reversed(xs), reversed(ys[1:]), reversed(self.machines)):
            t = machine.backward(x_i, t, y_i, update)
        return t


class Processor(ABC):

    @abstractmethod
    def update_ys(self, x):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x, t, ys=None):
        pass

    @abstractmethod
    def get_y_out(self, ys):
        pass


class TH2NP(Processor):

    def update_ys(self, x):
        y = self.forward(x)
        return y, y

    def forward(self, x):
        return x.detach().cpu().numpy()

    def backward(self, x, t, ys=None):
        return th.from_numpy(t)

    def get_y_out(self, ys):
        return ys


class NP2TH(Processor):

    def __init__(self, dtype: th.dtype=th.float32):
        self.dtype = dtype

    def update_ys(self, x):
        y = self.forward(x)
        return y, y

    def forward(self, x):
        return th.tensor(x, dtype=self.dtype)
        # return th.from_numpy(x)

    def backward(self, x, t, ys=None):
        return t.detach().cpu().numpy()

    def get_y_out(self, ys):
        return ys


class CompositeProcessor(Processor):

    def __init__(self, processors: typing.List[Processor]):

        self.processors = processors

    def update_ys(self, x):
        ys = [x]
        y = x
        for processor in self.processors:
            y, ys_i = processor.update_ys(y)
            ys.append(ys_i)
        return y, ys

    def forward(self, x):
        y = x
        for processor in self.processors:
            y = processor.forward(y)
        return y

    def backward(self, x, t, ys=None):

        if ys is None:
            _, ys = self.update_ys(x)
        
        xs = ys[:-1]
        for x_i, y_i, processor in zip(
            reversed(xs), reversed(ys[1:-1]), reversed(self.processors)):
            t = processor.backward(x_i, t, y_i)
        return t

    def get_y_out(self, ys):
        if len(self.processors) == 0:
            return ys[0]
        return self.processors[-1].get_y_out(ys[-1])


class Processed(Machine):
    
    def __init__(
        self, processors: typing.List[Processor], 
        machine: Machine
    ):
        self.processors = CompositeProcessor(processors)
        self.machine = machine
    
    def assess(self, x, t, ys=None):
        if not ys:
            y, ys = self.update_ys(x)
        # if ys is None:
        #     y = self.forward(x)
        # else:
        #     y = self.machine.get_y_out(ys[1])
        y = self.processors.get_y_out(ys[0])
        return self.machine.assess(y, t, ys[1])
    
    def update_ys(self, x):
        y, ys_i = self.processors.update_ys(x)
        # y = ys[-1] if ys[-1] is not None else x
        y, ys_j = self.machine.update_ys(y)
        return y, [ys_i, ys_j]

    def forward(self, x):
        y = self.processors.forward(x)
        return self.machine.forward(y)

    def backward(self, x, t, ys=None, update: bool=True):
        if ys is None:
            _, ys = self.update_ys(x)

        y_in = self.processors.get_y_out(ys[0])
        t = self.machine.backward(y_in, t, ys[1], update)
        return self.processors.backward(x, t, ys[0])

    def get_y_out(self, ys):
        return self.machine.get_y_out(ys[-1])


class WeightedLoss(nn.Module):
    
    def __init__(self, nn_loss: nn.Module, weight: float):

        super().__init__()
        self.nn_loss = nn_loss
        self.weight = weight

    def forward(self, x, t):

        return self.weight * self.nn_loss(x, t)


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


