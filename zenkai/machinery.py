

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


class Func(ABC):

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def reset(self, p):
        pass

class NNParamFunc(Func):

    def __init__(self, nn_module: nn.Module):

        self.nn_module = nn_module
        self.x = None

    def __call__(self, x):
        
        ys = []
        for p_i in x[0]:
            nn_utils.vector_to_parameters(p_i, self.nn_module.parameters())
            ys.append(self.nn_module(self.x))
        return th.stack(ys)

    def reset(self, p):
        self.x = p


class NNFunc(Func):

    def __init__(self, nn_module: nn.Module):
        self.nn_module = nn_module
    
    def __call__(self, x):

        n, b = x.size(0), x.size(1)
        x = x.view(n * b, *x.size()[2:])
        y = self.nn_module(x)
        return x.view(n, b, *y.size()[1:])

    def reset(self, p):
        nn_utils.vector_to_parameters(p, self.nn_module.parameters())


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

    def forward(self, x, t):
        t = t.unsqueeze(0)
        evaluation = self.eval(x, t)
        return self.reduction(evaluation)
    
    def best(self, evaluations, x):
        if self.maximize:
            return x[th.argmax(evaluations)]
        return x[th.argmin(evaluations)]


class LossObjective(nn.Module):

    def __init__(self, th_loss):
        super().__init__()
        loss = deepcopy(th_loss)
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


class Preparer(ABC):

    def __init__(self):
        self._x = None
    
    @abstractproperty
    def prepare_in(self):
        pass

    @abstractproperty
    def prepare_out(self):
        pass

    @property
    def x(self):

        return self._x

    @abstractmethod
    def prepare(self, x_seed):
        pass

    @abstractmethod
    def prepare_init(self, x_seed):
        pass







class PrepareGuassianNoise(Preparer):

    def __init__(self, r, k):
        self.r = r
        self.k = k

    def prepare(self, x):
        self._x = x + th.randn(self.k, *x.size()) * self.r
        return self._x

    def prepare_init(self, x):
        self._x = x + th.randn(self.k, *x.size()) * self.r
        return self._x


class Updater(ABC):

    def __init__(self, k: int):
        self._best = None
        self._x = None
        self.k = k
    
    @property
    def best(self):
        return self._best

    @abstractproperty
    def update_out(self):
        pass

    @abstractproperty
    def update_in(self):
        pass

    @property
    def x(self):
        return self._x
    
    @abstractmethod
    def step(self, x, fitness):
        pass


class MaxUpdater(Updater):

    def step(self, x: th.Tensor, fitness: th.Tensor):
        x_best = x[fitness.argmax()]
        self._best = x_best
        self._x = x

    @property
    def update_out(self):
        None

    @property
    def update_in(self):
        self.k


class Optimizer(ABC):

    def __init__(self, f: Func, objective: Objective):
        self._f = f
        self._objective = objective

    @abstractmethod
    def initialize(self, x):
        pass
    
    @abstractmethod
    def update(self, t, x=None):
        pass

    @abstractmethod
    def reset(self, p=None):
        pass

    @abstractproperty
    def best(self):
        pass

    @abstractproperty
    def evaluations(self):
        pass


class PGradOptimizer(Optimizer):

    def __init__(self, f: NNParamFunc, objective: Objective, optim):
        super().__init__(f, objective)
        self._optim: th.optim.Optimizer = optim(f.best_parameters())
        self._evaluation = None

    def initialize(self, p):
        self._p = p
        self._evaluation = None

    def update(self, t, x=None):
        if x is not None:
            self.reset(x)
    
        self._optim.zero_grad()
        self._evaluation = self._objective(self._f(self._x), t)
        self._evaluation.mean().backward()
        self._optim.step()

    def reset(self, p=None):
        self._x = p

    @property
    def best(self):
        return self._f.best
    
    @property
    def evaluations(self):
        return [self._evaluation]
    

class XGradOptimizer(Optimizer):

    def __init__(self, f: NNFunc, objective: Objective, optim):
        super().__init__(f, objective)
        self._evaluation = None

    def initialize(self, x):
        self._x = x
        self._evaluation = None

    def update(self, t, x=None):

        grad_set = x is not None and x.grad is not None
        x = self._x if x is None else x
        if not grad_set:
            self._evaluation = self._objective(self._f(x), t)
            self._evaluation.mean().backward()
        else:
            self._evaluation = None
        self._x = x - x.grad

    @property
    def best(self):
        return self._best
    
    @property
    def evaluations(self):
        return [self._evaluation]

    def reset(self, p=None):
        self._f.reset(p)
    

class HillClimberF(ABC):

    def __call__(self, x):
        pass


class GaussianHillClimberF(HillClimberF):

    def __init__(self, s: float, k: int):
        self.s = s
        self.k = k

    def __call__(self, x: th.Tensor):
        x = x.unsqueeze(0)
        y = th.randn(self.k, *x.size()) * self.s + x
        return th.cat([x, y])


class HillClimberOptimizer(Optimizer):

    def __init__(self, f: Func, objective: Objective, momentum: float, perturber: HillClimberF):
        super().__init__(f, objective)
        self._momentum = momentum
        self.perturber = perturber
        self._diff = None

    def initialize(self, x):
        self._x = x
        self._evaluation = None
        self._diff = None
    
    def update(self, t, x=None):

        if x is None: self.initialize(x)
        
        x = self.perturber(self._x)
        self._evaluations = self._objective(self._f(x), t)
        best =  self._objective.best(self._evaluations, x)
        if self._diff is not None and self._momentum is not None:
            self._diff = (1 - self._momentum) * (best - x) + self._momentum * self._diff
            self._x = self._x + self._diff
        elif self._momentum is not None:
            self._diff = (best - x)
            self._x = self._x + self._diff
        else:
            self._x = best
        
        self._true_best = best
        return self._evaluations

    def reset(self, p=None):
        self._f.reset(p)

    @property
    def best(self):
        return self._x
    
    @property
    def evaluations(self):
        return self._evaluations



# class EvolutionaryStrategy(OptimizerBuilder):
    
#     @abstractmethod
#     def reset(self):
#         raise NotImplementedError

#     @abstractproperty
#     def product(self) -> Optimizer:
#         pass

class TorchNN(Machine):

    def __init__(self, module: nn.Module, loss: nn.Module, x_updater, p_updater):
        
        self._module = module
        self._loss = loss
        # TODO* Build correctly
        self._p_updater: Optimizer = p_updater(self._module, loss)
        self._x_updater: Optimizer = x_updater(self._module, loss)

    def assess(self, x, t, ys=None):
        if ys is None:
            y = self.forward(x)
        else:
            y = ys[1]
        return self._loss(y, t)

    def update_ys(self, x):
        x = x.detach()
        y = self.forward(x)
        return y, [x, y]
    
    def get_y_out(self, ys):
        return ys[1]

    def forward(self, x):
        # use the grad in the backward method
        x.requires_grad_()
        x.retain_grad()
        return self._module.forward(x)

    def backward(self, x, t, ys=None, update: bool=True):
        
        if ys is not None:
            y = self.get_y_out(ys)
        else:
            y  = None

        if update:
            self._p_updater.reset(p=x)
            self._p_updater.update(t, y=y)
            self._x_updater.reset(self._p_updater.best)
        
        self._x_updater.initialize(x)
        self._x_updater.update(t, y=y)
        return self._x_updater.best

    @property
    def module(self):
        return self._module


class SklearnMachine(Machine):

    def __init__(self, machines: typing.List, loss: typing.Callable):
        super().__init__()
        self._machines = machines
        self._loss = loss

    def assess(self, x, t, ys=None):
        if y is None:
            y = self.forward(x)
        else:
            ys = ys
        return self._loss(y, t)

    def update_ys(self, x):
        y = self.forward(x)
        return y, y

    def forward(self, x):
        return np.stack([machine.predict(x) for machine in self._machines])

    def update(self, x, t, ys=None):
        for machine in self._machines:
            machine.partial_fit(x, t)
        
    def update_x(self, x, t, ys=None):
        pass
        # need to calculate the updated inputs
        # pattern search

    def get_y_out(self, ys):
        return ys

    def backward(self, x, t, ys=None, update: bool=True):
        
        if update:
            self.update(x, t, ys)
        return self.update_x(x, t, ys)
        

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


# class THLoss(Layer):

#     def __init__(self, th_loss: nn.Module, tau: float=1e-3):
#         self._loss = th_loss

#     def excite(self, x, t):
#         return self.forward(x)

#     def forward(self, x, t):
#         x.requires_grad_()
#         return self._loss(x, t)

#     def backward(self, x, ys, t, update: bool=True):
#         ys.backward()
#         return x + tau * x.grad

#     def update(self, x, ys, t):
#         pass
        
#     def update_x(self, x, ys, t):
#         return self.backward(x, ys, t, False)


# layer <- does processing
# connector <- doesn't actually process
# compound -> connector .. layer


# allow to pass in t or f for the target
# f is a "LearnF" object
# make ys optional
# net.assess(x, t, ys <- not necessary)
# net.update(x, t, ys <- not necessary)
# net.update_x(x, t, ys <- not necessary)
# net.update_x(x, f, ys) <- pass in f rather than t
# net.backward(x, t, ys)
# net.backward(x, f, ys)
# 
# builder.add_torch_layer()
# builder.add_numpy_layer()

# allow this this for optimization

# class LearnF(object):

#     def target(x):
#         pass

#     def eval(x):
#         pass


# need a way to "prepend the LearnF"


# class HillDescendingBackwardMixin(TorchNNBackwardMixin):

#     def backward(self, x, t, ys=None, update: bool=True):
#         if update:
#             TorchNN.backward(self, x, t, ys, update)
        
        # 1) x_samples = sample(x, count)
        # 2) result = eval(x_samples.batch_flatten(), agg=False) <- need a way to make eval not aggregate
        # 3) x_prime = select(x_samples, result.deflatten())
        # return x_prime

# if you want to use decision tree
# probably need to have a mixture of experts
# randomly choose what expert to update for each 
# sample... 
#
# output = 1 2 1 2 2 2
# update = 0 0 0 0 0 1 <- some small probability of updating
# otherwise the changes will be too great
# 



# class ObjectiveBuilder(BackwardBuilder):

#     def __init__(self, x_optim: Optimizer, p_optim: Optimizer):
#         self._x_optim = x_optim
#         self._p_optim = p_optim

#     def produce(self, module, loss) -> Backward:
#         return ObjectiveBackward(module, loss)


# class OptimizerBuilder(object):
    
#     @abstractmethod
#     def reset(self):
#         raise NotImplementedError
    
#     @abstractproperty
#     def product(self) -> Optimizer:
#         raise NotImplementedError




# class Optimizer(ABC):

#     def __init__(self, name: str, f: Func, objective: Objective, preparer: Preparer, updater: Updater, x_seed=None):

#         self.name = name
#         self.objective = objective
#         self.f = f
#         self.x_seed = x_seed
#         self.best = None
#         self.preparer = preparer
#         self.updater = updater
#         self.y = None
    


#     def update(self, t):
    
#         if not self.y:
#             x = self.preparer.prepare_init(self.x_seed)
#         else:
#             x = self.preparer.prepare(self.y)
#         fitness = self.objective(self.f(x), t)
#         self.updater(x, fitness)
#         self.y = self.updater.x
#         return self.y

#     def update_k(self, t, x_seed=None, k: int=1):
        
#         x = x_seed
#         for _ in range(k):
#             self.update(t, x)
#             x = self.updater.x
#         return self.best



# class Backward(ABC):

#     @abstractmethod
#     def __call__(self, x, t, ys, update: bool=True):
#         pass


# class OptimBackward(Backward):

#     def __init__(self, module: nn.Module, loss, optim_f, input_updater):
#         self._module = module
#         self._loss = loss
#         self._optim = optim_f(module.parameters())
#         self._input_updater = input_updater
    
#     def __call__(self, x: th.Tensor, t: th.Tensor, ys, update: bool=True):
#         if ys is None:
#             x = x.detach()
#             y = self._module.forward(x)
#         else:
#             y = ys[1]
#             x = ys[0]

#         self._optim.zero_grad()
#         result = self._loss(y, t)
#         result.backward()
#         if update:
#             self._optim.step()
#         return self._input_updater(x, x.grad)


# # TODO* This does not actually need to be torch
# class ObjectiveBackward(Backward):

#     def __init__(self, module: nn.Module, loss: nn.Module, x_optim: Optimizer, p_optim: Optimizer):

#         self._module = module
#         self._loss = loss
#         self._x_optim = x_optim
#         self._p_optim = p_optim

#     def __call__(self, x, t, ys=None, update: bool=True):

#         if update:
#             self._p_optim.x = x
#             self._p_optim.update(t)
#             self._module = self._p_optim.best
#         self._x_optim.f = NNFunc(self._module)
#         self._x_optim.update(t, x)
#         return self._x_optim.best


# class BackwardBuilder(ABC):

#     @abstractmethod
#     def produce(self, module, loss) -> Backward:
#         pass


# class OptimBuilder(BackwardBuilder):

#     def __init__(self, optim_f):
#         self._optim_f = optim_f
#         self._input_updater = None
#         self.std_update()
    
#     def std_update(self):
#         def update(x, x_grad):
#             return x - x_grad
#         self._input_updater = update
#         return self

#     def clamp_update(self, lower=0.0, upper=1.0):
#         def update(x, x_grad):
#             x = x - x_grad
#             return th.clamp(x, min=lower, max=upper).detach()
#         self._input_updater = update
#         return self

#     def produce(self, module, loss) -> Backward:

#         return OptimBackward(module, loss, self._optim_f, input_updater=self._input_updater)


# 1) Split up

# class XUpdater(object):
#     pass


# class PUpdater(object):
#     pass


# class ObjectiveBuilder(ABC):
#     pass


# class HillClimberBuilder(ObjectiveBuilder):
#     pass

# add x_updater and y_updater
# backward <- this is the target function
#   if update:
#     self.update_p(x, ys, etc)
#   return self.update_x(x, ys, etc)


# # TODO: remove this
# class ObjectiveBuilder(BackwardBuilder):

#     def __init__(self, x_size):
#         self._k = 1
#         self._x_size = x_size
#         self._x_updater_name = None
#         self._p_updater_name = None

#     def set_p_hill_climbing(self):

#         self._p_updater_name = 'HillClimber'
#         self._p_preparer = PrepareGuassianNoise(0.2, self._k)
#         self._p_updater = MaxUpdater(self._k)
#         return self

#     def set_x_hill_climbing(self, k: int=4):
#         self._x_updater_name = 'HillClimber'
#         self._k = k
#         self._x_preparer = PrepareGuassianNoise(0.2, k)
#         self._p_preparer.k = k
#         self._p_updater.k = k
#         self._x_updater = MaxUpdater(k)
#         return self

#     def reset(self):
#         raise NotImplementedError

#     def produce(self, module, objective_f, is_loss) -> Backward:

#         objective = LossObjective(objective_f)
#         x_f = NNFunc(module)
#         p_f = NNParamFunc(module)
        
#         p_init = nn_utils.parameters_to_vector(module.parameters())

#         x_optimizer = Optimizer(
#             self._x_updater_name,
#             x_f, self._objective, 
#             self._x_preparer, self._x_updater, th.zeros(*self.x_size)
#         )
#         p_optimizer = Optimizer(
#             self._p_updater_name,
#             p_f, self._objective, 
#             self._p_preparer, self._p_updater, p_init
#         )
#         return ObjectiveBackward(module, objective, x_optimizer, p_optimizer)
