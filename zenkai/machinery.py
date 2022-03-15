

from abc import ABC, abstractclassmethod, abstractmethod
from functools import singledispatchmethod
import typing
import torch as th
import torch.nn as nn

from abc import ABC, abstractproperty
from dataclasses import dataclass
import torch as th
import numpy as np


torch_id = "torch"
numpy_id = "numpy"
null_id = "null"


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
    def assess(self, x, t):
        pass

    @abstractmethod
    def update_ys(self, x):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x, t, ys=None, param_update: bool=True):
        pass

    def update(self, x, t, ys=None):
        self.backward(x, t, ys, True)    
        
    def update_x(self, x, t, ys=None):
        return self.backward(x, t, ys, False) 


class TorchNN(Machine):

    def __init__(self, module: nn.Module, optim_f, loss: nn.Module):
        
        self._module = module
        self._optim = optim_f(module.parameters())
        self._loss = loss

    def assess(self, x, t):
        y = self._module.forward(x)
        return self._loss(y, t)

    def update_ys(self, x):
        return self.forward(x)

    def forward(self, x):
        # use the grad in the backward method
        x.requires_grad_()
        x.retain_grad()
        return self._module.forward(x)

    def backward(self, x, t, ys=None, update: bool=True):
        ys = ys or self.update_ys(x)
        self._optim.zero_grad()
        result = self.assess(ys, t)
        result.backward()
        if update:
            self._optim.step()
        updated = x.grad
        return x + updated


class Sequence(Machine):

    def __init__(self, machines: typing.List[Machine]):
        if len(machines) == 0:
            raise ValueError(f'Length of sequence must be greater than 0')
        self.machines = machines
    
    def assess(self):
        # TODO: Write
        pass

    def update_ys(self, x):
        ys = []
        y = x
        for machine in self.machines:
            y = machine.update_ys(y)
            ys.append(y)
        return ys

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def backward(self, x, t, ys=None, update: bool=True):
        ys = ys or self.update_ys(x)
        xs = [x] + ys[:-1]
        for x_i, y_i, layer in reversed(xs, ys, self.layers):
            t = layer.backward(x_i, t, y_i, update)
        return t


class Processor(object):

    def update_ys(self, x):
        return self.forward(x)

    def forward(self, x):
        return x.detach().cpu().numpy()

    def backward(self, x, t, ys=None):
        return th.from_numpy(t)


class TH2NP(Processor):

    def update_ys(self, x):
        return self.forward(x)

    def forward(self, x):
        return x.detach().cpu().numpy()

    def backward(self, x, t, ys=None):
        return th.from_numpy(t)


class NP2TH(Processor):

    def update_ys(self, x):
        return self.forward(x)

    def forward(self, x):
        return th.from_numpy(x)

    def backward(self, x, t, ys=None):
        return t.detach().cpu().numpy()


class Processed(Machine):
    
    def __init__(self, processors: typing.List[Processor], machine: Machine):
        self.processors = processors
        self.machine = machine
    
    def assess(self):
        # TODO: Write
        pass
    
    def update_ys(self, x):
        ys = []
        y = x
        for processor in self.processors:
            y = processor.forward(y)
            ys.append(y)
        y.append(self.machine.update_ys(y))
        return ys

    def forward(self, x):
        y = x
        for processor in self.processors:
            y = processor.forward(y)
        return self.machine.forward(y)

    def backward(self, x, t, ys=None, update: bool=True):
        xs = [x] + ys[:-2]
        t = self.machine.backward(ys[-2], t, ys[-1], update)
        for x_i, y_i, processor in reversed(xs, ys[:-1], self.processors):
            t = processor.backward(x_i,  t, y_i,)
        return t


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
class LearnF(object):

    def target(x):
        pass

    def eval(x):
        pass


# need a way to "prepend the LearnF"
