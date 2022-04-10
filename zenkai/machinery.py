

from abc import ABC, abstractclassmethod, abstractmethod
from audioop import reverse
from functools import singledispatchmethod
import typing
import sklearn
import torch as th
import torch.nn as nn

from abc import ABC, abstractproperty
from dataclasses import dataclass
import torch as th
import numpy as np
import torch

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


class TorchNN(Machine):

    def __init__(self, module: nn.Module, optim_f, loss: nn.Module):
        
        self._module = module
        self._optim = optim_f(module.parameters())
        self._loss = loss

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

        if ys is None:
            x = x.detach()
            y = self.forward(x)
        else:
            y = ys[1]
            x = ys[0]
        self._optim.zero_grad()
    
        result = self._loss(y, t)
        result.backward()
        if update:
            self._optim.step()
        updated = x.grad
        return x + updated


class TorchNNBackwardMixin(object):

    def __init__(self, *args, **kwargs):
        TorchNN.__init__(self, *args, **kwargs)
    
    @abstractmethod
    def backward(self, x, t, ys=None, update: bool=True):
        raise NotImplementedError


class ClassBackwardMixin(TorchNNBackwardMixin):

    def backward(self, x, t, ys=None, update: bool=True):
        x_prime = TorchNN.backward(self, x, t, ys, update)
        return torch.clamp(x_prime, 0.0, 1.0)


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
    
    # maybe i need to pass in all ys
    # cause this is not optimized
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

        for x_i, y_i, machine in zip(reversed(xs), reversed(ys), reversed(self.machines)):
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

    def __init__(self, dtype: torch.dtype=torch.float32):
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
    
    def __init__(self, nn_loss: torch.nn.Module, weight: float):

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
class LearnF(object):

    def target(x):
        pass

    def eval(x):
        pass


# need a way to "prepend the LearnF"
