

from abc import ABC, abstractclassmethod, abstractmethod
from functools import singledispatchmethod
import typing
import torch as th

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


class Layer(ABC):

    @abstractmethod
    def assess(self, x, t):
        pass

    @abstractmethod
    def updateOutputs(self, x):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x, t, param_update: bool=True):
        pass

    def update(self, x, ys, t):
        self.backward(x, ys, t, True)    
        
    def update_x(self, x, ys, t):
        return self.backward(x, ys, t, False) 


class TorchNN(Layer):

    def __init__(self, module: th.Module, optim_f, loss: th.Module, input_tau: float=1e-3):
        
        self._module = module
        self._optim = optim_f(module.parameters())
        self._loss = loss
        self.input_tau = input_tau

    def assess(self, x, t):
        y = self._module.forward(x)
        return self._loss(y, t)

    def excite(self, x):
        return self.forward(x)

    def forward(self, x):
        # use the grad in the backward method
        x.retain_grad()
        return self._module.forward(x)

    def backward(self, x, ys, t, update: bool=True):
        self._optim.zero_grad()
        result = self.assess(ys, t)
        result.backward()
        if update:
            self._optim.step()
        updated = x.grad
        return x + updated * self.input_tau


class Sequence(Layer):

    def __init__(self, layers: typing.List[Layer]):

        if len(self._layers) == 0:
            raise ValueError(f'Length of sequence must be greater than 0')
        self._layers = layers

    def excite(self, x):
        ys = []
        y = x
        for layer in self._layers:
            y = layer.excite(y)
            ys.append(y)
        return ys

    def forward(self, x):
        y = x
        for layer in self._layers[:-1]:
            y = layer.forward(y)
        return self._module.forward(x)

    def backward(self, x, ys, t, update: bool=True):
        xs = [x] + ys[:-1]
        for x_i, y_i, layer in reversed(xs, ys, self._layers):
            t = layer.backward(x_i, y_i, t, update)
        return t

    def update(self, x, ys, t):
        self.backward(x, ys, t, True)    
        
    def update_x(self, x, ys, t):
        return self.backward(x, ys, t, False)    


class TH2NP(Layer):

    def excite(self, x):
        return self.forward(x)

    def forward(self, x):
        return x.detach().cpu().numpy()

    def backward(self, x, ys, t, update: bool=True):
        return th.from_numpy(t)

    def update(self, x, ys, t):
        self.backward(x, ys, t, True)    
        
    def update_x(self, x, ys, t):
        return self.backward(x, ys, t, False)   


class NP2TH(Layer):

    def excite(self, x):
        return self.forward(x)

    def forward(self, x):
        return th.from_numpy(x)

    def backward(self, x, ys, t, update: bool=True):
        return t.detach().cpu().numpy()

    def update(self, x, ys, t):
        self.backward(x, ys, t, True)    
        
    def update_x(self, x, ys, t):
        return self.backward(x, ys, t, False)
