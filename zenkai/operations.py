import typing
import numpy as np
import torch
from abc import ABC, abstractclassmethod, abstractmethod

from torch._C import Value
from . import _utils

from ._utils import NumpyPort, Port, TorchPort


class Operation(object):

    def __init__(self, port: Port):
        super().__init__()
        self._port = port

    @property
    def ports(self) -> typing.List[Port]:
        return self._port.to_list()

    def validate(self, x):
        if not self._port.accept(x):
            raise ValueError("Invalid signal for port.")

    def forward(self, x):
        self.validate(x)
        return x

    def __call__(self, x: typing.Iterable[_utils.Signal]):
        return self.forward(x)


class TorchOperation(Operation):

    @property
    def ports(self) -> typing.List[TorchPort]:
        pass

    @abstractmethod
    def forward(self, x):
        pass


class NumpyOperation(Operation):

    @property
    def ports(self) -> typing.List[NumpyPort]:
        pass

    @abstractmethod
    def forward(self, x):
        pass


class Numpy2Torch(TorchOperation):

    def __init__(self, numpy_operation: NumpyOperation, to_device: str='cpu'):
        super().__init__(self)
        self._operation = numpy_operation
        self._to_device = to_device

    @property
    def ports(self) -> typing.List[TorchPort]:
        pass

    @abstractmethod
    def forward(self, x):
        pass


class Torch2Numpy(NumpyOperation):

    def __init__(self, torch_operation: TorchOperation, port: _utils.TorchPort):
        super().__init__(self)
        self._operation = torch_operation
        self._port = port

    @property
    def ports(self) -> typing.List[NumpyPort]:
        pass

    def forward(self, x: typing.Iterable[_utils.TorchSignal]):
        super().__init__(self)

        return self._operation(
            map(lambda x: x.to_numpy(), x)
        )
