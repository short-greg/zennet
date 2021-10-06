import typing
import numpy as np
import torch
from abc import ABC, abstractclassmethod, abstractmethod

from .utils import NumpyPort, Port, TorchPort


class Operation(ABC):

    @property
    def ports(self) -> typing.List[Port]:
        pass

    @abstractmethod
    def forward(self, x):
        pass


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
        # self.

    @property
    def ports(self) -> typing.List[TorchPort]:
        pass

    @abstractmethod
    def forward(self, x):
        pass


class Torch2Numpy(NumpyOperation):

    def __init__(self, torch_operation: TorchOperation):
        super().__init__(self)
        # self.

    @property
    def ports(self) -> typing.List[NumpyPort]:
        pass

    @abstractmethod
    def forward(self, x):
        pass


