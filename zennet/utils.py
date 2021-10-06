import dataclasses
import typing
import numpy as np
import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class SignalType(object):

    size: typing.Tuple[int]
    dtype: str


class Signal(ABC):

    @abstractmethod
    @property
    def signal_type(self) -> SignalType:
        pass

    @abstractmethod
    def to_torch(self, to_device: str='cpu'):
        pass

    @abstractmethod
    def to_numpy(self):
        pass


class TorchSignal(Signal):

    def __init__(self, contents: torch.Tensor):
        
        self._contents = contents
    
    @property
    def contents(self) -> torch.Tensor:
        return self._contents

    def to_torch(self, to_device: str='cpu'):

        pass


    def to_numpy(self):

        pass


@dataclass
class Port(ABC):

    dtype: str
    size: typing.Tuple[int]

    def check_combatibility(self, other):
        other: Port = other
        if self.dtype != other.dtype:
            return False
        
        if self.size != other.size:
            return False
        return True


@dataclass
class TorchPort(Port):
    device: str

    def check_combatibility(self, other: Port):
        if not super().check_combatibility(other):
            return False
        
        if not isinstance(other, TorchPort):
            return False
        
        if not self.device == other.device:
            return False

        return True

@dataclass
class NumpyPort(Port):

    def check_combatibility(self, other: Port):

        if not super().check_combatibility(other):
            return False
        
        if not isinstance(other, NumpyPort):
            return False
        return True
