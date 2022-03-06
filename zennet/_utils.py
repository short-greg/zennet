import typing
import numpy as np
import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class SignalType(object):

    size: typing.Tuple[int]
    dtype: str


@dataclass
class TorchSignalType(SignalType):

    device: str


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
        self._signal_type = TorchSignalType(
            dtype=contents.dtype,
            size=tuple(self._contents.shape),
            device=contents.device
        )
    
    @property
    def contents(self) -> torch.Tensor:
        return self._contents

    def to_torch(self, device: str='cpu'):

        if device != self._contents.device:
            return TorchSignal(self._contents.to(device))
        
        return self

    def to_numpy(self):

        return NumpySignal(self._contents.detach().cpu().numpy())


class NumpySignal(Signal):

    def __init__(self, contents: np.ndarray):
        
        self._contents = contents
        self._signal_type = SignalType(
            dtype=contents.dtype,
            size=tuple(self._contents.shape)
        )
    
    @property
    def contents(self) -> torch.Tensor:
        return self._contents

    def to_torch(self, device: str='cpu') -> TorchSignal:

        return TorchSignal(torch.from_numpy(self._contents, device=device))
        

    def to_numpy(self):

        return self



@dataclass
class Port(ABC):

    @property
    def signal_type(self) -> SignalType:
        pass

    def check_combatibility(self, other):
        other: Port = other
        if self.dtype != other.dtype:
            return False
        
        if self.size != other.size:
            return False
        return True
    
    @abstractmethod
    def accept(self, x: Signal):
        pass
    
    @abstractmethod
    def to_list(self):
        pass

# all ports need to accept multiple inputs

class TorchPort(Port):

    def __init__(self, torch_signal_type: TorchSignalType):
        self._signal_type = torch_signal_type
    
    @property
    def signal_type(self) -> TorchSignalType:
        return self._signal_type

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
