

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

from .modules import Objective, SklearnModule, Skloss
from .optimizers import (
    SklearnThetaOptim, GradInputOptim, GradThetaOptim, Scorer, GradThetaOptim, GradInputOptim
)
from .hill_climbing import HillClimbInputOptim
from .optim_builders import ThetaOptimBuilder, SklearnOptimBuilder, InputOptimBuilder

from sklearn import multioutput

torch_id = "torch"
numpy_id = "numpy"
null_id = "null"



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
    def assess(self, y, t):
        pass

    @abstractmethod
    def output_ys(self, x) -> typing.Tuple[typing.Any, typing.Any]:
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def forward_update(self, x, t, scorer: Scorer=None, update_theta: bool=True):
        pass

    @abstractmethod
    def backward_update(self, x, t, outs=None, update_theta: bool=True, update_inputs: bool= True):
        pass

    @abstractmethod
    def get_y_out(self, outs):
        pass
    
    def __call__(self, x):
        return self.forward(x)


def to_float(x: typing.List[th.Tensor]):
    return list(map(lambda xi: xi.mean().item(), x))


class TorchNN(Machine):

    def __init__(
        self, module: nn.Module, objective: Objective , 
        theta_updater: ThetaOptimBuilder=None, input_updater: InputOptimBuilder=None,
        fixed: bool=False, 
        recorder: Recorder=None):
        
        self._module = module
        self._objective = objective
        self._theta_updater =  (
            theta_updater(module, objective)
            if theta_updater is not None else GradThetaOptim (module, objective)
        )
        self._input_updater = (
            input_updater(module, objective) 
            if input_updater is not None else GradInputOptim(module, objective, skip_eval=True)
        )
        self._fixed = fixed
        self._recorder = recorder

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
        return self._module.forward(x)
    
    def forward_update(self, x, t, scorer: Scorer=None, update_theta: bool=True):

        if update_theta and not self._fixed:
            self._theta_updater.step(x, t, scorer=scorer)
            if self._recorder:
                self._recorder.record_theta(
                    id(self), nn_utils.parameters_to_vector(self._module.parameters()), self._theta_updater.theta,
                    to_float(self._theta_updater.evaluations)
                )
            nn_utils.vector_to_parameters(self._theta_updater.theta, self._module.parameters())
        
        y = self.forward(x)
        return y

    def backward_update(self, x, t, outs=None, update_theta: bool=True, update_inputs: bool= True):
        if outs is not None:
            y = self.get_y_out(outs)
            x = self.get_in(outs)
        else:
            x = x.detach().requires_grad_()
            x.retain_grad()
            y = self._module(x)

        if update_theta and not self._fixed:
            self._theta_updater.step(x, t, y=y)
        
            if self._recorder:
                self._recorder.record_theta(
                    id(self), nn_utils.parameters_to_vector(self._module.parameters()), 
                    self._theta_updater.theta,
                    evaluations=to_float(self._theta_updater.evaluations)
                )
            nn_utils.vector_to_parameters(self._theta_updater.theta, self._module.parameters())
        
        if update_inputs:
            x_prime = self._input_updater.step(x, t, y=y)
            if self._recorder:
                self._recorder.record_inputs(
                    id(self), x, x_prime,
                    evaluations=to_float(self._input_updater.evaluations)
                )
            assert x_prime is not None, f'{self._input_updater}'
            return x_prime

    @property
    def module(self):
        return self._module


class SklearnMachine(Machine):

    def __init__(
        self, module: SklearnModule, loss, 
        theta_updater: SklearnOptimBuilder, input_updater: InputOptimBuilder,
        fixed: bool=False, partial: bool=False, recorder: Recorder=None
    ):
        super().__init__()
        self._module = module
        self._loss = loss
        self._theta_updater =  theta_updater(self._module)
        self._input_updater = input_updater(self._module, Skloss(self._module))
        self._fixed = fixed
        self._partial = partial
        self._fit = False
        self._recorder = recorder

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
        y_np = self._module.forward(x)
        return y_np.to(device)

    def forward_update(self, x, t, scorer: Scorer=None, update_theta: bool=True):
        """forward update on the module. SKLearn Module cannot be updated on Forward update

        Args:
            x (_type_): _description_
            t (_type_): _description_
            scorer (Scorer, optional): Has no effect on SKLearn module. Defaults to None.
            update_theta (bool, optional): Whether to update the paramters. Defaults to True.

        Returns:
            output: _description_
        """
        y = self.forward(x)
        return y
        
    def get_y_out(self, outs):
        return outs[1]

    def get_in(self, outs):
        return outs[0]
    
    def backward_update(self, x, t, outs=None, update_theta: bool=True, update_inputs: bool= True):
        
        if update_theta and not self._fixed:
            self._theta_updater.step(x, t)
        
        if update_inputs:
            x_prime = self._input_updater.step(x, t)
            
            if self._recorder:
                self._recorder.record_inputs(
                    id(self), x, x_prime,
                    evaluations=to_float(self._input_updater.evaluations)
                )
            return x_prime


class BlackboxMachine(Machine):

    def __init__(self, f, loss, input_updater: InputOptimBuilder):
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

    def forward_update(self, x, t, scorer: Scorer=None, update_theta: bool=True):
        y = self.forward(x)
        return y

    def get_y_out(self, outs):
        return outs[1]

    def get_in(self, outs):
        return outs[0]
    
    def backward_update(self, x, t, outs=None, update_theta: bool=True, update_inputs: bool= True):
        
        if update_inputs:
            return self._input_updater.step(x, t)


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

    def forward_update(self, x, t, scorer: Scorer=None, update_theta: bool=True):
        if not update_theta:
            y = self.forward(x)
            return y

        y = x
        for i, machine in enumerate(self.machines):
            if i < len(self.machines) - 1:
                cur_scorer = SequenceScorer(self.machines[i + 1:], t, scorer)
            else:
                cur_scorer = scorer
            y = machine.forward_update(y, t, cur_scorer, update_theta)

        return y

    def backward_update(self, x, t, outs=None, update_theta: bool=True, update_inputs: bool= True):
        if outs is None:
            _, outs = self.output_ys(x)
        xs = [x]
        for y_i, machine in zip(outs[1:-1], self.machines[:-1]):
            xs.append(machine.get_y_out(y_i))

        for i, (x_i, y_i, machine) in enumerate(zip(reversed(xs), reversed(outs[1:]), reversed(self.machines))):
            _update_inputs = i < len(xs) or update_inputs
            t = machine.backward_update(x_i, t, y_i, update_theta, _update_inputs)
        
            assert t is not None, f'{machine}'
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
    
    def forward_update(self, x, t, scorer: Scorer=None, update_theta: bool=True):
        x = self.processors.forward(x)
        return self.machine.forward_update(x, t, scorer, update_theta)

    def backward_update(self, x, t, outs=None, update_theta: bool=True, update_inputs: bool= True):
        if outs is None:
            _, outs = self.output_ys(x)

        y_in = self.processors.get_y_out(outs[0])
        t = self.machine.backward_update(y_in, t, outs[1], update_theta, update_inputs)
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
                    f'Input Evaluation {i}':k
                    for i, k in enumerate(results['inputs']['evaluations'])
                }

                df_results.append({
                    'Layer': name,
                    'Step': i,
                    'Input Deviation': results['inputs']['deviation'],
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


