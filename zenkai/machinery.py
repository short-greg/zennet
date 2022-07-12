

from functools import partial, singledispatchmethod
import typing
import torch as th
import torch.nn as nn
from abc import ABC
import torch
import numpy as np
import torch.nn.utils as nn_utils
import pandas as pd

from . import utils

from .modules import SklearnModule
from .optimizers import (
    GradInputOptim, GradThetaOptim, GradThetaOptim, GradInputOptim
)
from .optim_builders import ThetaOptimBuilder, SklearnOptimBuilder, InputOptimBuilder
from .base import MachineObjective, Objective, ObjectivePair, ParameterizedMachine, Processor, Machine, BatchAssessment, Result, Score, TorchScore


class TorchNN(ParameterizedMachine):
    """Layer that wraps a Torch neural network
    """

    def __init__(
        self, module: nn.Module, loss_factory: typing.Type[nn.Module], 
        theta_updater: ThetaOptimBuilder=None, input_updater: InputOptimBuilder=None,
        fixed: bool=False
    ):
        """initializer

        Args:
            module (nn.Module): Torch neural network
            objective (Objective)
            theta_updater (ThetaOptimBuilder, optional): Optimizer used to update theta. Defaults to None.
            input_updater (InputOptimBuilder, optional): Optimizer used to update the inputs. Defaults to None.
            fixed (bool, optional): Whether theta is fixed or learnable. Defaults to False.
        """
        self._module = module
        self._score = TorchScore(loss_factory)
        self._theta_updater =  (
            theta_updater(self)
            if theta_updater is not None else GradThetaOptim (module)
        )
        self._input_updater = (
            input_updater(self) 
            if input_updater is not None else GradInputOptim(module, skip_eval=True)
        )
        self._fixed = fixed
        self._maximize = False

    def assess(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, batch_assess: bool=True):
        if y is None:
            y, reg = self.forward(x, get_reg=True)
        else:
            reg = None
        evaluation = self._score(y, t, reduce=not batch_assess)
        return BatchAssessment(evaluation, evaluation, False) + reg
    
    @property
    def theta(self):
        return utils.get_parameters(self._module)
    
    @theta.setter
    def theta(self, theta: torch.Tensor):
        utils.set_parameters(theta, self._module)

    def forward(self, x: torch.Tensor, get_ys: bool=False, get_reg: bool=False):
        result = Result(x, get_ys, get_reg)
        result.update(self._module.forward(result.y))
        return result.output
    
    def forward_update(self, x, t, outer: Objective=None, update_theta: bool=True):

        if update_theta and not self._fixed:
            inner_objective = MachineObjective(self)
            objective = inner_objective if outer is None else ObjectivePair(inner_objective, outer)
            self._theta_updater.step(x, t, objective=objective)
 
        y = self.forward(x)
        return y

    def backward_update(self, x, t, ys=None, update_theta: bool=True, update_inputs: bool= True) -> torch.Tensor:
        if ys is not None:
            y = ys[-1]
            x = ys[0]
        else:
            x = x.detach().requires_grad_()
            x.retain_grad()
            y = self._module(x)

        if update_theta and not self._fixed:            
            self._theta_updater.step(x, t, objective=MachineObjective(self), y=y)
        
        if update_inputs:
            x_prime, _ = self._input_updater.step(x, t, objective=MachineObjective(self), y=y)
            return x_prime

    @property
    def maximize(self) -> bool:
        False

    @property
    def module(self):
        return self._module


class SklearnMachine(Machine):
    """Layer that wraps a scikit-learn machine
    """

    def __init__(
        self, module: SklearnModule, objective: Objective, 
        theta_updater: SklearnOptimBuilder, input_updater: InputOptimBuilder,
        fixed: bool=False, partial: bool=False
    ):
        """initializer
        Args:
            module (SklearnModule)
            objective (Objective)
            theta_updater (SklearnOptimBuilder)
            input_updater (InputOptimBuilder): _description_
            fixed (bool, optional): Whether the parameters are fixed. Defaults to False.
            partial (bool, optional): Whether to use partial_fit() (True) or fit(). Defaults to False.
        """
        super().__init__()
        self._module = module
        self._objective = objective
        self._theta_updater = theta_updater(self._module, objective)
        self._input_updater = input_updater(self._module, objective)
        self._fixed = fixed
        self._partial = partial
        self._fit = False

    def fix(self, fixed: bool=True):
        self._fixed = fixed

    def assess(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, batch_assess: bool=True):
        # # TODO: Check
        # if y is None:
        #     y = self.forward(x)
        # else:
        #     outs = outs
        return self._objective(y, t)

    def forward(self, x: torch.Tensor, get_ys: bool=False):
        device = x.device
        y_np = self._module.forward(x)
        return y_np.to(device)

    def forward_update(self, x, t, objective: Objective=None, update_theta: bool=True):
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
    
    def backward_update(self, x, t, ys=None, update_theta: bool=True, update_inputs: bool=True):
        
        if update_theta and not self._fixed:
            self._theta_updater.step(x, t)
        
        if update_inputs:
            x_prime = self._input_updater.step(x, t, objective=MachineObjective(self))
    
            return x_prime


class BlackboxMachine(Machine):
    """Layer that wraps a function that does not learn
    """

    def __init__(self, f, score: Score, input_updater: InputOptimBuilder):
        """initializer

        Args:
            f: Function/Callable wrapped by layer
            loss (_type_): _description_
            input_updater (InputOptimBuilder): _description_
        """
        super().__init__()
        self._f = f
        self._input_updater = input_updater(f)

    def assess(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, batch_assess: bool=True):
        if y is None:
            y = self.forward(x)
        else:
            outs = outs
        return self._objective(y, t)

    def forward(self, x: torch.Tensor, get_ys: bool=False, get_reg: bool=False):
        result = Result(x, get_ys, get_reg)
        result.update(self._f(result.y))
        return result.output

    def forward_update(self, x, t, objective: Objective=None, update_theta: bool=True):
        y = self.forward(x)
        return y
    
    def backward_update(self, x, t, ys=None, update_theta: bool=True, update_inputs: bool= True):
        
        if update_inputs:
            return self._input_updater.step(x, t, objective=MachineObjective(self))


class Sequence(Machine):
    """Wraps multiple layers that execut in succession
    """

    def __init__(self, machines: typing.List[Machine]):
        """intializer

        Args:
            machines (typing.List[Machine]): Machines to execute

        Raises:
            ValueError: If the sequence is empty
        """
        if len(machines) == 0:
            raise ValueError(f'Length of sequence must be greater than 0')
        self.machines = machines
    
    def assess(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, batch_assess: bool=True):
        
        return self.machines[-1].assess(
            y, t
        )

    def forward(self, x: torch.Tensor, get_ys: bool=False, get_reg: bool=False):
        result = Result(x, get_ys, get_reg)

        for layer in self.machines:
            result.update(layer.forward(result.y))
        return result.output

    def forward_update(self, x, t, objective: Objective=None, update_theta: bool=True):
        if not update_theta or self._fixed:
            y = self.forward(x)
            return y
        y = x
        for i, machine in enumerate(self.machines):
            if i < len(self.machines) - 1 and objective is None:
                
                cur_objective = MachineObjective(Sequence(self.machines[i + 1:]))
            elif i < len(self.machines) - 1:
                # TODO: This will not factor in the regularizations from the sub machines
                # So need to improve on this
                cur_objective = ObjectivePair(MachineObjective(Sequence(self.machines[i + 1:])), objective)
            else:
                cur_objective = objective
            y = machine.forward_update(y, t, cur_objective, update_theta)

        return y

    def backward_update(self, x, t, ys=None, update_theta: bool=True, update_inputs: bool= True):
        if ys is None:
            _, ys = self.forward(x, )
        xs = [x]
        for y_i, machine in zip(ys[1:-1], self.machines[:-1]):
            xs.append(machine.get_y_out(y_i))

        for i, (x_i, y_i, machine) in enumerate(zip(reversed(xs), reversed(ys[1:]), reversed(self.machines))):
            _update_inputs = i < len(xs) or update_inputs
            t = machine.backward_update(x_i, t, y_i, update_theta, _update_inputs)
        
            assert t is not None, f'{machine}'
        return t


class CompositeProcessor(Processor):

    def __init__(self, processors: typing.List[Processor]):
        self.processors = processors
    
    def forward(self, x: torch.Tensor, get_ys: bool=False):
        y = x
        ys = []
        if get_ys:
            ys.append(x)
        for processor in self.processors:
            y = processor.forward(y)
            if get_ys:
                ys.append(y)
        
        if get_ys:
            return y, ys
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
    
    def assess(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, batch_assess: bool=True):
        y_in = self.processors.forward(x, False)
        return self.machine.assess(y_in, t, y, batch_assess=batch_assess)

    def forward(self, x: torch.Tensor, get_ys: bool=False, get_reg: bool=False):
        
        result = Result(x, get_ys, get_reg)
        
        result.update(self.processors.forward(x))
        result.update(self.machine.forward(result.y))
        return result.output
    
    @property
    def maximize(self) -> bool:
        return self.machine.maximize
    
    def forward_update(self, x, t, scorer: Score=None, update_theta: bool=True):
        #  objective=self._input_objective(x, t)
        x = self.processors.forward(x)
        return self.machine.forward_update(x, t, scorer, update_theta)

    def backward_update(self, x, t, outs=None, update_theta: bool=True, update_inputs: bool= True):
        if outs is None:
            _, outs = self.output_ys(x)

        y_in = self.processors.get_y_out(outs[0])
        t = self.machine.backward_update(y_in, t, outs[1], update_theta, update_inputs)
        if update_inputs:
            return self.processors.backward(x, t, outs[0])


class WeightedLoss(nn.Module):
    
    def __init__(self, nn_loss: nn.Module, weight: float):

        super().__init__()
        self.nn_loss = nn_loss
        self.weight = weight

    def forward(self, x, t):

        return self.weight * self.nn_loss(x, t)


class Transform(Machine):
    """Layer that transforms an input and does inverse transform on targets
    """

    def __init__(
        self, f, inv_f, objective: Objective
    ):
        """initializer

        Args:
            objective (Objective)
        """
        self._f = f
        self._inv_f = inv_f
        self._objective = objective

    def assess(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, batch_assess: bool=True):
        return self._objective(y, t)

    def forward(self, x: torch.Tensor, get_ys: bool=False):
        if get_ys:
            x = x.detach().requires_grad_()
            x.retain_grad()
            y = self.forward(x)
            return y, [x, y]
        return self._f(x)
    
    def forward_update(self, x, t, scorer: Score=None, update_theta: bool=True):
        return self.forward(x)

    def backward_update(self, x, t, outs=None, update_theta: bool=True, update_inputs: bool= True):
        if outs is not None:
            y = self.get_y_out(outs)
            x = self.get_in(outs)
        else:
            x = x.detach().requires_grad_()
            x.retain_grad()
            y = self._f(x)

        if update_inputs:
            x_prime = self._inv_f(y)
            assert x_prime is not None, f'{self._inv_f}'
            return x_prime


# if you want to use decision tree
# probably need to have a mixture of experts
# randomly choose what expert to update for each 
# sample... 
#
# output = 1 2 1 2 2 2
# update = 0 0 0 0 0 1 <- some small probability of updating
# otherwise the changes will be too great
# 
