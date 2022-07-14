
import typing
import torch as th
import torch.nn as nn
from abc import ABC
import torch
import numpy as np
import pandas as pd

from . import utils

from .modules import SklearnModule
from .optimizers import (
    GradInputOptim, GradThetaOptim, GradThetaOptim, GradInputOptim, SklearnThetaOptim
)
from .optim_builders import ThetaOptimBuilder, SklearnOptimBuilder, InputOptimBuilder
from .base import MachineObjective, Objective, ObjectivePair, ParameterizedMachine, Machine, BatchAssessment, Result, Score, TorchScore


class TorchNN(ParameterizedMachine):
    """Layer that wraps a Torch neural network
    """

    def __init__(
        self, module: nn.Module, loss_factory: typing.Type[nn.Module], 
        theta_updater: ThetaOptimBuilder=None, input_updater: InputOptimBuilder=None,
        update_theta: bool=True
    ):
        """initializer

        Args:
            module (nn.Module): Torch neural network
            objective (Objective)
            theta_updater (ThetaOptimBuilder, optional): Optimizer used to update theta. Defaults to None.
            input_updater (InputOptimBuilder, optional): Optimizer used to update the inputs. Defaults to None.
            fixed (bool, optional): Whether theta is fixed or learnable. Defaults to False.
        """
        super().__init__(update_theta)
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
        self._maximize = False

    def assess_output(self, y: torch.Tensor, t: torch.Tensor):
        evaluation = self._score(y, t, reduce=False)
        return BatchAssessment(evaluation, evaluation, False)
    
    @property
    def theta(self):
        return utils.get_parameters(self._module)
    
    @theta.setter
    def theta(self, theta: torch.Tensor):
        utils.set_parameters(theta, self._module)

    def forward(self, x: torch.Tensor, full_output: bool=False):
        x = utils.freshen(x)
        y = self._module.forward(x)
        if full_output:
            return y, Result(x).update(y)
        return y
    
    def forward_update(self, x, t, outer: Objective=None):

        if self._update_theta:
            inner_objective = MachineObjective(self)
            objective = inner_objective if outer is None else ObjectivePair(inner_objective, outer)
            self._theta_updater.step(x, t, objective=objective)
 
        y = self.forward(x)
        return y

    def backward_update(self, x, t, result: Result=None, update_inputs: bool= True) -> torch.Tensor:
        
        x = utils.freshen(x)

        if self._update_theta:
            self._theta_updater.step(x, t, objective=MachineObjective(self), result=result)
        
        if update_inputs:
            x_prime, _ = self._input_updater.step(x, t, objective=MachineObjective(self), result=result)
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
        self, module: SklearnModule, scorer: TorchScore, 
        theta_updater: SklearnOptimBuilder, input_updater: InputOptimBuilder,
        update_theta: bool=True, partial: bool=False
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
        super().__init__(update_theta)
        self._module = module
        self._theta_updater: SklearnThetaOptim = theta_updater(self._module)
        self._input_updater = input_updater(self._module)
        self._partial = partial
        self._fit = False
        self._score = scorer

    def assess_output(self, y: torch.Tensor, t: torch.Tensor)-> BatchAssessment:
        evaluation = self._score(y, t, reduce=False)
        return BatchAssessment(evaluation, evaluation, False)

    def forward(self, x: torch.Tensor, full_output: bool=False):
        device = x.device
        y = self._module.forward(x).to(device)
        if full_output:
            return y, Result(x).update(y)
            
        return y

    def forward_update(self, x, t, outer: Objective=None):
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
    
    def backward_update(self, x, t, result: Result=None, update_inputs: bool= True) -> torch.Tensor:
        
        if self._update_theta:
            self._theta_updater.step(x, t, MachineObjective(self), result=result)
        
        if update_inputs:
            x_prime, _ = self._input_updater.step(x, t, objective=MachineObjective(self), result=result)
    
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
        super().__init__(update_theta=False)
        self._f = f
        self._input_updater = input_updater(f)
        self._score = score

    def assess_output(self, y: torch.Tensor, t: torch.Tensor)-> BatchAssessment:
        return self._score(y, t)

    def forward(self, x: torch.Tensor, full_output: bool=False):
        y = self._f(x)
        if full_output:
            return y, Result(x).update(y)

        return y

    def forward_update(self, x, t, outer: Objective=None):
        y = self.forward(x)
        return y
    
    def backward_update(self, x, t, result: Result=None, update_inputs: bool= True) -> torch.Tensor:
        
        if update_inputs:
            return self._input_updater.step(x, t, objective=MachineObjective(self), result=result)


class Sequence(Machine):
    """Wraps multiple layers that execut in succession
    """

    def __init__(self, machines: typing.List[Machine], update_theta: bool=True):
        """intializer

        Args:
            machines (typing.List[Machine]): Machines to execute

        Raises:
            ValueError: If the sequence is empty
        """
        super().__init__(update_theta)
        if len(machines) == 0:
            raise ValueError(f'Length of sequence must be greater than 0')
        self.machines = machines
    
    def assess_output(self, y: torch.Tensor, t: torch.Tensor)-> BatchAssessment:
        return self.machines[-1].assess_output(
            y, t
        )

    def forward(self, x: torch.Tensor, full_output: bool=False):
        result = Result(x)

        for layer in self.machines:
            y = layer.forward(result.y, full_output)
            if full_output:
                result.update(*y)
                y, _ = y

        if full_output:
            return y, result
        return y

    def forward_update(self, x, t, outer: Objective=None):
        if not self._update_theta:
            y = self.forward(x)
            return y

        y = x
        for i, machine in enumerate(self.machines):
            if i < len(self.machines) - 1 and outer is None:
                
                cur_objective = MachineObjective(Sequence(self.machines[i + 1:]))
            elif i < len(self.machines) - 1:
                # TODO: This will not factor in the regularizations from the sub machines
                # So need to improve on this
                cur_objective = ObjectivePair(MachineObjective(Sequence(self.machines[i + 1:])), outer)
            else:
                cur_objective = outer
            y = machine.forward_update(y, t, cur_objective)

        return y

    def backward_update(self, x, t, result: Result=None, update_inputs: bool= True) -> torch.Tensor:
        if result is None:
            _, result = self.forward(x, full_output=True)
        
        for i in reversed(range(self.machines)):
            _update_inputs = update_inputs if i > 0 else True
            machine = self.machines[i]
            result_i = result.outputs[i][1]
            x_i = x if i == 0 else result.outputs[i - 1][0]

            t = machine.backward_update(x_i, t, result_i, _update_inputs)
            assert t is not None, f'{machine}'
        
        return t


class WeightedLoss(nn.Module):
    
    def __init__(self, nn_loss: nn.Module, weight: float):

        super().__init__()
        self.nn_loss = nn_loss
        self.weight = weight

    def forward(self, x, t):

        return self.weight * self.nn_loss(x, t)


# class Transform(Machine):
#     """Layer that transforms an input and does inverse transform on targets
#     """

#     def __init__(
#         self, f, inv_f, objective: Objective,
#     ):
#         """initializer

#         Args:
#             objective (Objective)
#         """
#         self._f = f
#         self._inv_f = inv_f
#         self._objective = objective

#     def assess(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, batch_assess: bool=True):
#         return self._objective(y, t)

#     def forward(self, x: torch.Tensor, get_ys: bool=False):
#         if get_ys:
#             x = x.detach().requires_grad_()
#             x.retain_grad()
#             y = self.forward(x)
#             return y, [x, y]
#         return self._f(x)
    
#     def forward_update(self, x, t, scorer: Score=None):
#         return self.forward(x)

#     def backward_update(self, x, t, outs=None, update_inputs: bool= True):
#         if outs is not None:
#             y = self.get_y_out(outs)
#             x = self.get_in(outs)
#         else:
#             x = x.detach().requires_grad_()
#             x.retain_grad()
#             y = self._f(x)

#         if update_inputs:
#             x_prime = self._inv_f(y)
#             assert x_prime is not None, f'{self._inv_f}'
#             return x_prime




# class CompositeProcessor(Processor):

#     def __init__(self, processors: typing.List[Processor]):
#         self.processors = processors
    
#     def forward(self, x: torch.Tensor, get_ys: bool=False):
#         y = x
#         ys = []
#         if get_ys:
#             ys.append(x)
#         for processor in self.processors:
#             y = processor.forward(y)
#             if get_ys:
#                 ys.append(y)
        
#         if get_ys:
#             return y, ys
#         return y

#     def backward(self, x, t, outs=None):

#         if outs is None:
#             _, outs = self.output_ys(x)
        
#         xs = outs[:-1]
#         for x_i, y_i, processor in zip(
#             reversed(xs), reversed(outs[1:-1]), reversed(self.processors)):
#             t = processor.backward(x_i, t, y_i)
#         return t

#     def get_y_out(self, outs):
#         if len(self.processors) == 0:
#             return outs[0]
#         return self.processors[-1].get_y_out(outs[-1])


# class Processed(Machine):
    
#     def __init__(
#         self, processors: typing.List[Processor], 
#         machine: Machine
#     ):
#         self.processors = CompositeProcessor(processors)
#         self.machine = machine
    
#     def assess_output(self, y: torch.Tensor, t: torch.Tensor)-> BatchAssessment:
#         return self.machine.assess_output(y, t)

#     def forward(self, x: torch.Tensor, full_output: bool=False):
        
#         result = Result(x, full_output)
        
#         result.update(self.processors.forward(x, full_output))
#         result.update(self.machine.forward(result.y, full_output))
#         return result.output
    
#     @property
#     def maximize(self) -> bool:
#         return self.machine.maximize
    
#     def forward_update(self, x, t, outer: Objective=None):
#         x = self.processors.forward(x)
#         return self.machine.forward_update(x, t, outer)

#     def backward_update(self, x, t, result: Result=None, update_inputs: bool= True) -> torch.Tensor:
#         if result is None:
#             _, result = self.forward(x, True)
        
#         y_in = result.outputs[0][1].y
#         result_machine = result.outputs[1][1]
#         t = self.machine.backward_update(y_in, t, result_machine, update_inputs)
#         if update_inputs:
#             return self.processors.backward(x, t, result.outputs[0][1])



# if you want to use decision tree
# probably need to have a mixture of experts
# randomly choose what expert to update for each 
# sample... 
#
# output = 1 2 1 2 2 2
# update = 0 0 0 0 0 1 <- some small probability of updating
# otherwise the changes will be too great
# 



# class TorchGradNN(ParameterizedMachine):
#     """Layer that wraps a Torch neural network
#     """

#     def __init__(
#         self, module: nn.Module, loss_factory: typing.Type[nn.Module], 
#         theta_updater: ThetaGradOptimBuilder=None,
#         input_updater: InputOptimBuilder=None, update_theta: bool=False,
#         update_reps: int=1
#     ):
#         """initializer

#         Args:
#             module (nn.Module): Torch neural network
#             objective (Objective)
#             theta_updater (ThetaOptimBuilder, optional): Optimizer used to update theta. Defaults to None.
#             input_updater (InputOptimBuilder, optional): Optimizer used to update the inputs. Defaults to None.
#             fixed (bool, optional): Whether theta is fixed or learnable. Defaults to False.
#         """
#         super().__init__(update_theta)
#         self._module = module
#         self._score = TorchScore(loss_factory)
#         if optim_factory is None:
#             self._optim = torch.optim.Adam(module.parameters(), lr=1e-2)
#         else:
#             self._optim = optim_factory(module.parameters())
#         if input_updater is not None:
#             self._input_updater: InputOptim = input_updater(self)
#         else: self._input_updater = None 
#         self._update_theta = update_theta
#         self._update_reps = update_reps

#     @property
#     def update_reps(self): return self._update_reps

#     @update_reps.setter
#     def update_reps(self, reps: int):
#         assert reps > 0
#         self._update_reps = reps

#     def update_optim(self, optim_factory):
#         self._optim = optim_factory(self._module.parameters())

#     def assess_output(self, y: torch.Tensor, t: torch.Tensor):
#         evaluation = self._score(y, t, reduce=False)
#         return BatchAssessment(evaluation, evaluation, False)

#     @property
#     def theta(self):
#         return utils.get_parameters(self._module)
    
#     @theta.setter
#     def theta(self, theta: torch.Tensor):
#         utils.set_parameters(theta, self._module)

#     def forward(self, x: torch.Tensor, full_output: bool=False):
#         x = utils.freshen(x)
#         y = self._module.forward(x)
#         if full_output:
#             return y, Result(x).update(y)
#         return y
    
#     def forward_update(self, x, t, outer: Objective=None):
#         if self._update_theta:
#             y, result = self.forward(x, True)
#             if outer is not None:
#                 asseesment = outer.assess(y, t)
#             else:
#                 assessment = self.assess_output(y, t)
#             assessment += result.regularization
#             self._optim.zero_grad()
#             asseesment.mean().backward()
#             self._optim.step()
 
#         return self.forward(x)

#     def backward_update(self, x, t, result: Result=None, update_inputs: bool=True) -> torch.Tensor:

#         self._optim.zero_grad()
#         if result is None:
#             x = utils.freshen(x)
#             assessment = self.assess(x, t, True)
#         else:
#             assessment = self.assess_output(result.y, t) + result.regularization

#         assessment.mean().backward()
#         if self._update_theta:
#             self._optim.step()

#         for _ in range(self._update_reps if self._update_theta else 0):
#             self._optim.zero_grad()
#             x = utils.freshen(x)
#             assessment = self.assess(x, t, True)
#             assessment.mean().backward()
#             self._optim.step()

#         if update_inputs:
#             x_prime, _ = self._input_updater.step(x, t, MachineObjective(self))
#             return x_prime

#     @property
#     def maximize(self) -> bool:
#         False

#     @property
#     def module(self):
#         return self._module