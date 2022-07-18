
import typing
import torch
import torch.nn as nn
from . import utils
from .modules import SklearnModule
from .optimizers import (
    GradInputOptim, GradThetaOptim, GradThetaOptim, GradInputOptim, SklearnThetaOptim
)
from .optim_builders import ThetaOptimBuilder, SklearnOptimBuilder, InputOptimBuilder
from .base import Objective, ParameterizedMachine, Machine, BatchAssessment, Result, Score, TorchScore


class OuterPair(Objective):    
    
    def __init__(
        self, first: Objective, second: Objective=None
    ):
        """initializer

        Args:
            module (nn.Module): Torch neural network
            objective (Objective)
            theta_updater (ThetaOptimBuilder, optional): Optimizer used to update theta. Defaults to None.
            input_updater (InputOptimBuilder, optional): Optimizer used to update the inputs. Defaults to None.
            fixed (bool, optional): Whether theta is fixed or learnable. Defaults to False.
        """
        super().__init__()
        self._first = first
        self._second = second

    def differentiable(self) -> bool:
        if self._second:
            return self._first.differentiable and self._second.differentiable
        return self._first.differentiable

    def assess_output(self, y: torch.Tensor, t: torch.Tensor):
        if self._second is None:
            return self._first.assess_output(y, t)
        return self._second.assess_output(y, t)

    def forward(self, x: torch.Tensor, full_output: bool=False):
        if self._second is None:
            return self._first.forward(x, full_output)
        elif full_output:
            result = Result(x, self.maximize)
            y, result1 = self._first.forward(x, True)
            result.update(y, result1)
            y, result2 = self._second.forward(y, True)
            result.update(y, result2)
            return y, result

        return self._second.forward(self._first.forward(x))

    @property
    def maximize(self) -> bool:
        return self._first.maximize


class TorchNN(ParameterizedMachine):
    """Layer that wraps a Torch neural network
    """

    def __init__(
        self, module: nn.Module, loss_factory: typing.Type[nn.Module], 
        theta_updater: ThetaOptimBuilder=None, input_updater: InputOptimBuilder=None,
        update_theta: bool=True, differentiable: bool=True
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
        self._differentiable = differentiable

    def differentiable(self) -> bool:
        return self._differentiable

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
            return y, Result(x, self.maximize).update(y)
        return y
    
    def forward_update(self, x, t, outer: Objective=None):
        if self._update_theta:
            self._theta_updater.step(x, t, OuterPair(self, outer))
 
        y = self.forward(x)
        return y

    def backward_update(self, x, t, result: Result=None, update_inputs: bool= True) -> torch.Tensor:

        if self._update_theta:
            self._theta_updater.step(x, t, self, result=result)
        
        if update_inputs:
            x_prime, _ = self._input_updater.step(x, t, self, result=result)
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

    def differentiable(self) -> bool:
        return False

    def assess_output(self, y: torch.Tensor, t: torch.Tensor)-> BatchAssessment:
        evaluation = self._score(y, t, reduce=False)
        return BatchAssessment(evaluation, evaluation, False)

    def forward(self, x: torch.Tensor, full_output: bool=False):
        device = x.device
        y = self._module.forward(x).to(device)
        if full_output:
            return y, Result(x, self.maximize).update(y)
            
        return y

    @property
    def maximize(self) -> bool:
        return self._score.maximize

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
            self._theta_updater.step(x, t, self, result=result)
        
        if update_inputs:
            x_prime, _ = self._input_updater.step(x, t, objective=self, result=result)
    
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
            return y, Result(x, self.maximize).update(y)

        return y

    def forward_update(self, x, t, outer: Objective=None):
        y = self.forward(x)
        return y
    
    def backward_update(self, x, t, result: Result=None, update_inputs: bool= True) -> torch.Tensor:
        
        if update_inputs:
            return self._input_updater.step(x, t, self, result=result)

    @property
    def differentiable(self) -> bool:
        return False

    @property
    def maximize(self) -> bool:
        return self._score.maximize


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

    def differentiable(self) -> bool:
        for machine in self.machines:
            if not machine.differentiable: return False
        return True

    def forward(self, x: torch.Tensor, full_output: bool=False):
        result = Result(x, self.maximize)
        sub_result = None
        for layer in self.machines:
            if full_output:
                y, sub_result = layer.forward(result.y, True)
            else:
                y = layer.forward(result.y, False)
            result.update(y, sub_result)

        if full_output:
            return y, result
        return y

    def forward_update(self, x, t, outer: Objective=None):
        if not self._update_theta:
            y = self.forward(x)
            return y

        y = x
        for i, machine in enumerate(self.machines):
            if i == len(self.machines) - 1: # final machine in sequence
                cur_objective = outer
            else:
                cur_objective = OuterPair(Sequence(self.machines[i+1:]), outer)
            y = machine.forward_update(y, t, cur_objective)

        return y

    def backward_update(self, x, t, result: Result=None, update_inputs: bool= True) -> torch.Tensor:
        if result is None:
            _, result = self.forward(x, full_output=True)
        
        for i in reversed(range(len(self.machines))):
            _update_inputs = update_inputs if i > 0 else True
            machine = self.machines[i]
            y_i, sub_result = result[i]
            x_i = x if sub_result is None else sub_result.x

            t = machine.backward_update(x_i, t, sub_result, _update_inputs)
            assert t is not None, f'{machine}'
        
        return t
    
    @property
    def maximize(self):
        return self.machines[0].maximize


class TargetTransform(Machine):
    """Layer that transforms an input and does inverse transform on targets
    """

    def __init__(
        self, loss_reverse, lr: float=1e-2
    ):
        """initializer

        Args:
            objective (Objective)
        """
        super().__init__(False)
        self._lr = lr
        self._loss_reverse = loss_reverse
        self._score = TorchScore(self._loss_reverse.loss)

    def assess_output(self, y: torch.Tensor, t: torch.Tensor)-> BatchAssessment:
        return self._loss_reverse.score(y, t)

    def forward(self, x: torch.Tensor, full_output: bool=False):
        if full_output:
            return x, Result(x, self.maximize)
        return x
    
    def forward_update(self, x, t, outer: Objective=None):
        return x

    def backward_update(self, x, t, result: Result=None, update_inputs: bool= True) -> torch.Tensor:

        if update_inputs:
            return self._loss_reverse.reverse(x, t, self._lr)
