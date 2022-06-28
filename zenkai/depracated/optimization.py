
from abc import ABC, abstractproperty
from abc import ABC, abstractclassmethod, abstractmethod
from functools import partial
import typing
import torch.nn.utils as nn_utils
import torch as th
import torch.nn as nn
from .modules import  Blackbox, Objective, SklearnModule


class Scorer(ABC):

    @abstractmethod
    def assess(self, x: th.Tensor) -> th.Tensor:
        pass

    @abstractproperty
    def maximize(self):
        pass


class Optimizer(ABC):

    @abstractmethod
    def reset_inputs(self, inputs):
        pass

    @abstractmethod
    def reset_theta(self, theta):
        pass

    @abstractmethod
    def update_theta(self, t, inputs=None, theta=None, y=None, scorer: Scorer=None):
        pass
    
    @abstractmethod
    def update_inputs(self, t, inputs=None, theta=None, y=None, scorer: Scorer=None):
        pass

    @abstractproperty
    def inputs(self):
        pass
    
    @abstractproperty
    def theta(self):
        pass

    @abstractproperty
    def evaluations(self):
        pass


class TorchOptimizer(Optimizer):

    def __init__(self, net: nn.Module, objective: Objective):
        super().__init__()
        self._objective = objective
        self._net = net
        self._inputs = None
        self._theta = None

    def update_theta(self, t, inputs=None, theta=None, y=None, scorer: Scorer=None):
        if inputs is not None:
            self.reset_inputs(inputs)
        if theta is not None:
            self.reset_theta(theta)
        
        self._evaluations = self._update_theta(t, y, scorer)

    def update_inputs(self, t, inputs=None, theta=None, y=None, scorer: Scorer=None):
        if inputs is not None:
            self.reset_inputs(inputs)
        if theta is not None:
            self.reset_theta(theta)
        
        self._evaluations = self._update_inputs(t, y, scorer=scorer)
    
    @abstractmethod
    def _update_inputs(self, t, y=None, scorer: Scorer=None):
        pass

    @abstractmethod
    def _update_theta(self, t, y=None, scorer: Scorer=None):
        pass

    @property
    def evaluations(self):
        return self._evaluations


class GradOptimizer(TorchOptimizer):

    def __init__(self, net: nn.Module, objective: Objective, optim, input_weight: float=1e-1):
        super().__init__(net, objective)
        self._net = net
        self._optim: th.optim.Optimizer = optim(self._net.parameters())
        self._input_weight = input_weight

    def reset_theta(self, theta=None):
        self._theta = theta
        nn_utils.vector_to_parameters(theta, self._net.parameters())
        self._evaluation = None

    def reset_inputs(self, inputs):
        self._inputs = inputs
        self._evaluation = None

    def _update_theta(self, t, y=None, scorer: Scorer=None):
        self._optim.zero_grad()
        x = self._inputs.detach()

        # if y is None:
        y = self._net(x)
        if scorer is None:
            evaluation = self._objective(y, t)
        else:
            evaluation = scorer.assess(y)
        evaluation.mean().backward()
        self._optim.step()
        return evaluation.view(1, -1)

    def _update_inputs(self, t, y=None, scorer: Scorer=None):
        
        x: th.Tensor = self._inputs.detach()
        x.requires_grad_()
        x.retain_grad()
        # if y is None:
        y = self._net(x)

        if scorer is None:
            evaluation = self._objective(y, t)
        else:
            evaluation = scorer.assess(y)
        if x.grad is None:
            evaluation.sum().backward()
        if x.grad is None:
            raise RuntimeError("Input is not a leaf node or does not retain grad")
        self._inputs = self._inputs - self._input_weight * x.grad
        return evaluation.view(1, -1)

    @property
    def inputs(self):
        return self._inputs
    
    @property
    def theta(self):
        return nn.utils.parameters_to_vector(self._net.parameters())
    

class XPOptimizer(Optimizer):

    def __init__(self, x_updater: Optimizer, p_updater: Optimizer):
        super().__init__()
        self._x_updater: Optimizer = x_updater
        self._p_updater: Optimizer = p_updater
        self._updated_inputs = False
        self._updated_theta = False

    def reset_theta(self, theta=None):
        self._x_updater.reset_theta(theta)
        self._p_updater.reset_theta(theta)

    def reset_inputs(self, inputs):
        self._x_updater.reset_inputs(inputs)
        self._p_updater.reset_inputs(inputs)

    def update_theta(self, t, inputs=None, theta=None, y=None, scorer: Scorer=None):
        if inputs is not None:
            self._x_updater.reset_inputs(inputs)
        if theta is not None:
            self._x_updater.reset_theta(theta)
        self._p_updater.update_theta(t, theta=theta, inputs=inputs, y=y, scorer=scorer)
        self._x_updater.reset_theta(self._p_updater.theta)
        self._updated_inputs = False
        self._updated_theta = True

    def update_inputs(self, t, inputs: th.Tensor=None, theta=None, y=None, scorer: Scorer=None):
        if inputs is not None:
            self._p_updater.reset_inputs(inputs)
        if theta is not None:
            self._p_updater.reset_theta(theta)
        self._x_updater.update_inputs(t, theta=theta, inputs=inputs, y=y, scorer=scorer)
        self._p_updater.reset_inputs(self._x_updater.inputs)
        self._updated_inputs = True
        self._updated_theta = False

    @property
    def inputs(self):
        return self._x_updater.inputs
    
    @property
    def theta(self):
        return self._p_updater.theta
    
    @property
    def evaluations(self):
        if self._updated_inputs:
            return self._x_updater.evaluations
        if self._updated_theta:
            return self._p_updater.evaluations


class HillClimberProcessor(ABC):

    @abstractmethod
    def __call__(self, x):
        pass
    
    @abstractmethod
    def spawn(self):
        pass
    
    @abstractmethod
    def report(self, evaluations):
        pass


def get_best(x: th.Tensor, evaluations, maximize: bool=True):
    if maximize:
        idx = th.argmax(evaluations)
    else:
        idx = th.argmin(evaluations)
    return x[idx]


class StepHillClimberProcessor(HillClimberProcessor):

    def __init__(self, s_mean: int=-2, s_std=1, k: int=1, momentum: float=None, maximize: bool=True, update_after: int=20):
        self.s_mean = s_mean
        self.s_std = s_std
        self.k = k
        self.maximize = maximize
        self._momentum = momentum
        self._diff = None
        self._x_updated = None
        self._out = None
        self._s = None
        self._step_evaluations = {}
        self._keep_s = 100
        self._i = 0
        self._update_after = update_after
        self._mean_offset = 0.
    
    def reset_state(self):
        self._diff = None
        self._x_updated = None
        self._out = None
        self._s = None

    def __call__(self, x: th.Tensor):
        
        self._x = x
        self._s = th.randn(self.k)  * self.s_std + self.s_mean
        s = (10 ** self._s).view(-1, *([1] * x.dim())).repeat(1, *x.size())
        y = th.randn(self.k, *x.size()) * s + x
        self._out = th.cat([x.unsqueeze(0), y])
        return self._out
    
    def report(self, evaluations: th.Tensor):

        self._step_evaluations = {
            k: v for i, (k, v) in enumerate(
                sorted(self._step_evaluations.items(), key=lambda item: item[1], reverse=self.maximize)
            ) if i < self._keep_s
        }
        self._step_evaluations.update(dict(zip(self._s.tolist(), evaluations[1:].tolist())))

        # self.s_mean = statistics.mean(self._step_evaluations.keys())
        # # print(self.s_mean)
        # if len(self._step_evaluations) > 2:
        #    self.s_std = statistics.stdev(self._step_evaluations.keys())
        
        # self.s_mean = self._mean_offset

        if self._out is None:
            raise RuntimeError('Have not processed any inputs yet (i.e. use __call__)')
        best = get_best(self._out, evaluations, self.maximize)
        if self._diff is not None and self._momentum is not None:
            self._diff = (1 - self._momentum) * (best - self._x) + self._momentum * self._diff
            x_updated = self._x + self._diff
        elif self._momentum is not None:
            self._diff = (best - self._x)
            x_updated = self._x + self._diff
        else:
            x_updated = best
        self._i += 1
        if self._i % self._update_after == 0:
            self.s_mean -= 0.5
            # self._mean_offset -= 0.5
        return x_updated

    def spawn(self, maximize: bool=None):
        return StepHillClimberProcessor(
            self.s_mean, self.s_std, self.k, self._momentum, self.maximize if maximize is None else maximize
        )


class BinaryHillClimberProcessor(HillClimberProcessor):
    
    def __init__(self, lower: float=0.01, upper: float=0.4, k: int=1, maximize: bool=True, update_after: int=20):
        self.lower = lower
        self.upper = upper
        self.k = k
        self.maximize = maximize
        self._diff = None
        self._x_updated = None
        self._out = None
        self._s = None
        self._step_evaluations = {}
        self._i = 0
        self._update_after = update_after
        self._mean_offset = 0.
    
    def reset_state(self):
        self._diff = None
        self._x_updated = None
        self._out = None
        self._s = None

    def __call__(self, x: th.Tensor):
        
        self._x = x
         
        self._s = th.rand(self.k) * (self.upper - self.lower) + self.lower
        s = (self._s).view(-1, *([1] * x.dim())).repeat(1, *x.size())
        same = th.rand(self.k, *x.size()) >= s
        complement = 1 - same
        y = x * same + (1 - x) * complement
        self._out = th.cat([x.unsqueeze(0), y])
        return self._out
    
    def report(self, evaluations: th.Tensor):

        self._step_evaluations = {
            k: v for i, (k, v) in enumerate(
                sorted(self._step_evaluations.items(), key=lambda item: item[1], reverse=self.maximize)
            ) if i < self._keep_s
        }
        self._step_evaluations.update(dict(zip(self._s.tolist(), evaluations[1:].tolist())))

        if self._out is None:
            raise RuntimeError('Have not processed any inputs yet (i.e. use __call__)')
        x_updated = get_best(self._out, evaluations, self.maximize)

        self._i += 1
        if self._i % self._update_after == 0:
            self.upper *= 0.5
        return x_updated

    def spawn(self, maximize: bool=None):
        return BinaryHillClimberProcessor(
            self.lower, self.upper, self.k, self.maximize if maximize is None else maximize
        )

    

class HillClimberOptimizer(TorchOptimizer):

    def __init__(self, net: nn.Module, objective: Objective, processor: typing .Union[str, StepHillClimberProcessor]="gaussian"):
        super().__init__(net, objective)
        if processor == "gaussian":
            self.input_processor = StepHillClimberProcessor(maximize=objective.maximize)
            self.input_processor = StepHillClimberProcessor(maximize=objective.maximize)
        else:
            self.input_processor = processor.spawn(maximize=objective.maximize)
            self.theta_processor = processor.spawn(maximize=objective.maximize)
        self._inputs_diff = None
        self._theta_diff = None
        self._x = None
        try:
            self._theta = nn_utils.parameters_to_vector(self._net.parameters())
        except NotImplementedError:
            # in some cases the list will be empty
            # such as when an sklearn module will be used. This will
            # raise a notimplementederror
            self._theta = None

    def reset_inputs(self, inputs):
        self._inputs = inputs
        self._evaluation = None
        self._inputs_diff = None
        self.input_processor.reset_state()

    def reset_theta(self, theta):
        self._theta = theta
        nn_utils.vector_to_parameters(theta, self._net.parameters())
        self._evaluation = None
        self._theta_diff = None
        self.theta_processor.reset_state()

    def _update_theta(self, t, y=None, scorer: Scorer=None):
        theta = self.theta_processor(self._theta)
        evaluations = []
        for theta_i in theta:
            nn_utils.vector_to_parameters(theta_i, self._net.parameters())
            y = self._net(self._inputs)
            if scorer is None:
                evaluations.append(self._objective.forward(y, t))
            else:
                evaluations.append(scorer.assess(y))
        evaluations = th.stack(evaluations)
        self._theta = self.theta_processor.report(evaluations)
        return evaluations
    
    def _update_inputs(self, t, y=None, scorer: Scorer=None):

        inputs = self.input_processor(self._inputs)
        y = self._net(inputs.view(-1, *inputs.shape[2:]))
        y = y.view(inputs.shape[0], inputs.shape[1], *y.shape[1:])
        if scorer is None:
            evaluations = self._objective.forward_multi(y, t)
        else:
            evaluations = scorer.assess_multi(y, t)
        self._inputs = self.input_processor.report(evaluations)
        return evaluations

    @property
    def inputs(self):
        return self._inputs
    
    @property
    def theta(self):
        return nn_utils.parameters_to_vector(self._net.parameters())

    @property
    def evaluations(self):
        return self._evaluations


class RepeatOptimizer(Optimizer):

    def __init__(self, sub_optimizer: TorchOptimizer):
        super().__init__()
        self._sub_optimizer = sub_optimizer
        self._evaluations = None
    
    def reset_inputs(self, inputs):
        self._sub_optimizer.reset_inputs(inputs)

    def reset_theta(self, theta):
        self._sub_optimizer.reset_theta(theta)

    @abstractmethod
    def update_theta(self, t, inputs=None, theta=None, y=None):
        pass

    @abstractmethod
    def update_inputs(self, t, inputs=None, theta=None, y=None):
        pass

    @property
    def inputs(self):
        return self._sub_optimizer.inputs
    
    @property
    def theta(self):
        return self._sub_optimizer.theta

    @property
    def evaluations(self):
        return self._evaluations


class NRepeatOptimizer(RepeatOptimizer):

    def __init__(self, sub_optimizer: TorchOptimizer, n: int):
        super().__init__(sub_optimizer)
        self._n = n

    def update_theta(self, t, inputs=None, theta=None, y=None, scorer: Scorer=None):
        evaluations = []
        for _ in range(self._n):
            self._sub_optimizer.update_theta(t, inputs, theta, y, scorer=scorer)
            evaluations.append(self._sub_optimizer.evaluations)
        self._evaluations = evaluations

    def update_inputs(self, t, inputs=None, theta=None, y=None, scorer: Scorer=None):
        evaluations = []
        for _ in range(self._n):
            self._sub_optimizer.update_inputs(t, inputs, theta, y, scorer=scorer)
            evaluations.append(self._sub_optimizer.evaluations)
        self._evaluations = evaluations


class SklearnOptimizer(Optimizer):

    def __init__(self, machine: SklearnModule, input_optimizer: TorchOptimizer, partial_fit: bool=False):
        super().__init__()
        self._machine = machine
        self._input_optimizer = input_optimizer
        self._partial_fit = partial_fit
        self._evaluations = None
        self._fitted = False
        self._inputs = None
        self._theta = None
    
    def reset_inputs(self, inputs):
        self._input_optimizer.reset_inputs(inputs)
        self._inputs = inputs

    def reset_theta(self, theta):
        pass

    def update_theta(self, t, inputs=None, theta=None, y=None, scorer: Scorer=None):

        if inputs is not None:
            self.reset_inputs(inputs)
        
        score = self._machine.score(self._inputs, t)
        if score is not None:
            self._evaluations = [th.tensor(score)]
        else:
            self._evaluations = []

        if self._partial_fit:
            self._machine.partial_fit(inputs, t)
        else:
            self._machine.fit(inputs, t)

    def update_inputs(self, t, inputs=None, theta=None, y=None, scorer: Scorer=None):
        self._input_optimizer.update_inputs(t, inputs, theta, y, scorer=scorer)
        self._evaluations = self._input_optimizer.evaluations

    @property
    def inputs(self):
        return self._input_optimizer.inputs
    
    @property
    def theta(self):
        return self._machine

    @property
    def evaluations(self):
        return self._evaluations


class THOptimBuilder(ABC):

    @abstractmethod
    def __call__(self, net, loss) -> Optimizer:
        pass


class SingleOptimBuilder(object):

    def __init__(self):
        
        self._optim = None
        self.grad(lr=1e-2)
        self.n_repetitions = 1

    def grad(self, optim_cls=None, **kwargs):

        if optim_cls is None:
            optim_cls = th.optim.Adam

        self._optim = partial (
            GradOptimizer,
            optim=partial(optim_cls, **kwargs)
        )
        return self
    
    def step_hill_climber(self, momentum=0.5, mean: float=-1e-2, std: float=1, update_after: int=400):

        processor = StepHillClimberProcessor(k=16, momentum=momentum, s_mean=mean, s_std=std, update_after=update_after)
        self._optim = partial(HillClimberOptimizer, processor=processor)
        return self
    
    def repeat(self, n_repetitions: int):

        if n_repetitions <= 0:
            raise RuntimeError("Repetitions must be greater than or equal to 1")
        self.n_repetitions = n_repetitions
        return self

    def __call__(self, net, loss) -> Optimizer:
        if self.n_repetitions > 1:
            return NRepeatOptimizer(self._optim(net, loss), self.n_repetitions)
        else:
            return self._optim(net, loss)


class THXPBuilder(object):

    def __init__(self, x_builder: THOptimBuilder, p_builder: THOptimBuilder):
        
        self._x_builder = x_builder
        self._p_builder = p_builder

    def __call__(self, net, loss):

        x_updater = self._x_builder(net, loss)
        p_updater = self._p_builder(net, loss)
        return XPOptimizer(x_updater, p_updater)


class SKOptimBuilder(object):

    def __init__(self, x_builder: THOptimBuilder):

        self._x_builder = x_builder
        self._partial_fit = False

    def partial_fit(self, to_partial_fit: bool=True):
        self._partial_fit = to_partial_fit

    def __call__(self, module: SklearnModule, loss) -> Optimizer:
        
        return SklearnOptimizer(
            module, self._x_builder(module, loss), self._partial_fit)


class BlackboxOptimBuilder:

    def __init__(self):
        
        self._optim = None
        self.step_hill_climber()
        self.n_repetitions = 1

    def step_hill_climber(self, momentum=0.5, mean: float=-1e-2, std: float=1, update_after: int=400):

        processor = StepHillClimberProcessor(k=16, momentum=momentum, s_mean=mean, s_std=std, update_after=update_after)
        self._optim = partial(HillClimberOptimizer , processor=processor)
        return self
    
    def repeat(self, n_repetitions: int):

        if n_repetitions <= 0:
            raise RuntimeError("Repetitions must be greater than or equal to 1")
        self.n_repetitions = n_repetitions
        return self

    def __call__(self, net: Blackbox, loss) -> Optimizer :
        if self.n_repetitions > 1:
            return NRepeatOptimizer (self._optim(net, loss), self.n_repetitions)
        else:
            return self._optim(net, loss)



# theta_optim.reset(<theta>)
# input_optim.step()
# input_optim.reset(<theta>)
# 
# self._updater.update_theta()
# self._updater.update_inputs()

