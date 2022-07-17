from functools import partial
import sklearn.linear_model
import sklearn.multioutput
import torch.nn as nn

from . import optim_builders
from .machinery import SklearnMachine

from .base import SklearnModule, TorchScore

from .machinery import TorchNN
from . import modules
import torch as th
from . import optimizers
from . import modules


class TestGradThetaOptim:

    def test_evaluations_is_one(self):
        linear = nn.Linear(2, 2)
        objective = TorchNN(linear, nn.MSELoss)
        optim = optimizers.NRepeatThetaOptim(optimizers.GradThetaOptim(
            linear, False
        ), 2)
        assessment = optim.step(th.randn(1, 2), th.randn(1, 2), objective)
        assert assessment.maximize is False
        
    def test_theta_has_changed(self):
        linear = nn.Linear(2, 2)
        objective = TorchNN(linear, nn.MSELoss)
        optim = optimizers.NRepeatThetaOptim(optimizers.GradThetaOptim(
            linear
        ), 2)
        theta = th.clone(optim.theta)
        optim.step(th.randn(1, 2), th.randn(1, 2), objective)
        assert (theta != optim.theta).any()


class TestGradInputOptim:

    def test_assessment_maximize_is_false(self):
        linear = nn.Linear(2, 2)
        objective = TorchNN(linear, nn.MSELoss)
        optim = optimizers.GradInputOptim(
            linear
        )
        x_prime, assessment = optim.step(th.randn(1, 2), th.randn(1, 2), objective)
        assert assessment.maximize is False
        
    def test_theta_has_changed(self):
        linear = nn.Linear(2, 2)
        objective = TorchNN(linear, nn.MSELoss)
        optim = optimizers.GradInputOptim(
            linear
        )
        x1 = th.randn(1, 2)
        x2, _ = optim.step(x1, th.randn(1, 2), objective)
        assert (x1 != x2).any()
        
    def test_theta_has_changed_with_y(self):
        linear = nn.Linear(2, 2)
        objective = TorchNN(linear, nn.MSELoss)
        optim = optimizers.GradInputOptim(
            linear
        )
        x1 = th.randn(1, 2, requires_grad=True)
        x1.retain_grad()
        
        y, result = objective.forward(x1, full_output=True)
        x2, _ = optim.step(x1, th.randn(1, 2), objective, result)
        assert (x1 != x2).any()


class TestNRepeatInputOptim:

    def test_evaluations_is_one(self):
        linear = nn.Linear(2, 2)
        objective = TorchNN(linear, nn.MSELoss)
        optim = optimizers.NRepeatInputOptim(optimizers.GradInputOptim(
            linear
        ), 3)
        _, assessment = optim.step(th.randn(1, 2), th.randn(1, 2), objective)
        assert assessment.maximize is False
        
    def test_theta_has_changed(self):
        linear = nn.Linear(2, 2)
        objective = TorchNN(linear, nn.MSELoss)
        optim =  optimizers.NRepeatInputOptim(optimizers.GradInputOptim(
            linear
        ), 3)
        x1 = th.randn(1, 2)
        x2, _ = optim.step(x1, th.randn(1, 2), objective)
        assert (x1 != x2).any()


class TestSklearnThetaOptim:

    def test_evaluations_with_fit_is_one(self):
        module = modules.SklearnWrapper(sklearn.linear_model.LinearRegression(), 2, 2)
        optim = optimizers.SklearnThetaOptim(
            module, False
        )
        objective = SklearnMachine(
            module, TorchScore(nn.MSELoss, maximize=False),
            optim_builders.SklearnOptimBuilder(), 
            optim_builders.InputOptimBuilder().step_hill_climber(),
            partial=False, 
        )
        assessment = optim.step(th.randn(2, 2), th.randn(2), objective)
        assert assessment.maximize is False

    def test_evaluations_with_partial_fit_is_one(self):
        module = modules.SklearnWrapper(sklearn.linear_model.LinearRegression(), 2, 2)
        optim = optimizers.SklearnThetaOptim(
            module, False
        )
        objective = SklearnMachine(
            module, TorchScore(nn.MSELoss, maximize=False),
            optim_builders.SklearnOptimBuilder(), 
            optim_builders.InputOptimBuilder().step_hill_climber(),
            partial=True, 
        )
        assessment = optim.step(th.randn(2, 2), th.randn(2), objective)
        assert assessment.maximize is False
