from functools import partial
import sklearn.linear_model
import sklearn.multioutput
import torch.nn as nn
from . import modules
import torch as th
from . import optim as optimizers
from . import modules


class TestGradThetaOptim:

    def test_evaluations_is_one(self):
        linear = nn.Linear(2, 2)
        optim = optimizers.NRepeatThetaOptim(optimizers.GradThetaOptim(
            linear, modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        ), 2)
        optim.step(th.randn(1, 2), th.randn(1, 2))
        assert len(optim.evaluations) == 2
        
    def test_theta_has_changed(self):
        linear = nn.Linear(2, 2)
        optim = optimizers.NRepeatThetaOptim(optimizers.GradThetaOptim(
            linear, modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        ), 2)
        theta = th.clone(optim.theta)

        optim.step(th.randn(1, 2), th.randn(1, 2))
        assert (theta != optim.theta).any()


class TestGradInputOptim:

    def test_evaluations_is_one(self):
        linear = nn.Linear(2, 2)
        optim = optimizers.GradInputOptim(
            linear, modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        )
        optim.step(th.randn(1, 2), th.randn(1, 2))
        assert len(optim.evaluations) == 1
        
    def test_theta_has_changed(self):
        linear = nn.Linear(2, 2)
        optim = optimizers.GradInputOptim(
            linear, modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        )
        x1 = th.randn(1, 2)
        x2 = optim.step(x1, th.randn(1, 2))
        assert (x1 != x2).any()
        
    def test_theta_has_changed_with_y(self):
        linear = nn.Linear(2, 2)
        optim = optimizers.GradInputOptim(
            linear, modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        )
        x1 = th.randn(1, 2, requires_grad=True)
        x1.retain_grad()
        
        y = linear(x1)
        x2 = optim.step(x1, th.randn(1, 2), y)
        assert (x1 != x2).any()


class TestNRepeatInputOptim:

    def test_evaluations_is_one(self):
        linear = nn.Linear(2, 2)
        optim = optimizers.NRepeatInputOptim(optimizers.GradInputOptim(
            linear, modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        ), 3)
        optim.step(th.randn(1, 2), th.randn(1, 2))
        assert len(optim.evaluations) == 3
        
    def test_theta_has_changed(self):
        linear = nn.Linear(2, 2)
        optim =  optimizers.NRepeatInputOptim(optimizers.GradInputOptim(
            linear, modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        ), 3)
        x1 = th.randn(1, 2)
        x2 = optim.step(x1, th.randn(1, 2))
        assert (x1 != x2).any()


class TestSklearnThetaOptim:

    def test_evaluations_with_fit_is_one(self):
        optim = optimizers.SklearnThetaOptim(
            sklearn.linear_model.LinearRegression(), False
        )
        optim.step(th.randn(2, 2), th.randn(2))
        assert len(optim.evaluations) == 1

    def test_evaluations_with_partial_fit_is_one(self):
        optim = optimizers.SklearnThetaOptim(
            sklearn.linear_model.SGDRegressor(), True
        )
        optim.step(th.randn(3, 2), th.randn(3))
        assert len(optim.evaluations) == 1


class TestHillClimbThetaOptim:

    def test_evaluations_is_one(self):
        linear = nn.Linear(2, 2)
        optim = optimizers.HillClimbThetaOptim(
            linear, modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        )
        optim.step(th.randn(1, 2), th.randn(1, 2))
        assert len(optim.evaluations) == 2
        
    def test_theta_has_changed(self):
        th.manual_seed(1)
        linear = nn.Linear(2, 2)
        optim = optimizers.HillClimbThetaOptim(
            linear, modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        )
        theta = th.clone(optim.theta)

        optim.step(th.randn(1, 2), th.randn(1, 2))
        assert (theta != optim.theta).any()


class TestHillClimbInputOptim:

    def test_evaluations_is_one(self):
        linear = nn.Linear(2, 2)
        optim = optimizers.HillClimbInputOptim(
            linear, modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        )
        optim.step(th.randn(1, 2), th.randn(1, 2))
        assert len(optim.evaluations) == 2
        
    def test_theta_has_changed(self):
        th.manual_seed(9)
        linear = nn.Linear(2, 2)
        optim = optimizers.HillClimbInputOptim(
            linear, modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        )
        x1 = th.randn(1, 2)
        x2 = optim.step(x1, th.randn(1, 2))
        assert (x1 != x2).any()