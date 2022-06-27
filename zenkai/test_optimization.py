from functools import partial
import sklearn.linear_model
import sklearn.multioutput
from . import optimization
import torch.nn as nn
from . import modules
import torch as th



class TestTHOptimBuilder:

    def test_grad_builder_builds_grad(self):

        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        net = nn.Linear(2, 2)
        optimizer = optimization.SingleOptimBuilder().grad()(net, objective)
        assert isinstance(optimizer, optimization.GradOptimizer)
    
    def test_hill_climber_builder_builds_hill_climber(self):
        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        net = nn.Linear(2, 2)
        optimizer = optimization.SingleOptimBuilder().step_hill_climber()(net, objective)
        assert isinstance(optimizer, optimization.HillClimberOptimizer)


class TestHillClimbingOptimizer:

    def test_update_theta_with_one_perturbation(self):
        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = optimization.HillClimberOptimizer(
            net, objective, processor=optimization.StepHillClimberProcessor(k=1)
        )
        optimizer.reset_theta(th.rand(6))
        best = optimizer.theta
        optimizer.update_theta(t=th.rand(2, 2), inputs=th.rand(2, 2))
        assert len(optimizer.evaluations) == 2

    def test_update_theta_with_five_perturbations(self):
        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = optimization.HillClimberOptimizer(
            net, objective, processor=optimization.StepHillClimberProcessor(k=4)
        )
        optimizer.reset_theta(th.rand(6))
        best = optimizer.theta
        optimizer.update_theta(t=th.rand(2, 2), inputs=th.rand(2, 2))
        assert len(optimizer.evaluations) == 5

    def test_update_input_with_one_perturbation(self):
        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = optimization.HillClimberOptimizer(
            net, objective, processor=optimization.StepHillClimberProcessor(k=1)
        )
        optimizer.reset_theta(th.rand(6))
        best = optimizer.theta
        optimizer.update_inputs(t=th.rand(2, 2), inputs=th.rand(2, 2))
        assert len(optimizer.evaluations) == 2
    
    def test_input_is_different_value_after_update(self):
        th.manual_seed(9)
        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = optimization.HillClimberOptimizer(
            net, objective, processor=optimization.StepHillClimberProcessor(k=1)
        )
        optimizer.reset_theta(th.rand(6))
        optimizer.reset_inputs(inputs=th.rand(2, 2))
        best = optimizer.inputs
        optimizer.update_inputs(t=th.rand(2, 2))
        assert (optimizer.inputs != best).any()

    def test_input_is_different_value_after_two_updates(self):
        th.manual_seed(5)
        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = optimization.HillClimberOptimizer(
            net, objective, processor=optimization.StepHillClimberProcessor(k=1)
        )
        optimizer.reset_theta(th.rand(6))
        optimizer.reset_inputs(inputs=th.rand(2, 2))
        best = optimizer.inputs
        optimizer.update_inputs(t=th.rand(2, 2))
        optimizer.update_inputs(t=th.rand(2, 2))
        assert (optimizer.inputs != best).any()


class TestGradOptimizer:

    def test_update_theta(self):
        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = optimization.GradOptimizer(net, objective, partial(th.optim.Adam, lr=1e-2))
        optimizer.reset_theta(th.rand(6))
        best = optimizer.theta
        optimizer.update_theta(t=th.rand(2, 2), inputs=th.rand(2, 2))
        assert (best != optimizer.theta).any()
        
    def test_update_works_after_reset_fixed(self):
        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = optimization.GradOptimizer(net, objective, partial(th.optim.Adam, lr=1e-2))
        optimizer.reset_theta(th.rand(6))
        best = optimizer.theta
        optimizer.reset_inputs(th.rand(2, 2))
        optimizer.update_theta(t=th.rand(2, 2))
        assert (best != optimizer.theta).any()
    
    def test_evaluates_returns_one(self):
        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = optimization.GradOptimizer(net, objective, partial(th.optim.Adam, lr=1e-2))
        optimizer.reset_theta(th.rand(6))
        best = optimizer.theta
        optimizer.update_theta(t=th.rand(2, 2), inputs=th.rand(2, 2))
        assert len(optimizer.evaluations) == 1
        
    def test_update_works_when_y_is_set(self):
        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = optimization.GradOptimizer(net, objective, partial(th.optim.Adam, lr=1e-2))
        optimizer.reset_theta(th.rand(6))
        best = optimizer.theta
        optimizer.reset_inputs(th.rand(2, 2))
        y = net(th.rand(2, 2))
        optimizer.update_theta(t=th.rand(2, 2), y=y)
        assert (best != optimizer.theta).any()

    def test_update_inputs_works(self):
        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = optimization.GradOptimizer(net, objective, optim=partial(th.optim.Adam, lr=1e-2))

        x = th.rand(2, 2).requires_grad_()
        x.retain_grad()
        optimizer.reset_inputs(inputs=x)
        best = optimizer.inputs
        optimizer.update_inputs(t=th.rand(2, 2), theta=th.rand(6))
        assert (best != optimizer.inputs).any()
        
    def test_update_inputs_works_after_reset_fixed(self):
        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = optimization.GradOptimizer(net, objective, optim=partial (th.optim.Adam, lr=1e-2))
        optimizer.reset_theta(theta=th.rand(6))        
        x = th.rand(2, 2).requires_grad_()
        x.retain_grad()
        optimizer.reset_inputs(inputs=x)
        best = optimizer.inputs
        optimizer.update_inputs(t=th.rand(2, 2))
        assert (best != optimizer.inputs).any()

    def test_update_inputs(self):
        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = optimization.GradOptimizer(net, objective, partial(th.optim.Adam, lr=1e-2))
        x = th.rand(2, 2).requires_grad_()
        x.retain_grad()
        optimizer.reset_inputs(x)
        best = optimizer.inputs

        optimizer.update_inputs(t=th.rand(2, 2), theta=th.rand(6))
        assert (best != optimizer.inputs).any()  


class TestNRepeatOptimizer:

    def test_update_theta_with_one_perturbation(self):
        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = optimization.NRepeatOptimizer(optimization.HillClimberOptimizer(
            net, objective, processor=optimization.StepHillClimberProcessor(k=1)
        ), 3)

        optimizer.reset_theta(th.rand(6))
        optimizer.update_theta(t=th.rand(2, 2), inputs=th.rand(2, 2))
        assert len(optimizer.evaluations) == 3

    def test_update_theta_with_five_perturbations(self):
        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = optimization.NRepeatOptimizer(optimization.HillClimberOptimizer(
            net, objective, processor=optimization.StepHillClimberProcessor(k=1)
        ), 3)
        optimizer.reset_theta(th.rand(6))
        optimizer.update_theta(t=th.rand(2, 2), inputs=th.rand(2, 2))
        assert len(optimizer.evaluations) == 3



class TestSKLearnOptimizer:

    def _build(self):
        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())

        regressor = sklearn.linear_model.LinearRegression()
        regressor = sklearn.multioutput.MultiOutputRegressor(regressor)
        net = modules.SklearnModule(regressor)

        input_optimizer = optimization.HillClimberOptimizer(
            net, objective, processor=optimization.StepHillClimberProcessor(k=1)
        )
        return optimization.SklearnOptimizer(regressor, input_optimizer, False)

    def test_sklearn_optimizer_with_fit(self):

        optimizer = self._build()
        optimizer.update_theta(t=th.rand(3, 2), inputs=th.rand(3, 2))
        assert len(optimizer.evaluations) == 0
