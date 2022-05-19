from re import X
from . import machinery as mac
import torch.nn as nn
import torch as th
from functools import partial
import numpy as np


class TestLossObjective:

    def test_loss_objective_maximize_is_false_with_mse(self):

        objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
        assert objective.maximize is False

    def test_loss_objective_evaluation_with_single_result(self):

        objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
        assert objective.forward(th.rand(2, 2), th.rand(2, 2)).size() == th.Size([])

    def test_loss_objective_evaluation_with_multi_result(self):

        objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
        assert objective.forward_multi(th.rand(2, 2, 2), th.rand(2, 2)).size() == th.Size([2])
    

class TestGradOptimizer:

    def test_update_theta(self):
        objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = mac.GradOptimizer(net, objective, partial(th.optim.Adam, lr=1e-2))
        optimizer.reset_theta(th.rand(6))
        best = optimizer.theta
        optimizer.update_theta(t=th.rand(2, 2), inputs=th.rand(2, 2))
        assert (best != optimizer.theta).any()
        
    def test_update_works_after_reset_fixed(self):
        objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = mac.GradOptimizer(net, objective, partial(th.optim.Adam, lr=1e-2))
        optimizer.reset_theta(th.rand(6))
        best = optimizer.theta
        optimizer.reset_inputs(th.rand(2, 2))
        optimizer.update_theta(t=th.rand(2, 2))
        assert (best != optimizer.theta).any()
    
    def test_evaluates_returns_one(self):
        objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = mac.GradOptimizer(net, objective, partial(th.optim.Adam, lr=1e-2))
        optimizer.reset_theta(th.rand(6))
        best = optimizer.theta
        optimizer.update_theta(t=th.rand(2, 2), inputs=th.rand(2, 2))
        assert len(optimizer.evaluations) == 1
        
    def test_update_works_when_y_is_set(self):
        objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = mac.GradOptimizer(net, objective, partial(th.optim.Adam, lr=1e-2))
        optimizer.reset_theta(th.rand(6))
        best = optimizer.theta
        optimizer.reset_inputs(th.rand(2, 2))
        y = net(th.rand(2, 2))
        optimizer.update_theta(t=th.rand(2, 2), y=y)
        assert (best != optimizer.theta).any()

    def test_update_inputs_works(self):
        objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = mac.GradOptimizer(net, objective, optim=partial(th.optim.Adam, lr=1e-2))

        x = th.rand(2, 2).requires_grad_()
        x.retain_grad()
        optimizer.reset_inputs(inputs=x)
        best = optimizer.inputs
        optimizer.update_inputs(t=th.rand(2, 2), theta=th.rand(6))
        assert (best != optimizer.inputs).any()
        
    def test_update_inputs_works_after_reset_fixed(self):
        objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = mac.GradOptimizer(net, objective, optim=partial(th.optim.Adam, lr=1e-2))
        optimizer.reset_theta(theta=th.rand(6))        
        x = th.rand(2, 2).requires_grad_()
        x.retain_grad()
        optimizer.reset_inputs(inputs=x)
        best = optimizer.inputs
        optimizer.update_inputs(t=th.rand(2, 2))
        assert (best != optimizer.inputs).any()

    def test_update_inputs(self):
        objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = mac.GradOptimizer(net, objective, partial(th.optim.Adam, lr=1e-2))
        x = th.rand(2, 2).requires_grad_()
        x.retain_grad()
        optimizer.reset_inputs(x)
        best = optimizer.inputs

        optimizer.update_inputs(t=th.rand(2, 2), theta=th.rand(6))
        assert (best != optimizer.inputs).any()  


class TestHillClimbingOptimizer:

    def test_update_theta_with_one_perturbation(self):
        objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = mac.HillClimberOptimizer(
            net, objective, processor=mac.GaussianHillClimberProcessor(k=1)
        )
        optimizer.reset_theta(th.rand(6))
        best = optimizer.theta
        optimizer.update_theta(t=th.rand(2, 2), inputs=th.rand(2, 2))
        assert len(optimizer.evaluations) == 2

    def test_update_theta_with_five_perturbations(self):
        objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = mac.HillClimberOptimizer(
            net, objective, processor=mac.GaussianHillClimberProcessor(k=4)
        )
        optimizer.reset_theta(th.rand(6))
        best = optimizer.theta
        optimizer.update_theta(t=th.rand(2, 2), inputs=th.rand(2, 2))
        assert len(optimizer.evaluations) == 5

    def test_update_input_with_one_perturbation(self):
        objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = mac.HillClimberOptimizer(
            net, objective, processor=mac.GaussianHillClimberProcessor(k=1)
        )
        optimizer.reset_theta(th.rand(6))
        best = optimizer.theta
        optimizer.update_inputs(t=th.rand(2, 2), inputs=th.rand(2, 2))
        assert len(optimizer.evaluations) == 2
    
    def test_input_is_different_value_after_update(self):
        th.manual_seed(1)
        objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
 
        net = nn.Linear(2, 2)
        optimizer = mac.HillClimberOptimizer(
            net, objective, processor=mac.GaussianHillClimberProcessor(k=1)
        )
        optimizer.reset_theta(th.rand(6))
        optimizer.reset_inputs(inputs=th.rand(2, 2))
        best = optimizer.inputs
        optimizer.update_inputs(t=th.rand(2, 2))
        assert (optimizer.inputs != best).any()
    

class TestTHOptimBuilder:

    def test_grad_builder_builds_grad(self):

        objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
        net = nn.Linear(2, 2)
        optimizer = mac.THOptimBuilder().grad()(net, objective)
        assert isinstance(optimizer, mac.GradOptimizer)
    
    def test_hill_climber_builder_builds_hill_climber(self):
        objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
        net = nn.Linear(2, 2)
        optimizer = mac.THOptimBuilder().hill_climber()(net, objective)
        assert isinstance(optimizer, mac.HillClimberOptimizer)


class TestProcessed:

    def _build_layer_and_machine(self, processors: list):
        layer = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid()
        )

        objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
        optim = mac.THOptimBuilder().hill_climber()
        machine = mac.TorchNN(layer, objective, optim)
        processed = mac.Processed(processors, machine)

        return processed, layer, objective

    def test_processed_with_no_processor(self):
        machine, layer, loss = self._build_layer_and_machine([])
        x = th.zeros(3, 2)
        result = machine.forward(x)
        target = layer.forward(x)
        assert (result == target).all()

    def test_processed_with_np_to_th_processor(self):
        th.manual_seed(1)
        machine, layer, loss = self._build_layer_and_machine([mac.NP2TH()])
        x = np.zeros((3, 2))
        result = machine.forward(x)
        target = layer.forward(th.tensor(x, dtype=th.float32))
        assert (result == target).all()

    def test_torch_nn_assess_with_linear_plus_sigmoid(self):
        th.manual_seed(1)
        machine, layer, loss = self._build_layer_and_machine([mac.NP2TH()])
        x = np.zeros((3, 2))
        t = th.rand(3, 2)
        target = loss(layer.forward(th.tensor(x, dtype=th.float32)), t)
        result = machine.assess(x, t)
        assert (result == target).all()

    def test_torch_nn_backward_with_linear_plus_sigmoid(self):
        machine, layer, loss = self._build_layer_and_machine([mac.NP2TH()])
        th.manual_seed(1)
        x = np.zeros((3, 2))
        t = th.rand(3, 2)
        x_t = machine.update_x(x, t)
        assert (x_t.shape == x.shape)

    def test_torch_nn_assess_with_linear_plus_sigmoid_with_no_processor(self):
        th.manual_seed(1)
        machine, layer, loss = self._build_layer_and_machine([])
        x = th.zeros(3, 2)
        t = th.rand(3, 2)
        target = loss(layer.forward(x), t)
        result = machine.assess(x, t)
        assert (result == target).all()

class TestSequence:

    def _build_layer_and_machine(self):
        layer1 = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid()
        )
        layer2 = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid()
        )

        objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
        optim = mac.THOptimBuilder().grad()
        optim2 = mac.THOptimBuilder().grad()

        machine = mac.TorchNN(layer1, objective, optim)
        machine2 = mac.TorchNN(layer2, objective, optim2)
        sequence = mac.Sequence([machine, machine2])
        return sequence, layer1, layer2, objective, objective

    def test_sequence_forward(self):
        machine, layer1, layer2, loss1, loss2 = self._build_layer_and_machine()
        x = th.zeros(3, 2)
        result = machine.forward(x)
        target = layer2(layer1.forward(x))
        assert (result == target).all()

    def test_torch_nn_assess_with_linear_plus_sigmoid(self):
        machine, layer1, layer2, loss1, loss2 = self._build_layer_and_machine()
        th.manual_seed(1)
        x = th.zeros((3, 2))
        t = th.rand(3, 2)
        result = loss2(layer2(layer1(x)), t)
        assert (machine.assess(x, t) == result).all()

    def test_torch_nn_backward_with_linear_plus_sigmoid(self):
        machine, layer1, layer2, loss1, loss2 = self._build_layer_and_machine()
        th.manual_seed(1)
        x = th.zeros((3, 2), requires_grad=True)
        x.retain_grad()
        t = th.rand(3, 2)
        x_t = machine.update_x(x, t)
        assert (x_t.shape == x.shape)
