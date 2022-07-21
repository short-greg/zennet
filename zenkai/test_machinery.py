import typing

from . import builders as optim_builders

from . import base
from . import machinery
import torch.nn as nn
import torch as torch
from functools import partial
import numpy as np
from . import reversers


class TestTorchNN:

    def _build_layer_and_machine(self) -> typing.Tuple[machinery.TorchNN, nn.Module, machinery.Objective]:
        layer = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid()
        )

        machine = machinery.TorchNN(layer, nn.MSELoss)
        return machine, layer

    def test_backward_update_does_not_update_when_fixed(self):
        torch.manual_seed(1)
        machine, layer = self._build_layer_and_machine()
        machine.update_theta(False)
        before = next(layer.parameters()).clone()
        x = torch.rand(3, 2)
        t = torch.rand(3, 2)
        machine.backward_update(x, t)
        after = next(layer.parameters()).clone()
        assert (before == after).all()

    def test_backward_updates_when_not_fixed(self):
        torch.manual_seed(1)
        machine, layer = self._build_layer_and_machine()
        before = next(layer.parameters()).clone()
        x = torch.rand(3, 2)
        t = torch.rand(3, 2)
        machine.backward_update(x, t)
        after = next(layer.parameters()).clone()
        assert (before != after).any()

    def test_forward_update_updates_without_scorer(self):
        torch.manual_seed(1)
        machine, layer = self._build_layer_and_machine()
        before = next(layer.parameters()).clone()
        x = torch.rand(3, 2)
        t = torch.rand(3, 2)
        machine.backward_update(x, t)
        after = next(layer.parameters()).clone()
        assert (before != after).any()
    
#     # TODO: Fix the torch scorer

    # def test_forward_update_updates_with_scorer(self):

    #     outer = base.MachineObjective(nn.MSELoss)
    #     torch.manual_seed(1)
    #     machine, layer = self._build_layer_and_machine()
    #     before = next(layer.parameters()).clone()
    #     x = torch.rand(3, 2)
    #     t = torch.rand(3, 2)
    #     machine.forward_update(x, t, outer=base.MachineObjective(outer))
    #     after = next(layer.parameters()).clone()
    #     assert (before != after).any()

    def test_forward_update_does_not_update_when_fixed(self):
        torch.manual_seed(1)
        machine, layer = self._build_layer_and_machine()
        machine.update_theta(False)
        before = next(layer.parameters()).clone()
        x = torch.rand(3, 2)
        t = torch.rand(3, 2)
        machine.forward_update(x, t)
        after = next(layer.parameters()).clone()
        assert (before == after).all()


class TestSequence:

    def _build_layer_and_machine(self):
        layer1 = nn.Sequential(
            nn.Linear(2, 3),
            nn.Sigmoid()
        )
        layer2 = nn.Sequential(
            nn.Linear(3, 2),
            nn.Sigmoid()
        )

        optim_theta = optim_builders.ThetaOptimBuilderStd().step_hill_climber()
        optim_input = optim_builders.InputOptimBuilderStd().step_hill_climber()
        machine = machinery.TorchNN(layer1, nn.MSELoss, optim_theta, optim_input)
        machine2 = machinery.TorchNN(layer2, nn.MSELoss, optim_theta, optim_input)
        sequence = machinery.Sequence([machine, machine2])
        return sequence, layer1, layer2

    def test_sequence_forward(self):
        machine, layer1, layer2 = self._build_layer_and_machine()
        x = torch.zeros(3, 2)
        result = machine.forward(x)
        target = layer2(layer1.forward(x))
        assert (result == target).all()

    def test_torch_nn_assess_with_linear_plus_sigmoid(self):
        machine, layer1, layer2 = self._build_layer_and_machine()
        torch.manual_seed(1)
        x = torch.zeros(3, 2)
        t = torch.rand(3, 2)
        # the mean of the batch mean
        target = nn.MSELoss(reduction='none')(layer2(layer1(x)), t).mean(dim=1).mean()
        reg, _ = machine.assess(x, t).mean().item()
        assert (reg == target).all()

    def test_torch_nn_backward_with_linear_plus_sigmoid(self):
        machine, _, _ = self._build_layer_and_machine()
        torch.manual_seed(1)
        x = torch.zeros((3, 2), requires_grad=True)
        x.retain_grad()
        t = torch.rand(3, 2)
        x_t = machine.backward_update(x, t)
        assert (x_t.shape == x.shape)

    def test_sequence_forward_update_updates_with_no_scorer(self):

        outer = machinery.TorchNN(
            nn.Linear(2, 4), nn.MSELoss
        )
        torch.manual_seed(6)

        machine, layer1, layer2 = self._build_layer_and_machine()
        torch.manual_seed(1)
        x = torch.zeros((3, 2), requires_grad=True)
        x.retain_grad()
        t = torch.rand(3, 4)

        before = next(layer1.parameters()).clone()
        before2 = next(layer2.parameters()).clone()
        machine.forward_update(x, t, outer)
        after = next(layer1.parameters()).clone()
        after2 = next(layer2.parameters()).clone()
        
        # TODO: Do a better test.. Relies on manual seed
        # so if something changes in the code.. this will break.
        assert (before != after).any()


class TestTargetTransform:

    def test_transform_assess_is_same_as_mse(self):
        x = torch.zeros(3, 2)
        t = torch.zeros(3, 2)
        loss_reverse = reversers.MSELossReverse()
        transform = machinery.ScoreReverseTransform(loss_reverse, 0.5)
        result = transform.assess(x, t).mean().item()
        assert result, result == nn.MSELoss(reduction='mean')(x, t).item()

    def test_transform_backward_update_returns_nothing_when_inputs_is_false(self):
        x = torch.zeros(3, 2)
        t = torch.zeros(3, 2)
        loss_reverse = reversers.MSELossReverse()
        transform = machinery.ScoreReverseTransform(loss_reverse, 0.5)
        result = transform.backward_update(x, t, update_inputs=False)
        assert result is None

    def test_transform_backward_update_returns_updated_x_when_inputs_is_true(self):
        x = torch.zeros(3, 2)
        t = torch.zeros(3, 2)
        loss_reverse = reversers.MSELossReverse()
        transform = machinery.ScoreReverseTransform(loss_reverse, 0.5)
        x_prime = transform.backward_update(x, t, update_inputs=True)
        assert x_prime.size() == x.size()

    def test_transform_forward_outputs_x(self):
        x = torch.zeros(3, 2)
        t = torch.zeros(3, 2)
        loss_reverse = reversers.MSELossReverse()
        transform = machinery.ScoreReverseTransform(loss_reverse, 0.5)
        y = transform.forward(x)
        assert (y == x).all()

    def test_transform_forward_update_outputs_x(self):
        x = torch.zeros(3, 2)
        t = torch.zeros(3, 2)
        loss_reverse = reversers.MSELossReverse()
        transform = machinery.ScoreReverseTransform(loss_reverse, 0.5)
        y = transform.forward_update(x, t)
        assert (y == x).all()
