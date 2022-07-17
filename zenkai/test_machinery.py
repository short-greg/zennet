import typing

from . import optim_builders

from . import base
from . import machinery
import torch.nn as nn
import torch as torch
from functools import partial
import numpy as np


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

        optim_theta = optim_builders.ThetaOptimBuilder().step_hill_climber()
        optim_input = optim_builders.InputOptimBuilder().step_hill_climber()
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
        print(layer2(layer1(x)).size(), t.size())
        target = nn.MSELoss()(layer2(layer1(x)), t)
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
