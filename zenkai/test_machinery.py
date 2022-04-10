from re import X
from . import machinery as mac
import torch.nn as nn
import torch
from functools import partial
import numpy as np


class TestTorchNN:

    def _build_layer_and_machine(self):
        layer = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid()
        )
        optim = partial(torch.optim.Adam, lr=1e-2)
        loss = nn.MSELoss()
        machine = mac.TorchNN(layer, optim, loss)
        return machine, layer, loss

    def test_torch_nn_with_linear_plus_sigmoid(self):
        machine, layer, loss = self._build_layer_and_machine()
        x = torch.zeros(3, 2)
        result = machine.forward(x)
        target = layer.forward(x)
        assert (result == target).all()

    def test_torch_nn_assess_with_linear_plus_sigmoid(self):
        machine, layer, loss = self._build_layer_and_machine()
        torch.manual_seed(1)
        x = torch.rand(3, 2)
        t = torch.rand(3, 2)
        result = loss(layer(x), t)
        assert (machine.assess(x, t) == result).all()

    def test_torch_nn_backward_with_linear_plus_sigmoid(self):
        machine, layer, loss = self._build_layer_and_machine()
        torch.manual_seed(1)
        x = torch.rand(3, 2)
        x.requires_grad_()
        x.retain_grad()
        t = torch.rand(3, 2)
        x_t = machine.update_x(x, t)
        assert (x_t.shape == x.shape)


class TestProcessed:

    def _build_layer_and_machine(self, processors: list):
        layer = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid()
        )
        optim = partial(torch.optim.Adam, lr=1e-2)
        loss = nn.MSELoss()
        machine = mac.TorchNN(layer, optim, loss)
        processed = mac.Processed(processors, machine)
        return processed, layer, loss

    def test_processed_with_no_processor(self):
        machine, layer, loss = self._build_layer_and_machine([])
        x = torch.zeros(3, 2)
        result = machine.forward(x)
        target = layer.forward(x)
        assert (result == target).all()

    def test_processed_with_np_to_th_processor(self):
        torch.manual_seed(1)
        machine, layer, loss = self._build_layer_and_machine([mac.NP2TH()])
        x = np.zeros((3, 2))
        result = machine.forward(x)
        target = layer.forward(torch.tensor(x, dtype=torch.float32))
        assert (result == target).all()

    def test_torch_nn_assess_with_linear_plus_sigmoid(self):
        torch.manual_seed(1)
        machine, layer, loss = self._build_layer_and_machine([mac.NP2TH()])
        x = np.zeros((3, 2))
        t = torch.rand(3, 2)
        target = loss(layer.forward(torch.tensor(x, dtype=torch.float32)), t)
        result = machine.assess(x, t)
        assert (result == target).all()

    def test_torch_nn_backward_with_linear_plus_sigmoid(self):
        machine, layer, loss = self._build_layer_and_machine([mac.NP2TH()])
        torch.manual_seed(1)
        x = np.zeros((3, 2))
        t = torch.rand(3, 2)
        x_t = machine.update_x(x, t)
        assert (x_t.shape == x.shape)

    def test_torch_nn_assess_with_linear_plus_sigmoid_with_no_processor(self):
        torch.manual_seed(1)
        machine, layer, loss = self._build_layer_and_machine([])
        x = torch.zeros(3, 2)
        t = torch.rand(3, 2)
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
        optim = partial(torch.optim.Adam, lr=1e-2)
        optim2 = partial(torch.optim.Adam, lr=1e-2)
        loss = nn.MSELoss()
        loss2 = nn.MSELoss()
        machine = mac.TorchNN(layer1, optim, loss)
        machine2 = mac.TorchNN(layer2, optim2, loss2)
        sequence = mac.Sequence([machine, machine2])
        return sequence, layer1, layer2, loss, loss2

    def test_sequence_forward(self):
        machine, layer1, layer2, loss1, loss2 = self._build_layer_and_machine()
        x = torch.zeros(3, 2)
        result = machine.forward(x)
        target = layer2(layer1.forward(x))
        assert (result == target).all()

    def test_torch_nn_assess_with_linear_plus_sigmoid(self):
        machine, layer1, layer2, loss1, loss2 = self._build_layer_and_machine()
        torch.manual_seed(1)
        x = torch.zeros((3, 2))
        t = torch.rand(3, 2)
        result = loss2(layer2(layer1(x)), t)
        assert (machine.assess(x, t) == result).all()

    def test_torch_nn_backward_with_linear_plus_sigmoid(self):
        machine, layer1, layer2, loss1, loss2 = self._build_layer_and_machine()
        torch.manual_seed(1)
        x = torch.zeros((3, 2))
        t = torch.rand(3, 2)
        x_t = machine.update_x(x, t)
        assert (x_t.shape == x.shape)
