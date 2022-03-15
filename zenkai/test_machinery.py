from re import X
from . import machinery as mac
import torch.nn as nn
import torch
from functools import partial


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

    def test_torch_nn_asseess_with_linear_plus_sigmoid(self):
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
