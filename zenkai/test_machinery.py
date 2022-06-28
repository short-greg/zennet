from re import X
import typing
from . import machinery as mac
import torch.nn as nn
import torch as th
from functools import partial
import numpy as np
from . import optimizers
from . import modules



class TestTorchNN:

    def _build_layer_and_machine(self) -> typing.Tuple[mac.TorchNN, th.nn.Module, mac.Objective]:
        layer = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid()
        )

        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        machine = mac.TorchNN(layer, objective)

        return machine, layer, objective

    def test_backward_update_does_not_update_when_fixed(self):
        th.manual_seed(1)
        machine, layer, objective = self._build_layer_and_machine()
        machine.fix()
        before = next(layer.parameters()).clone()
        x = th.rand(3, 2)
        t = th.rand(3, 2)
        machine.backward_update(x, t)
        after = next(layer.parameters()).clone()
        assert (before == after).all()

    def test_backward_updates_when_not_fixed(self):
        th.manual_seed(1)
        machine, layer, objective = self._build_layer_and_machine()
        before = next(layer.parameters()).clone()
        x = th.rand(3, 2)
        t = th.rand(3, 2)
        machine.backward_update(x, t)
        after = next(layer.parameters()).clone()
        assert (before != after).any()

    def test_forward_update_updates_without_scorer(self):
        th.manual_seed(1)
        machine, layer, objective = self._build_layer_and_machine()
        before = next(layer.parameters()).clone()
        x = th.rand(3, 2)
        t = th.rand(3, 2)
        machine.backward_update(x, t)
        after = next(layer.parameters()).clone()
        assert (before != after).any()

    def test_forward_update_updates_with_scorer(self):

        class ScorerX(mac.Scorer):
            
            def __init__(self, t: th.Tensor):
                self.t = t

            def assess(self, x: th.Tensor):
                return ((x - self.t) ** 2).mean()

            @property
            def maximize(self):
                return False
        th.manual_seed(1)
        machine, layer, objective = self._build_layer_and_machine()
        before = next(layer.parameters()).clone()
        x = th.rand(3, 2)
        t = th.rand(3, 2)
        machine.forward_update(x, t, scorer=ScorerX(t))
        after = next(layer.parameters()).clone()
        assert (before != after).any()

    def test_forward_update_does_not_update_when_fixed(self):
        th.manual_seed(1)
        machine, layer, objective = self._build_layer_and_machine()
        machine.fix()
        before = next(layer.parameters()).clone()
        x = th.rand(3, 2)
        t = th.rand(3, 2)
        machine.forward_update(x, t)
        after = next(layer.parameters()).clone()
        assert (before == after).all()


class TestProcessed:

    def _build_layer_and_machine(self, processors: list):
        layer = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid()
        )

        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        machine = mac.TorchNN(layer, objective)
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
        result = machine.assess(machine.forward(x), t)
        assert (result == target).all()

    def test_torch_nn_backward_with_linear_plus_sigmoid(self):
        machine, layer, loss = self._build_layer_and_machine([mac.NP2TH()])
        th.manual_seed(1)
        x = np.zeros((3, 2))
        t = th.rand(3, 2)
        x_t = machine.backward_update(x, t, update_theta=False)
        assert (x_t.shape == x.shape)

    def test_torch_nn_forward_with_linear_plus_sigmoid_with_no_processor(self):
        th.manual_seed(1)
        machine, layer, loss = self._build_layer_and_machine([])
        x = th.zeros(3, 2)
        t = th.rand(3, 2)
        target = loss(layer.forward(x), t)
        result = machine.assess(machine.forward(x), t)
        assert (result == target).all()

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

        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        optim_theta = optimizers.ThetaOptimBuilder().step_hill_climber()
        optim_input = optimizers.InputOptimBuilder().step_hill_climber()
        # optim2 = optimization.SingleOptimBuilder().grad()

        machine = mac.TorchNN(layer1, objective, optim_theta, optim_input)
        machine2 = mac.TorchNN(layer2, objective, optim_theta, optim_input)
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
        target = loss2(layer2(layer1(x)), t)
        result = machine.assess(machine.forward(x), t)
        assert (result == target).all()

    def test_torch_nn_backward_with_linear_plus_sigmoid(self):
        machine, layer1, layer2, loss1, loss2 = self._build_layer_and_machine()
        th.manual_seed(1)
        x = th.zeros((3, 2), requires_grad=True)
        x.retain_grad()
        t = th.rand(3, 2)
        x_t = machine.backward_update(x, t, update_theta=False)
        assert (x_t.shape == x.shape)

    def test_sequence_forward_update_updates_with_no_scorer(self):

        class ScorerX(mac.Scorer):
            
            def __init__(self, t: th.Tensor):
                self.t = t

            def assess(self, x: th.Tensor):
                return ((x - self.t) ** 2).mean()

            @property
            def maximize(self):
                return False
        th.manual_seed(6)

        machine, layer1, layer2, loss1, loss2 = self._build_layer_and_machine()
        th.manual_seed(1)
        x = th.zeros((3, 2), requires_grad=True)
        x.retain_grad()
        t = th.rand(3, 2)

        before = next(layer1.parameters()).clone()
        before2 = next(layer2.parameters()).clone()
        machine.forward_update(x, t)
        after = next(layer1.parameters()).clone()
        after2 = next(layer2.parameters()).clone()
        
        # TODO: For some reason i cannot get grad optimizer to work with this
        assert (before2 != after2).any()
        assert (before != after).any()


# class TestEuclidRecorder:

#     def test_records_inputs_new_values(self):

#         recorder = mac.EuclidRecorder()
#         recorder.record_inputs(1, th.rand(2), th.rand(2), th.tensor(0.2))
#         assert recorder.pos == 0

#     def test_records_inputs_new_values_after_adv(self):

#         recorder = mac.EuclidRecorder()
#         recorder.adv()
#         recorder.record_inputs(1, th.rand(2), th.rand(2), th.tensor(0.2))
#         assert recorder.pos == 1

#     def test_get_input_df_after_adv(self):

#         recorder = mac.EuclidRecorder()
#         recorder.record_inputs(1, th.rand(2), th.rand(2), [0.2])
#         assert len(recorder.input_df) == 1

#     def test_records_theta_new_values(self):

#         recorder = mac.EuclidRecorder()
#         recorder.record_theta(1, th.rand(2), th.rand(2), th.tensor(0.2))
#         assert recorder.pos == 0

#     def test_records_theta_new_values_after_adv(self):

#         recorder = mac.EuclidRecorder()
#         recorder.adv()
#         recorder.record_theta(1, th.rand(2), th.rand(2), th.tensor(0.2))
#         assert recorder.pos == 1

#     def test_get_theta_df_(self):

#         recorder = mac.EuclidRecorder()
#         recorder.record_theta(1, th.rand(2), th.rand(2), [0.2])
#         assert len(recorder.theta_df) == 1

#     def test_get_inputs_df_after_add_theta(self):

#         recorder = mac.EuclidRecorder()
#         recorder.record_theta(1, th.rand(2), th.rand(2), [0.2])
#         assert len(recorder.input_df) == 0
