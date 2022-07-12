import typing

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
        machine.fix()
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
    
    # TODO: Fix the torch scorer

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
        machine.fix()
        before = next(layer.parameters()).clone()
        x = torch.rand(3, 2)
        t = torch.rand(3, 2)
        machine.forward_update(x, t)
        after = next(layer.parameters()).clone()
        assert (before == after).all()


class TestProcessed:

    def _build_layer_and_machine(self, processors: list):
        layer = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid()
        )

        machine = machinery.TorchNN(layer, nn.MSELoss)
        processed = machinery.Processed(processors, machine)

        return processed, layer

    def test_processed_with_no_processor(self):
        machine, layer = self._build_layer_and_machine([])
        x = torch.zeros(3, 2)
        result = machine.forward(x)
        target = layer.forward(x)
        assert (result == target).all()

    def test_torch_nn_forward_with_linear_plus_sigmoid_with_no_processor(self):
        torch.manual_seed(1)
        machine, layer = self._build_layer_and_machine([])
        x = torch.zeros(3, 2)
        t = torch.rand(3, 2)
        target = nn.MSELoss()(layer.forward(x), t)
        result = machine.assess(x, t, batch_assess=False)
        assert result.unregularized.item(), target.item()

# class TestSequence:

#     def _build_layer_and_machine(self):
#         layer1 = nn.Sequential(
#             nn.Linear(2, 3),
#             nn.Sigmoid()
#         )
#         layer2 = nn.Sequential(
#             nn.Linear(3, 2),
#             nn.Sigmoid()
#         )

#         objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
#         optim_theta = optim_builders.ThetaOptimBuilder().step_hill_climber()
#         optim_input = optim_builders.InputOptimBuilder().step_hill_climber()
#         # optim2 = optimization.SingleOptimBuilder().grad()

#         machine = mac.TorchNN(layer1, objective, optim_theta, optim_input)
#         machine2 = mac.TorchNN(layer2, objective, optim_theta, optim_input)
#         sequence = mac.Sequence([machine, machine2])
#         return sequence, layer1, layer2, objective, objective

#     def test_sequence_forward(self):
#         machine, layer1, layer2, loss1, loss2 = self._build_layer_and_machine()
#         x = th.zeros(3, 2)
#         result = machine.forward(x)
#         target = layer2(layer1.forward(x))
#         assert (result == target).all()

#     def test_torch_nn_assess_with_linear_plus_sigmoid(self):
#         machine, layer1, layer2, loss1, loss2 = self._build_layer_and_machine()
#         th.manual_seed(1)
#         x = th.zeros((3, 2))
#         t = th.rand(3, 2)
#         target = loss2(layer2(layer1(x)), t)
#         result = machine.assess(machine.forward(x), t)
#         assert (result == target).all()

#     def test_torch_nn_backward_with_linear_plus_sigmoid(self):
#         machine, layer1, layer2, loss1, loss2 = self._build_layer_and_machine()
#         th.manual_seed(1)
#         x = th.zeros((3, 2), requires_grad=True)
#         x.retain_grad()
#         t = th.rand(3, 2)
#         x_t = machine.backward_update(x, t, update_theta=False)
#         assert (x_t.shape == x.shape)

#     def test_sequence_forward_update_updates_with_no_scorer(self):

#         class ScorerX(mac.Scorer):
            
#             def __init__(self, t: th.Tensor):
#                 self.t = t

#             def assess(self, x: th.Tensor):
#                 return ((x - self.t) ** 2).mean()

#             @property
#             def maximize(self):
#                 return False
#         th.manual_seed(6)

#         machine, layer1, layer2, loss1, loss2 = self._build_layer_and_machine()
#         th.manual_seed(1)
#         x = th.zeros((3, 2), requires_grad=True)
#         x.retain_grad()
#         t = th.rand(3, 2)

#         before = next(layer1.parameters()).clone()
#         before2 = next(layer2.parameters()).clone()
#         machine.forward_update(x, t)
#         after = next(layer1.parameters()).clone()
#         after2 = next(layer2.parameters()).clone()
        
#         # TODO: For some reason i cannot get grad optimizer to work with this
#         assert (before2 != after2).any()
#         assert (before != after).any()


# # class TestEuclidRecorder:

# #     def test_records_inputs_new_values(self):

# #         recorder = mac.EuclidRecorder()
# #         recorder.record_inputs(1, th.rand(2), th.rand(2), th.tensor(0.2))
# #         assert recorder.pos == 0

# #     def test_records_inputs_new_values_after_adv(self):

# #         recorder = mac.EuclidRecorder()
# #         recorder.adv()
# #         recorder.record_inputs(1, th.rand(2), th.rand(2), th.tensor(0.2))
# #         assert recorder.pos == 1

# #     def test_get_input_df_after_adv(self):

# #         recorder = mac.EuclidRecorder()
# #         recorder.record_inputs(1, th.rand(2), th.rand(2), [0.2])
# #         assert len(recorder.input_df) == 1

# #     def test_records_theta_new_values(self):

# #         recorder = mac.EuclidRecorder()
# #         recorder.record_theta(1, th.rand(2), th.rand(2), th.tensor(0.2))
# #         assert recorder.pos == 0

# #     def test_records_theta_new_values_after_adv(self):

# #         recorder = mac.EuclidRecorder()
# #         recorder.adv()
# #         recorder.record_theta(1, th.rand(2), th.rand(2), th.tensor(0.2))
# #         assert recorder.pos == 1

# #     def test_get_theta_df_(self):

# #         recorder = mac.EuclidRecorder()
# #         recorder.record_theta(1, th.rand(2), th.rand(2), [0.2])
# #         assert len(recorder.theta_df) == 1

# #     def test_get_inputs_df_after_add_theta(self):

# #         recorder = mac.EuclidRecorder()
# #         recorder.record_theta(1, th.rand(2), th.rand(2), [0.2])
# #         assert len(recorder.input_df) == 0
