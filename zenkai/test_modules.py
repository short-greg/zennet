import torch
from zenkai.machinery import SklearnMachine
from . import modules
import sklearn.linear_model
import sklearn.multioutput
import numpy as np
import torch.nn as nn


class TestSklearnModule:
    
    def test_sklearn_module_with_1d_outputs_correct_size(self):

        machine = sklearn.linear_model.LinearRegression()
        machine.fit(np.random.rand(3, 1), np.random.rand(3, 1))
        module = modules.SKLearnModule(machine)
        assert module.forward(torch.rand(3, 1)).size() == torch.Size([3, 1])

    def test_sklearn_module_with_2d_outputs_correct_size(self):

        machine = sklearn.linear_model.LinearRegression()
        machine = sklearn.multioutput.MultiOutputRegressor(machine)
        machine.fit(np.random.rand(3, 3), np.random.rand(3, 3))
        module = modules.SKLearnModule(machine)
        assert module.forward(torch.rand(3, 3)).size() == torch.Size([3, 3])


class TestLossObjective:

    def test_loss_objective_maximize_is_false_with_mse(self):

        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        assert objective.maximize is False

    def test_loss_objective_evaluation_with_single_result(self):

        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        assert objective.forward(torch.rand(2, 2), torch.rand(2, 2)).size() == torch.Size([])

    def test_loss_objective_evaluation_with_multi_result(self):

        objective = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
        assert objective.forward_multi(torch.rand(2, 2, 2), torch.rand(2, 2)).size() == torch.Size([2])
    