import torch
from .machinery import SklearnMachine
from . import modules
import sklearn.linear_model
import sklearn.multioutput
import numpy as np


class TestSklearnModule:
    
    def test_sklearn_module_with_1d_outputs_correct_size(self):

        machine = sklearn.linear_model.LinearRegression()
        machine.fit(np.random.rand(3, 1), np.random.rand(3, 1))
        module = modules.SklearnWrapper(machine, 1, 1)
        assert module.forward(torch.rand(3, 1)).size() == torch.Size([3, 1])

    def test_sklearn_module_with_2d_outputs_correct_size(self):

        machine = sklearn.linear_model.LinearRegression()
        machine = sklearn.multioutput.MultiOutputRegressor(machine)
        machine.fit(np.random.rand(3, 3), np.random.rand(3, 3))
        module = modules.SklearnWrapper(machine, 3, 3)
        assert module.forward(torch.rand(3, 3)).size() == torch.Size([3, 3])


class TestLambda:
    
    def test_blackbox_with_one_argument_ouputs_correctly(self):
        def f(x):
            return x + 2
        
        module = modules.Lambda(f)
        x = torch.rand(2)
        assert (module.forward(x) == (x + 2)).all()

    def test_blackbox_with_one_argument_ouputs_correctly(self):
        def f(x1, x2):
            return x1 + x2
        
        module = modules.Lambda(f)
        x = torch.rand(2)
        x2 = torch.rand(2)
        assert (module.forward(x, x2) == (x + x2)).all()


class TestPerceptron:

    def test_that_perceptron_outputs_binary_values(self):
        perceptron = modules.Perceptron(2, 3)
        y = perceptron.forward(torch.rand(2))
        assert ((y == 0) | (y == 1)).all()

    def test_that_score_outputs_mean_of_correct(self):
        perceptron = modules.Perceptron(2, 3)
        x = torch.rand(2)
        y = perceptron.forward(x)
        t = (torch.rand(3) > 0.5).float()
        expected = (y == t).float().mean()
        outcome = perceptron.score(x, t)
        
        assert (expected == outcome).all()
