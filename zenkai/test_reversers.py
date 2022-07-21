
from .reversers import GeneralLossReverse, MSELossReverse
import torch
import math


class TestMSELossReverse:

    def test_mse_loss_reverse_with_distance_of_one(self):
        reverse = MSELossReverse()
        x = torch.tensor([[1.0], [2.0]])
        t = torch.tensor([[2.0], [1.0]])
        t_prime = reverse.reverse(
            x, t, lr=0.5
        )
        assert t_prime[0, 0] == 1.0 + math.sqrt(0.5)
        assert t_prime[1, 0] == 2.0 - math.sqrt(0.5)
    
    def test_mse_loss_reverse_with_distance_of_half(self):
        reverse = MSELossReverse()
        x = torch.tensor([[1.0], [2.0]])
        t = torch.tensor([[1.5], [1.5]])
        t_prime = reverse.reverse(
            x, t, lr=0.5
        )
        print(t_prime)
        assert t_prime[0, 0] == 1.0 + math.sqrt(0.5 * 0.5 ** 2)
        assert t_prime[1, 0] == 2.0 - math.sqrt(0.5 * 0.5 ** 2)


class TestGeneralLossReverse:

    def test_general_loss_produces_value_inside_bounds(self):
        reverse = GeneralLossReverse(torch.nn.MSELoss)
        x = torch.tensor([[1.0], [2.0]])
        t = torch.tensor([[2.0], [1.0]])
        t_prime = reverse.reverse(
            x, t, lr=0.5
        )
        assert (x[0, 0] <= t_prime[0, 0] <= t[0,0]).all()
        assert (t[1, 0] <= t_prime[1, 0] <= x[1, 0]).all()