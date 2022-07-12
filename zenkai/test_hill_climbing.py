
# from . import modules
# from . import hill_climbing
# import torch.nn as nn
# import torch as th


# class TestHillClimbThetaOptim:

#     def test_evaluations_is_one(self):
#         linear = nn.Linear(2, 2)
#         optim = hill_climbing.HillClimbThetaOptim(
#             linear, modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
#         )
#         optim.step(th.randn(1, 2), th.randn(1, 2))
#         assert len(optim.evaluations) == 2
        
#     def test_theta_has_changed(self):
#         th.manual_seed(1)
#         linear = nn.Linear(2, 2)
#         optim = hill_climbing.HillClimbThetaOptim(
#             linear, modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
#         )
#         theta = th.clone(optim.theta)

#         optim.step(th.randn(1, 2), th.randn(1, 2))
#         assert (theta != optim.theta).any()


# class TestHillClimbInputOptim:

#     def test_evaluations_is_one(self):
#         linear = nn.Linear(2, 2)
#         optim = hill_climbing.HillClimbInputOptim(
#             linear, modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
#         )
#         optim.step(th.randn(1, 2), th.randn(1, 2))
#         assert len(optim.evaluations) == 2
        
#     def test_theta_has_changed(self):
#         th.manual_seed(9)
#         linear = nn.Linear(2, 2)
#         optim = hill_climbing.HillClimbInputOptim(
#             linear, modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
#         )
#         x1 = th.randn(1, 2)
#         x2 = optim.step(x1, th.randn(1, 2))
#         assert (x1 != x2).any()
