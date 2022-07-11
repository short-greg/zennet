from abc import abstractmethod, abstractproperty
import typing
from .base import Objective,Machine, Reduction
import torch
import torch.nn as nn
from torch.nn import functional as nn_func


# class SequenceObjective(Objective):
#     """Outputs the score for a Sequence
#     """
#     def __init__(self, machines: typing.List[Machine], t: torch.Tensor, outer: Objective=None):

#         if len(machines) == 0:
#             raise RuntimeError("The number of machines must be greater than 0.")
#         self._machines = machines
#         self._outer = outer
#         self._t = t
    
#     def prepend(self, objective: Objective):
#         pass

#     def assess(self, x):
#         for machine in self._machines:
#             x = machine.forward(x)
#         if self._outer:
#             return self._outer.assess(x)
#         return self._machines[-1].assess(x, self._t)

#     @property
#     def maximize(self):
#         return self._machines[-1].maximize


# class MeanReduction(Reduction):
    
#     def __call__(self, grade):
        
#         return grade.reshape(grade.size(0), -1).mean(dim=1)


# class LossObjective(Objective):

#     def __init__(self, th_loss, reduction: Reduction):
#         super().__init__(reduction)
#         loss = th_loss(reduction="none")
#         loss.reduction = 'none'
#         self.loss = loss

#     @property
#     def maximize(self) -> bool:
#         return False

#     def eval(self, y, t):
#         return self.loss(y, t)


# class AdaptiveObjective(Objective):

#     def __init__(self, reduction: str='mean'):
#         pass

#     @abstractmethod
#     def update(self, x: torch.Tensor, t: torch.Tensor):
#         pass
    
#     @abstractproperty
#     def evaluations(self):
#         pass

#     @property
#     def maximize(self) -> bool:
#         return False

#     def eval(self, x, t):
#         return self.loss(x, t)


# # TODO: I think i want to use a machine
# # rather than objective for this... The machine will be limiting
# # I am on the right track but I don't think this will work well

# # t = adapt(x, t)
# # loss(x, t)
# # calculate gradients
# # how to deal with regularization

# class AdaptiveRegression(AdaptiveObjective):

#     def __init__(self, regression_loss: nn.Module, reduction: str='mean', w: float=0.1):
#         super().__init__(reduction)
#         self._regression_loss = regression_loss
#         assert 0 < w < 1, f"Argument w must be between value of 0 and 1 not {w}"
#         self._w = w
#         self._w_comp = 1 - self._w

#     def update(self, x: torch.Tensor, t: torch.Tensor):
#         # output the true evaluations
#         # just have it be like a regular 'objective'
#         # no 'update' like this. Then if you decide to use gradient or something you can
#         # and combine it with other losses
#         # include "preprocess" method for targets
#         sz = x.size()
#         x = x.view(x.size(0), -1).detach().requires_grad()
#         x.retains_grad()
#         t = self._w_comp * x + self._w * t.view(t.size(0), -1)
#         loss: torch.Tensor = self._loss(x, t)
#         self._evaluations = self.reduction(loss)
#         loss.backward()
#         return (x - x.grad).view(*sz)

#     @abstractproperty
#     def evaluations(self):
#         return self._evaluations


# class AdaptiveClassificaton(AdaptiveObjective):

#     def __init__(self, class_loss: nn.Module, reduction: str='mean', w: float=0.1):
#         super().__init__(reduction)
#         self._loss = class_loss
#         assert 0 < w < 1, f"Argument w must be between value of 0 and 1 not {w}"
#         self._w = w
#         self._w_comp = 1 - self._w

#     def update(self, x: torch.Tensor, t: torch.Tensor):
#         sz = x.size()
#         x = x.view(x.size(0), -1).detach().requires_grad()
#         x.retains_grad()
#         t: torch.FloatTensor = nn_func.one_hot(t.long()).float().view(t.size(0), -1)
#         t = self._w_comp * x + self._w * t
#         t = t / t.sum(dim=1, keepdim=True)
#         loss: torch.Tensor = self._loss(x, t)
#         self._evaluations = self.reduction(loss)
#         loss.backward()
#         return (x - x.grad).view(*sz)

#     @abstractproperty
#     def evaluations(self):
#         return self._evaluations



# # class Skloss(object):

# #     def __init__(self, sklearn_module: SklearnModule, objective: Objective):
# #         super().__init__()
# #         self._machine = sklearn_module
# #         self._objective = objective

# #     @property
# #     def maximize(self) -> bool:
# #         return False

# #     def forward_multi(self, x, t):
# #         y = self._machine(x)
# #         return self._objective.forward_multi(y, t)
    
# #     def eval(self, x, t):
# #         return self._machine.score(x, t)



