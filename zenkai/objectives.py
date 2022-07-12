from abc import abstractmethod, abstractproperty
import typing
from .base import BatchAssessment, MachineObjective, Objective, Machine, ParameterizedMachine, Regularize, ScalarAssessment, Score, TorchScore
import torch
import torch.nn as nn


class InputRegObjective(MachineObjective):
    
    def __init__(self, machine: Machine, init_x: torch.Tensor, score: TorchScore):

        super().__init__(machine)
        self.score = score
        self.init_x = init_x

    def assess(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, 
        batch_assess: bool=True
    ) -> typing.Union[ScalarAssessment, BatchAssessment]:
        assessment = super().assess(x, t, y, batch_assess)
        return self.score(x, self.init_x) + assessment

    def extend(
        self, x: torch.Tensor, t: torch.Tensor, 
        y: torch.Tensor=None, batch_assess: bool=True
    ) -> (
        typing.Tuple[torch.Tensor, typing.Union[ScalarAssessment, BatchAssessment]]
    ):
        y, assessment = super().assess(x, t, y, batch_assess)
        return y, self.score(x, self.init_x) + assessment


class ThetaRegObjective(MachineObjective):

    def __init__(self, machine: ParameterizedMachine, regularizer: Regularize):

        # super().__init__(machine)
        self._machine = machine
        self.regularizer = regularizer
    
    def assess(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor=None, 
        batch_assess: bool=True
    ) -> typing.Union[ScalarAssessment, BatchAssessment]:
        assessment = super().assess(x, t, y, batch_assess)
        return self.regularizer(self._machine.theta) + assessment

    def extend(
        self, x: torch.Tensor, t: torch.Tensor, 
        y: torch.Tensor=None, batch_assess: bool=True
    ) -> (
        typing.Tuple[torch.Tensor, typing.Union[ScalarAssessment, BatchAssessment]]
    ):
        y, assessment = super().assess(x, t, y, batch_assess)
        return y, self.regularizer(self._machine.theta) + assessment
