import typing
from .base import BatchAssessment, Objective, ParameterizedMachine, ScalarAssessment
import torch
from .base import Objective


class ObjectiveDecorator(Objective):

    def __init__(self, objective: Objective):
        
        self._objective = objective

    @property
    def maximize(self) -> bool:
        return self._objective.maximize
    
    def assess_output(
        self, y: torch.Tensor, t: torch.Tensor
    ) -> typing.Union[ScalarAssessment, BatchAssessment]:
        return self._objective.assess_output(y, t)

    def forward(
        self, x: torch.Tensor, full_output: bool=False
    ) -> (
        typing.Tuple[torch.Tensor, typing.Union[ScalarAssessment, BatchAssessment]]
    ):
        return self._objective.forward(x, full_output)


class IRObjective(ObjectiveDecorator):
    
    def __init__(self, objective: Objective, init_x: torch.Tensor, input_reg):

        super().__init__(objective)
        self._init_x = init_x
        self.input_reg = input_reg

    def forward(
        self, x: torch.Tensor, full_output: bool=False
    ) -> (
        typing.Tuple[torch.Tensor, typing.Union[ScalarAssessment, BatchAssessment]]
    ):
        if full_output:
            y, assessment = super().forward(x, True)
            return y, assessment + self._input_reg(self._init_x, x)

        return super().forward(x, False)


class TRObjective(ObjectiveDecorator):
    
    def __init__(self, objective: ParameterizedMachine, theta_reg):
        super().__init__(objective)
        self._objective: ParameterizedMachine = objective
        self._theta_reg = theta_reg

    def forward(
        self, x: torch.Tensor, full_output: bool=False
    ) -> (
        typing.Tuple[torch.Tensor, typing.Union[ScalarAssessment, BatchAssessment]]
    ):
        if full_output:
            y, assessment = super().forward(x, True)
            return y, assessment + self._theta_reg(self._objective.theta)

        return super().forward(x, False)
    