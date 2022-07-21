import math
from . import base
import torch


class TestAssessment:

    def test_scalar_maximize_is_true(self):
        assessment = base.ScalarAssessment(torch.tensor(1), torch.tensor(0.5), True)
        assert assessment.maximize is True

    def test_scalar_maximize_is_false(self):
        assessment = base.ScalarAssessment(torch.tensor(1), torch.tensor(0.5), False)
        assert assessment.maximize is False

    def test_to_maximize_reverses_sign(self):
        assessment = base.ScalarAssessment(torch.tensor(1), torch.tensor(0.5), False)
        result = assessment.to_maximize(True)
        assert (result.regularized == assessment.regularized * -1).all()

    def test_scalar_add_gives_correct_aswer(self):
        assessment = base.ScalarAssessment(torch.tensor(1), torch.tensor(0.5), False)
        assessment2 = base.ScalarAssessment(torch.tensor(1), torch.tensor(0.5), False)
        added = assessment + assessment2
        assert ((assessment.regularized + assessment2.regularized) == added.regularized).all()

    def test_scalar_multiply_gives_correct_aswer(self):
        assessment = base.ScalarAssessment(torch.tensor(1), torch.tensor(0.5), False)
        multiplied = assessment * 2
        assert ((assessment.regularized * 2) == multiplied.regularized).all()

    def test_scalar_item(self):
        assessment = base.ScalarAssessment(torch.tensor(1), torch.tensor(0.5), False)
        unreg, reg = assessment.item()

        assert reg == assessment.regularized.item()
        assert unreg == assessment.unregularized.item()

    def test_scalar_backward(self):
        x1 = torch.tensor(1.0, requires_grad=True)
        x1.retain_grad()
        x2 = torch.tensor(1.0, requires_grad=True)
        x2.retain_grad()

        assessment = base.ScalarAssessment(x1 + 1, x2 + 1, False)
        assessment.backward()
        assert x2.grad is not None
        assert x1.grad is None

    def test_scalar_backward_unreg(self):
        x1 = torch.tensor(1.0, requires_grad=True)
        x1.retain_grad()
        x2 = torch.tensor(1.0, requires_grad=True)
        x2.retain_grad()

        assessment = base.ScalarAssessment(x1 + 1, x2 + 1, False)
        assessment.backward(True)
        assert x1.grad is not None
        assert x2.grad is None

    def test_scalar_null_assessment(self):
        
        assessment = base.ScalarNullAssessment(torch.float32, 'cpu', True)
        assert assessment.unregularized.item() == 0.0

    def test_batch_assessment(self):
        assessment = base.BatchAssessment(torch.rand(2), torch.rand(2), False)
        assessment2 = base.BatchAssessment(torch.rand(2), torch.rand(2), False)
        added = assessment + assessment2
        assert ((assessment.regularized + assessment2.regularized) == added.regularized).all()

    def test_batch_mul_assessment(self):
        assessment = base.BatchAssessment(torch.rand(2), torch.rand(2), False)
        multiplied = assessment * 2
        assert ((assessment.regularized * 2) == multiplied.regularized).all()

    def test_batch_mean_assessment(self):
        x1 = torch.rand(2)
        x2 = torch.rand(2)
        assessment = base.BatchAssessment(x1, x2, False)
        scalar = assessment.mean()
        assert scalar.regularized.item() == x2.mean().item()

    def test_batch_null_assessment(self):
        assessment = base.BatchNullAssessment(torch.float32, 'cpu')
        scalar = assessment.mean()
        assert scalar.regularized.item() == 0.0

    def test_batch_null_assessment_add(self):
        assessment = base.BatchNullAssessment(torch.float32, 'cpu')
        assessment2 = base.BatchAssessment(torch.rand(2), torch.rand(2), False)
        batch = assessment + assessment2
        assert (batch.unregularized == assessment2.unregularized).all()

    def test_batch_null_assessment_mul(self):
        assessment = base.BatchNullAssessment(torch.float32, 'cpu')
        assessment2 = assessment * 2
        assert (assessment.unregularized == assessment2.unregularized)

    def test_population_assessment_best(self):
        x1 = torch.rand(2)
        x2 = torch.rand(2)
        assessment = base.PopulationAssessment(x1, x2, True)
        target = torch.argmax(x1)
        outcome, value = assessment.best(False)
        assert (target == outcome).all()

    def test_population_assessment_best_for_reg(self):
        x1 = torch.rand(2)
        x2 = torch.rand(2)
        assessment = base.PopulationAssessment(x1, x2, True)
        target = torch.argmax(x2)
        outcome, value = assessment.best(True)
        assert (target == outcome).all()

    def test_population_assessmgent_append(self):
        x1 = torch.rand(2)
        x2 = torch.rand(2)
        assessment = base.PopulationAssessment(x1, x2, True)
        last = base.ScalarAssessment(torch.tensor(1.0), torch.tensor(2.0), True)
        assessment = assessment.append(last)
        assert assessment[-1].regularized == last.regularized


class TestRecording:

    def test_record_inputs(self):

        recording = base.Recording()
        recording.record_inputs('X', {'Regularized': 1.0})
        assert recording.df.loc[0]['Regularized'] == 1.0

    def test_record_theta(self):

        recording = base.Recording()
        recording.record_theta('X', {'Regularized': 1.0})
        assert recording.df.loc[0]['Regularized'] == 1.0

