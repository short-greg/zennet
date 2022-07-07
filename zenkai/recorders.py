import torch
from .base import Evaluation, InputRecorder, ThetaRecorder
import pandas as pd


class EuclidInputRecorder(InputRecorder):
    
    def record(self, x: torch.Tensor, x_prime: torch.Tensor, evaluation: Evaluation):
        deviation = torch.sqrt(torch.sum((x - x_prime) ** 2)).item()
        self._recording.record_inputs(
            self._name, {
                'Deviation': deviation,
                'Regularized Evaluation': evaluation.regularized,
                'Unregularized Evaluation': evaluation.unregularized
            }
        )


class EuclidThetaRecorder(ThetaRecorder):
    
    def record(self, theta: torch.Tensor, theta_prime: torch.Tensor, evaluation: Evaluation):

        deviation = torch.sqrt(torch.sum((theta - theta_prime) ** 2)).item()
        self._recording.record_theta(
            self._name, {
                'Deviation': deviation,
                'Regularized Evaluation': evaluation.regularized,
                'Unregularized Evaluation': evaluation.unregularized
            }
        )


# class EuclidRecorder(Recorder):

#     def __init__(self):
        
#         self._layer_results = {}
#         self._cur_index = 0
#         self._layer_map = {}
#         self._layer_count = 0

#     def adv(self):
#         self._cur_index += 1
    
#     def get_or_set_layer_id(self, layer):
#         if layer not in self._layer_map:
#             self._layer_map[layer] = self._layer_count
#             self._layer_count += 1
#         return self._layer_map[layer]

#     def record_inputs(self, layer, prev_inputs, cur_inputs, evaluations):

#         layer = self.get_or_set_layer_id(layer)
#         if layer not in self._layer_results:
#             self._layer_results[layer] = []
#         if self._cur_index not in self._layer_results[layer]:
#             self._layer_results[layer] += [{}] * (self._cur_index - len(self._layer_results[layer]) + 1)
#         deviation = torch.sqrt(torch.sum((prev_inputs - cur_inputs) ** 2)).item()
#         self._layer_results[layer][self._cur_index]['inputs'] = dict(
#             deviation=deviation,
#             evaluations=evaluations
#         )

#     def record_theta(self, layer, prev_theta, cur_theta, evaluations):

#         layer = self.get_or_set_layer_id(layer)
#         if layer not in self._layer_results:
#             self._layer_results[layer] = []
#         if len(self._layer_results[layer]) <= self._cur_index:

#             self._layer_results[layer] += [{}] * (self._cur_index - len(self._layer_results[layer]) + 1)
#         deviation = torch.sqrt(torch.sum((prev_theta - cur_theta) ** 2)).item()
#         self._layer_results[layer][self._cur_index]['theta'] = dict(
#             deviation=deviation,
#             evaluations=evaluations
#         )

#     @property
#     def pos(self):
#         return self._cur_index

#     @property
#     def theta_df(self):
#         df_results = []
#         for name, layer in self._layer_results.items():
#             for i, results in enumerate(layer):
#                 if 'theta' not in results:
#                     continue
#                 evaluations = {
#                     f'Theta Evaluation {i}':k
#                     for i, k in enumerate(results['theta']['evaluations'])
#                 }

#                 df_results.append({
#                     'Layer': name,
#                     'Step': i,
#                     'Theta Deviation': results['theta']['deviation'],
#                     **evaluations
#                 })
#         return pd.DataFrame(df_results)

#     @property
#     def input_df(self):
#         df_results = []
#         for name, layer in self._layer_results.items():
#             for i, results in enumerate(layer):
#                 if 'inputs' not in results:
#                     continue
#                 evaluations = {
#                     f'Input Evaluation {i}':k
#                     for i, k in enumerate(results['inputs']['evaluations'])
#                 }

#                 df_results.append({
#                     'Layer': name,
#                     'Step': i,
#                     'Input Deviation': results['inputs']['deviation'],
#                     **evaluations
#                 })
#         return pd.DataFrame(df_results)

