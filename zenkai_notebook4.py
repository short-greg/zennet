# %%
from abc import ABC, abstractmethod, abstractproperty
import os
import statistics
import sys
import typing
import tqdm

def set_env():
    print(os.chdir('../'))
    
    dir_to_add = os.path.realpath(os.getcwd()) 
    # dir_to_add = os.path.realpath(os.path.join(os.getcwd(), '')) 
    sys.path.insert(0, dir_to_add)
set_env()

from functools import partial
from zenkai import machinery, modules, optim_builders, optimizers
import sklearn
import torch as th
import torch.nn as nn
import numpy as np
import exp_utils

from sklearn.datasets import make_blobs, make_classification

np.random.seed(1)
# th.manual_seed(1)

# %%


recorder = machinery.EuclidRecorder()

class Threshold(nn.Module):

    def forward(self, x):
        return (x >= 0.0).type(x.dtype)

# objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
# net = nn.Linear(2, 2)
# optimizer = mac.THOptimBuilder().hill_climber()(net, objective)

objective2 = modules.LossObjective(nn.CrossEntropyLoss, reduction=modules.MeanReduction())
objective1 = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
net1 = nn.Sequential(nn.Linear(2, 16), Threshold())
net2 = nn.Sequential(nn.Linear(16, 3))

theta_builder = optim_builders.ThetaOptimBuilder().step_gaussian_hill_climber().repeat(1)
input_builder2 = optim_builders.InputOptimBuilder().step_binary_hill_climber(8, 0.1, False).repeat(1)

theta_builder2 = optim_builders.ThetaOptimBuilder().grad()
input_builder = optim_builders.InputOptimBuilder().step_gaussian_hill_climber() # (momentum=None).repeat(4)

layer1 = machinery.SklearnMachine(
    modules.Perceptron(2, 16), 
    objective1,
    optim_builders.SklearnOptimBuilder().partial(True).repeat(4),
    input_builder, partial=True
)

# layer1 = machinery.TorchNN(
#     net1, objective1, theta_builder, input_builder
# )
layer2 = machinery.TorchNN(
    net2, objective2, theta_builder2, input_builder2
)
sequence = machinery.Sequence(
    [layer1, layer2]
)
# sequence = layer2
# print('result: ', layer1.backward(X, t))


batch_size = 128

# n_iterations = len(X) // batch_size
losses = []
n_epochs = 10

blobs = exp_utils.datasets.SimpleClassification(batch_size=batch_size)

for _ in range(n_epochs):
    batch_size = min(2048, batch_size * 2)
    n_iterations = blobs.n_iterations
    with tqdm.tqdm(total=n_iterations) as tq:   
        for x_i, t_i in blobs:

            x_i = th.tensor(x_i, dtype=th.float32)
            t_i = th.tensor(t_i, dtype=th.int64).view(-1)
            # t_i = t_processor.forward(t_i)
            y, ys = sequence.output_ys(x_i)
            assessment = sequence.assess(y, t_i)
            losses.append(assessment.item())
            sequence.backward_update(x_i, t_i, update_theta=True)
            # print(next(layer1.module.parameters()))
            recorder.adv()
            tq.set_postfix({'Loss': statistics.mean(losses[-20:])}, refresh=True)
            tq.update(1)
        

# print(recorder.input_df)
# print(recorder.theta_df)
print(losses)

# layer = machinery.Processed(
#     [machinery.NP2TH(dtype=th.float32)], layer1
# )

# t_processor = machinery.NP2TH(dtype=th.int64)
#t = t_processor.forward(t)


# y = layer.forward(X)

# # print(layer.backward(X, t))

# # %%


# layer_nn = machinery.TorchNN(
#     nn.Sequential(nn.Linear(2, 3), nn.Sigmoid()), 
#     partial(th.optim.AdamW, lr=1), nn.MSELoss()
# )

# layer1 = machinery.Processed(
#     [machinery.NP2TH(dtype=th.float32)], layer_nn
# )

# t_processor = machinery.NP2TH(dtype=th.int64)

# layer2 = machinery.TorchNN(
#     nn.Linear(3, 4), partial(th.optim.AdamW, lr=1), 
#     machinery.WeightedLoss(nn.CrossEntropyLoss(), 1e-2)
# )

# sequence = machinery.Sequence([layer1, layer2])

# t = t_processor.forward(t)
# y = sequence.forward(X)

# print(X, t)
# # print(layer.assess(X, t))

# print(sequence.backward(X, t))

# %%
