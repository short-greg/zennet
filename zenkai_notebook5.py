# %%
from abc import ABC, abstractmethod, abstractproperty
import os
import statistics
import sys
import typing
import tqdm

from sklearn import multioutput

def set_env():
    print(os.chdir('../'))
    
    dir_to_add = os.path.realpath(os.getcwd()) 
    # dir_to_add = os.path.realpath(os.path.join(os.getcwd(), '')) 
    sys.path.insert(0, dir_to_add)
set_env()

from functools import partial
from zenkai import machinery
import sklearn
import torch as th
import torch.nn as nn
import numpy as np

from sklearn.datasets import make_blobs, make_classification

np.random.seed(1)
# th.manual_seed(1)

# %%

from torchvision.datasets import MNIST
from torchvision import transforms


recorder = machinery.EuclidRecorder()


# objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
# net = nn.Linear(2, 2)
# optimizer = mac.THOptimBuilder().hill_climber()(net, objective)

objective2 = machinery.LossObjective(nn.CrossEntropyLoss, reduction=machinery.MeanReduction())
objective1 = machinery.LossObjective(nn.MSELoss, reduction=machinery.MeanReduction())
# net1 = nn.Sequential(nn.Linear(2, 4), nn.Sigmoid())
net2 = nn.Sequential(nn.Linear(8, 10))
builder = machinery.THXPBuilder(
    machinery.SingleOptimBuilder().step_hill_climber().repeat(4), # step_hill_climber(momentum=None, std=0.2, update_after=600).repeat(40),
    machinery.SingleOptimBuilder().step_hill_climber(momentum=None, update_after=400).repeat(10)
)


# sequence = machinery.TorchNN(
#     nn.Sequential(net1, net2), objective2, builder
# )
from sklearn import svm

layer1_machines = []
for i in range(8):
    layer1_machines.append(svm.SVR())


layer1 = machinery.SklearnMachine(
    layer1_machines, objective1, builder
)

layer2 = machinery.TorchNN(
    net2, objective2, builder
)


sequence = machinery.Sequence(
    [layer1, layer2]
)
# sequence = layer2
# print('result: ', layer1.backward(X, t))


batch_size = 2048

# n_iterations = len(X) // batch_size
losses = []
n_epochs = 20
transform = transforms.Compose([transforms.ToTensor()])

# blobs = SimpleClassification(batch_size=batch_size)
mnist_trainset = MNIST(root='./data', train=True, download=True, transform=None)
import torch.utils.data as data_utils

dataloader = data_utils.DataLoader(mnist_trainset, shuffle=True)

for _ in range(n_epochs):
    with tqdm.tqdm(total=len(dataloader)) as tq:   
        for x_i, t_i in dataloader:

            x_i = th.tensor(x_i, dtype=th.float32) / 255.0
            t_i = th.tensor(t_i, dtype=th.int64).view(-1)
            # t_i = t_processor.forward(t_i)
            y, ys = sequence.output_ys(x_i)
            assessment = sequence.assess(y, t_i)
            losses.append(assessment.item())
            sequence.backward_update(x_i, t_i, update_theta=True, recorder=recorder)
            # print(next(layer1.module.parameters()))
            recorder.adv()
            tq.set_postfix({'Loss': statistics.mean(losses[-20:])}, refresh=True)
            tq.update(1)
        

# print(recorder.input_df)
print(recorder.theta_df)
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
