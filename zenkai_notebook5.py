# %%
from abc import ABC, abstractmethod, abstractproperty
import os
import statistics
import sys
import typing
import tqdm
import sklearn.multioutput
from torchvision import transforms

from sklearn import multioutput

def set_env():
    print(os.chdir('../'))
    
    dir_to_add = os.path.realpath(os.getcwd()) 
    # dir_to_add = os.path.realpath(os.path.join(os.getcwd(), '')) 
    sys.path.insert(0, dir_to_add)
set_env()

from functools import partial
from zenkai import machinery, modules, optimization
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

objective2 = modules.LossObjective(nn.CrossEntropyLoss, reduction=modules.MeanReduction())
objective1 = modules.LossObjective(nn.MSELoss, reduction=modules.MeanReduction())
# net1 = nn.Sequential(nn.Linear(2, 4), nn.Sigmoid())
net2 = nn.Sequential(nn.BatchNorm1d(8), nn.Linear(8, 10))
builder2 = optimization.THXPBuilder(
    optimization.SingleOptimBuilder().grad(), 
    # step_hill_climber(momentum=None, std=0.2, update_after=600).repeat(40),
    optimization.SingleOptimBuilder().grad().repeat(10)
)
builder = optimization.SKOptimBuilder(
    optimization.SingleOptimBuilder().step_hill_climber().repeat(4)
)

# sequence = machinery.TorchNN(
#     nn.Sequential(net1, net2), objective2, builder
# )
from sklearn import svm


# for i in range(8):
#    layer1_machines.append(svm.SVR())
machine = sklearn.multioutput.MultiOutputRegressor(
    svm.SVR(kernel='rbf', epsilon=0.05)
)

layer1 = machinery.SklearnMachine(
    modules.SKLearnModule(machine, 64, 8),  objective1, builder
)

layer2 = machinery.TorchNN(
    net2, objective2, builder2
)
sequence = machinery.Sequence(
    [layer1, layer2]
)
# sequence = layer2
# print('result: ', layer1.backward(X, t))
#%%

batch_size = 1600

# n_iterations = len(X) // batch_size
losses = []
n_epochs = 100
transform = transforms.Compose([transforms.ToTensor()])

# blobs = SimpleClassification(batch_size=batch_size)
from .exp_utils.datasets import Digits

digits_trainset = Digits(True)
# mnist_trainset = MNIST(root='./data', train=True, download=True, transform=transform)
import torch.utils.data as data_utils

dataloader = data_utils.DataLoader(digits_trainset, shuffle=True, batch_size=batch_size, drop_last=True)
accuracies = [] 
with tqdm.tqdm(total=len(dataloader) * n_epochs) as tq:   
    for _ in range(n_epochs):
        for x_i, t_i in dataloader:

            x_i = x_i.view(x_i.shape[0], -1).float() # / 255.0
            t_i = t_i.long().view(-1)
            # t_i = t_processor.forward(t_i)
            y, ys = sequence.output_ys(x_i)
            accuracies.append((th.argmax(y, dim=1) == t_i.flatten()).float().mean().item())
            assessment = sequence.assess(y, t_i)
            losses.append(assessment.item())
            sequence.backward_update(x_i, t_i, update_theta=True, recorder=recorder)
            # print(next(layer1.module.parameters()))
            recorder.adv()
            tq.set_postfix(
                {'Loss': statistics.mean(losses[-2:]), 
                'Accuracy': statistics.mean(accuracies[-2:])}, 
                refresh=True
            )
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
