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
from zenkai import machinery
import sklearn
import torch as th
import torch.nn as nn
import numpy as np

from sklearn.datasets import make_blobs, make_classification

np.random.seed(1)
# th.manual_seed(1)

# %%


class Dataset(object):

    def __init__(self, X, t, shuffle=True, batch_size: int=128):
        self.X = X
        self.t = t
        self.shuffle = shuffle
        self.batch_size = batch_size
    
    @property
    def n_iterations(self):
        return len(self.X) // self.batch_size
    
    def __iter__(self) -> typing.Iterator:

        if self.shuffle:
            order = np.random.permutation(len(self.X))
        else:
            order = np.arange(0, len(self.X))
        batch_size = self.batch_size
        for i in range(self.n_iterations):
            from_idx = i * batch_size
            to_idx = (i + 1) * batch_size
            t_i = self.t[order[from_idx:to_idx]]
            x_i = self.X[order[from_idx:to_idx]]
            yield x_i, t_i


class Blobs(Dataset):

    def __init__(self, shuffle=True, batch_size: int=128):
        X, t = make_blobs(
            n_samples=20000, centers=3, n_features=2, 
            random_state=0
        )
        
        super().__init__(X, t, shuffle, batch_size)



class Blobs2(Dataset):

    def __init__(self, shuffle=True, batch_size: int=128):
        X, t = make_blobs(
            n_samples=20000, centers=6, n_features=2, 
            random_state=0
        )
        t[t==1] = 2
        t[t==2] = 0
        t[t==3] = 1
        t[t==4] = 2
        t[t==5] = 1

        super().__init__(X, t, shuffle, batch_size)



class SimpleClassification(Dataset):

    def __init__(self, shuffle=True, batch_size: int=128):
        X, t = make_classification(
            n_samples=20000, n_features=2, n_redundant=0, n_informative=2, 
            n_clusters_per_class=1, n_classes=3
        )   

        super().__init__(X, t, shuffle, batch_size)



# objective = mac.LossObjective(nn.MSELoss, reduction=mac.MeanReduction())
# net = nn.Linear(2, 2)
# optimizer = mac.THOptimBuilder().hill_climber()(net, objective)

objective2 = machinery.LossObjective(nn.CrossEntropyLoss, reduction=machinery.MeanReduction())
objective1 = machinery.LossObjective(nn.MSELoss, reduction=machinery.MeanReduction())
net1 = nn.Sequential(nn.Linear(2, 4), nn.Sigmoid())
net2 = nn.Sequential(nn.Linear(4, 3))
builder = machinery.THXPBuilder(
    machinery.SingleOptimBuilder().step_hill_climber(momentum=None, std=0.2, update_after=600).repeat(2),
    machinery.SingleOptimBuilder().step_hill_climber(momentum=None, update_after=400)
)


# sequence = machinery.TorchNN(
#     nn.Sequential(net1, net2), objective2, builder
# )

layer1 = machinery.TorchNN(
    net1, objective1, builder
)

layer2 = machinery.TorchNN(
    net2, objective2, builder
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

blobs = SimpleClassification(batch_size=128)

for _ in range(n_epochs):
    n_iterations = blobs.n_iterations
    with tqdm.tqdm(total=n_iterations) as tq:   
        for x_i, t_i in blobs:

            x_i = th.tensor(x_i, dtype=th.float32)
            t_i = th.tensor(t_i, dtype=th.int64).view(-1)
            # t_i = t_processor.forward(t_i)
            y, ys = sequence.update_outs(x_i)
            assessment = sequence.assess(x_i, t_i, ys)
            losses.append(assessment.item())
            sequence.backward(x_i, t_i, ys, update_inputs=False)
            # print(next(layer1.module.parameters()))
            tq.set_postfix({'Loss': statistics.mean(losses[-20:])}, refresh=True)
            tq.update(1)
        

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
