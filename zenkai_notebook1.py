# %%
import os
import sys

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

from sklearn.datasets import make_blobs

np.random.seed(1)
th.manual_seed(1)

# %%


X, t = make_blobs(
    n_samples=10, centers=3, n_features=2,
    random_state=0
)

layer1 = machinery.TorchNN(
    nn.Sequential(nn.Linear(2, 3), nn.Sigmoid()), 
    partial(th.optim.AdamW, lr=1e-2), machinery.WeightedLoss(
        nn.CrossEntropyLoss(), 1e-2
    )
)

layer = machinery.Processed(
    [machinery.NP2TH(dtype=th.float32)], layer1
)

t_processor = machinery.NP2TH(dtype=th.int64)

t = t_processor.forward(t)
y = layer.forward(X)

# print(layer.backward(X, t))

# %%


layer_nn = machinery.TorchNN(
    nn.Sequential(nn.Linear(2, 3), nn.Sigmoid()), 
    partial(th.optim.AdamW, lr=1), nn.MSELoss()
)

layer1 = machinery.Processed(
    [machinery.NP2TH(dtype=th.float32)], layer_nn
)

t_processor = machinery.NP2TH(dtype=th.int64)

layer2 = machinery.TorchNN(
    nn.Linear(3, 4), partial(th.optim.AdamW, lr=1), 
    machinery.WeightedLoss(nn.CrossEntropyLoss(), 1e-2)
)

sequence = machinery.Sequence([layer1, layer2])

t = t_processor.forward(t)
y = sequence.forward(X)

print(X, t)
# print(layer.assess(X, t))

print(sequence.backward(X, t))

# %%
