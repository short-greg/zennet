# %%
import os
import sys
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

from sklearn.datasets import make_blobs

np.random.seed(1)
th.manual_seed(1)

# %%


X, t = make_blobs(
    n_samples=1000, centers=3, n_features=2,
    random_state=0
)

layer1 = machinery.TorchNN(
    nn.Sequential(nn.Linear(2, 4), nn.Sigmoid()), 
    partial(th.optim.AdamW, lr=1e-2), machinery.WeightedLoss(
        nn.BCELoss(), 1e-2
    )
)

layer2 = machinery.TorchClassNN(
    nn.Sequential(nn.Linear(4, 3)), 
    partial(th.optim.AdamW, lr=1e-2), machinery.WeightedLoss(
        nn.CrossEntropyLoss(), 1e-2
    )
)

machine1 = machinery.Processed(
    [machinery.NP2TH(dtype=th.float32)], layer1
)

sequence = machinery.Sequence(
    [machine1, layer2]
)

t_processor = machinery.NP2TH(dtype=th.int64)
#%%

batch_size = 32

n_iterations = len(X) // batch_size
losses = []
n_epochs = 20

with tqdm.tqdm(total=n_iterations) as tq:

    for _ in range(n_epochs):
        order = np.random.permutation(len(X))
        for i in range(n_iterations):
            from_idx = i * batch_size
            to_idx = (i + 1) * batch_size
            t_i = t[order[from_idx:to_idx]]
            x_i = X[order[from_idx:to_idx]]

            t_i = t_processor.forward(t_i)
            y, ys = sequence.update_ys(x_i)
            assessment = sequence.assess(x_i, t_i, ys)
            losses.append(assessment.item())
            sequence.backward(x_i, t_i, ys)
            # print(next(layer1.module.parameters()))
            tq.set_postfix({'Loss': assessment.item()}, refresh=True)
            tq.update(1)

print(losses)

# %%
