# %%

from . import machinery
import sklearn
import torch

from sklearn.datasets import make_blobs
X, y = make_blobs(
    n_samples=10, centers=3, n_features=2,
    random_state=0)


