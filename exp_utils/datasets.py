from sklearn.datasets import load_digits
from torch.utils.data import Dataset
import torch

# %%

import typing
import sklearn
import torch as th
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_blobs, make_classification


class Digits(Dataset):

    def __init__(self, train: bool=True):

        X, y = load_digits(return_X_y=True)
        self._X = torch.from_numpy(X).type(torch.float32)
        self._y = torch.from_numpy(y).type(torch.long)
        generator = torch.Generator()
        # make sure the text and train sets are always the same
        generator.manual_seed(1)
        order = torch.randperm(len(X), generator=generator)
        n_train = int(len(X) * 0.9)
        if train:
            self._order = order[:n_train]
        else:
            self._order = order[n_train:]
    
    def __getitem__(self, idx):
        return self._X[self._order[idx]], self._y[self._order[idx]]
    
    def __len__(self):
        return len(self._order)



class ZenDataset(object):

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


class Blobs(ZenDataset):

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



class SimpleClassification(ZenDataset):

    def __init__(self, shuffle=True, batch_size: int=128):
        X, t = make_classification(
            n_samples=20000, n_features=2, n_redundant=0, n_informative=2, 
            n_clusters_per_class=1, n_classes=3
        )   

        super().__init__(X, t, shuffle, batch_size)

