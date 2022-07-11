from abc import abstractmethod, abstractproperty
import typing
from .base import Objective,Machine
import torch
import torch.nn as nn
from torch.nn import functional as nn_func

