# %%

from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import time
torch.manual_seed(time.time_ns())
# x.unsqueeze(1).float() @ linear.weight.T + linear.bias

y = torch.tensor([[-1, 0.2, 0.9, 2.1], [2.1, 0.9, 0.2, -1]]).T
x = torch.tensor([0, 1, 2, 3])
A = torch.vstack([x, torch.ones(len(x))]).T

result = torch.linalg.lstsq(
    A, y
)

print(A @ result.solution)
print(y)

linear = nn.Linear(1, 2)
# linear.weight = Parameter(
#     result.solution[0].unsqueeze(1)
# )

# linear.bias = Parameter(
#     result.solution[1]
# )

# print(linear.forward(x.unsqueeze(1).float()))

AA = linear.weight.T
# tt = y.T - 

#torch.cat(
#    [linear.weight, linear.bias.unsqueeze(1)], dim=1
#)
print(AA.shape, 'y: ', y.shape)


tt = (y - linear.bias.unsqueeze(0))
print(AA.size())
print(tt.size())

result = torch.linalg.lstsq(
    AA.T,tt.T
)
print(result.solution.shape)
print(result.solution.T)

# print(result.shape, x.shape)

# # got this working
# # 

