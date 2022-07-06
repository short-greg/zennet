

# import torch

# class Layer(object):

#     def forward(self, x):

#         pass
    
#     def forward_update(self, x):
#         pass


#     def backward_update(self, x, t, ys):
        
#         x = utils.freshen(x)
#         if ys is not None:
#             y = ys[2]
#         else:
#             y = ...
        
#         loss = torch.nn.MSELoss(y, t)
#         loss.backward()

#         return x - x.grad

