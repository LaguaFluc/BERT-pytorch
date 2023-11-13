
#%%
a  = [1, 2, 3]
b = [0, 0, 0]
a.extend(b)
a

#%%
import torch
import math
max_len = 5
# torch.arange(0, max_len).float().unsqueeze(1).shape
d_model = 20
(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()