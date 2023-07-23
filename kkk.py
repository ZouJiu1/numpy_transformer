import numpy as np
import torch

k = np.arange(2*2*3).reshape((2, 2, 3))
k = torch.from_numpy(k)
kk = torch.flatten("C")

i = 0