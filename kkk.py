import numpy as np
import torch

# def create_triangle(shape):
#     one = np.ones((1, shape, shape))
#     trimask = np.triu(one, k = 1)
#     return trimask

# def create_masks(array, num_h):
#     if array is not None:
#         target_masks = (array != -1)
#         target_masks = target_masks[:, None, :]
#         sequence_length = array.shape[1]  #sequence_length
#         trimask = create_triangle(sequence_length)

#         target_masks = trimask & target_masks
#         target_masks = target_masks.astype(np.float32)
#         target_masks = target_masks * (-1e9)
#         target_masks = target_masks[:, None, :, :]
#         target_masks = [target_masks for i in range(num_h)]
#         target_masks = np.concatenate(target_masks, axis=1)
#     else:
#         target_masks = None
#     return target_masks

# array = np.ones((1, 200))
# num_h = 20
# target_masks = create_masks(array, num_h)
# k  =0


kkk = torch.zeros((6, 6))

k = torch.arange(6)

kkk[k, k] = 1

kkk


        