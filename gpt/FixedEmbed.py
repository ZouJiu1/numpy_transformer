import os
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)

import numpy as np
import matplotlib.pyplot as plt

class Position_Fixed():
    def __init__(self, context_length, embed_dim):
        self.posk = np.zeros((context_length, embed_dim))
        for i in range(context_length):
            for j in range(embed_dim):
                if j%2==0:
                    self.posk[i][j] = np.sin(i/(10000**(j/embed_dim)))
                else:
                    self.posk[i][j] = np.cos(i/(10000**((j - 1)/embed_dim)))
        # plt.imshow(self.posk)
        # plt.show()
        # plt.close()

    def forward(self, inputs):
        return self.posk[inputs, :]

    def backward(self, delta):
        return delta

    def update(self, lr):
        return

    def setzero(self):
        return

    def save_model(self):
        return []

    def restore_model(self, models):
        return

if __name__=="__main__":
    batchsize = 10
    embed_dim = 300
    n_patch = 10
    inputs = np.random.randn(batchsize, n_patch**2, embed_dim)
    posit = Position_Fixed(n_patch, embed_dim)
    output = posit.forward(inputs)

    delta = np.ones_like(output)
    input_delta = posit.backward(delta)
    k = 0