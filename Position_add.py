import numpy as np
import matplotlib.pyplot as plt

class Position_learnable():
    def __init__(self, n_patch, embed_dim, fixed=False):
        self.param = np.random.normal(0, 1, (1, n_patch**2, embed_dim))
        self.param_delta = np.zeros_like(self.param)
        self.fixed = fixed
        self.posk = np.zeros((1, n_patch**2, embed_dim))
        for i in range(n_patch**2):
            for j in range(embed_dim):
                if j%2==0:
                    self.posk[0][i][j] = np.sin(i/(10000**(j/embed_dim)))
                else:
                    self.posk[0][i][j] = np.cos(i/(10000**((j - 1)/embed_dim)))
        # plt.imshow(self.posk[0])
        # plt.show()
        # plt.close()

    def forward(self, inputs):
        if self.fixed:
            return self.posk + inputs
        return self.param + inputs

    def backward(self, delta):
        if self.fixed:
            return delta
        self.param_delta += np.sum(delta, axis=(0))
        return delta

    def update(self, lr):
        if self.fixed:
            return 
        self.param -= self.param_delta * lr

    def setzero(self):
        if self.fixed:
            return 
        self.param_delta[...] = 0

    def save_model(self):
        if self.fixed:
            return []
        return [self.param]

    def restore_model(self, models):
        if self.fixed:
            return 
        self.param = models[0]

if __name__=="__main__":
    batchsize = 10
    embed_dim = 300
    n_patch = 10
    inputs = np.random.randn(batchsize, n_patch**2, embed_dim)
    posit = Position_learnable(n_patch, embed_dim)
    output = posit.forward(inputs)

    delta = np.ones_like(output)
    input_delta = posit.backward(delta)
    k = 0