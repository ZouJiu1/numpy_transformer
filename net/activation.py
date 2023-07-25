import numpy as np
from copy import deepcopy
import torch
import math
from scipy.special import erf

class ReLU(object):

    def forward(self, inputs):
        self.inputs = deepcopy(inputs)
        inputs[inputs < 0 ] = 0
        return inputs

    def backward(self, delta):
        return (self.inputs > 0) * delta

    def setzero(self):
        pass

    def update(self, lr = 1e-10):
        pass

class SiLU(object):
    def forward(self, inputs):
        self.e_x = np.exp(-inputs)
        self.div = 1/(1 + self.e_x)
        self.output = inputs * self.div
        return self.output    

    def backward(self, delta):
        cal = self.e_x * self.div
        return (self.div + self.output * cal) * delta

    def setzero(self):
        pass

    def update(self, lr = 1e-10):
        pass

class Softmax(object):
    def forward(self, inputs, axis):
        maxval = np.max(inputs)
        self.inputs = inputs - maxval
        try:
            self.out = np.exp(self.inputs) / np.sum(np.exp(self.inputs), axis = axis, keepdims=True)
        except:
            k = 0
        return self.out

    def backward(self, delta, out):
        k = (out - out**2) * delta
        return k

    def setzero(self):
        pass

    def update(self, lr = 1e-10):
        pass
    
class GELU(object):
    # https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/activations/activations.py
    def forward(self, inputs):
        self.inputs = inputs
        out = (1/2.0) * inputs * (1 + erf(inputs / np.sqrt(2)))
        return out

    def backward(self, delta):
        s = self.inputs / np.sqrt(2)
        erf_prime = lambda x: (2 / np.sqrt(np.pi)) * np.exp(-(x ** 2))
        dx = 0.5 + 0.5 * erf(s) + ((0.5 * self.inputs * erf_prime(s)) / np.sqrt(2))
        return dx * delta

    def setzero(self):
        pass

    def update(self, lr = 1e-10):
        pass

if __name__=="__main__":
    # ReLU()
    input = np.random.rand(10, 300)
    delta = np.random.rand(10, 300)
    gelu = torch.nn.GELU().requires_grad_(True)
    i = torch.tensor(input, requires_grad=True)
    out = gelu(i)
    d = torch.tensor(delta)
    out.backward(d)
    i.retain_grad()
    k = i.grad
    
    g = GELU()
    o = g.forward(input)
    kk = g.backward(delta)
    assert np.mean(np.abs(kk - k.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(kk - k.cpu().detach().numpy()))