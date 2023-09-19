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
        maxval = np.max(inputs, axis = axis, keepdims=True)
        self.inputs = inputs - maxval
        self.axis = axis
        try:
            self.out = np.exp(self.inputs) / np.sum(np.exp(self.inputs), axis = axis, keepdims=True)
        except Exception as e:
            k = 0
        return self.out

    def backward(self, delta, out):
        # https://zhuanlan.zhihu.com/p/657177292
        if len(out.shape)!=2 or self.axis not in [1, -1]:
            exit(-1)
        shape0, shape1 = out.shape
        d = np.zeros((shape0, shape1))
        for i in range(shape0):
            kk = out[i].reshape((-1, 1))
            '''
                k0 = np.diagflat(kk)     ## 将对角线填充完整，也就是 out填充了对角线, 其他都是0的 
                k1 = np.dot(kk, kk.T)    ## nx1 dot 1xn = nxn, 也就是 element 的 multiply
            '''
            kkk = np.diagflat(kk) - np.dot(kk, kk.T)
            row_delta = np.array([delta[i, :]])
            k = np.dot(row_delta, kkk)
            d[i, :] += k[0]
        return d

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