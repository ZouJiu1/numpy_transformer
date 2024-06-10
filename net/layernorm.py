# https://zhuanlan.zhihu.com/p/642043155
import numpy as np
import torch 
from torch import nn
from copy import deepcopy

def torch_compare_layernorm(normalized_shape, inputs, gamma, beta, elementwise_affine, delta=''):
    network = nn.LayerNorm(normalized_shape, elementwise_affine = elementwise_affine).requires_grad_(True)
    network.double()
    cnt = 0
    for i in network.parameters():
        if cnt==0:
            i.data = torch.from_numpy(gamma)
            i.retain_grad = True
        else:
            i.data = torch.from_numpy(beta)
            i.retain_grad = True
        cnt += 1
    inputs = torch.tensor(inputs, requires_grad=True, dtype=torch.float64)
    output = network(inputs)
    delta = torch.tensor(delta)
    output.backward(delta)
    # sum = torch.sum(output) # make sure the gradient is 1
    # kk = sum.backward()
    grad_gamma = 0
    grad_beta   = 0
    cnt = 0
    for i in network.parameters():
        if cnt==0:
            grad_gamma = i.grad
        else:
            grad_beta = i.grad
        cnt += 1
    inputs.retain_grad()
    output.retain_grad()
    k = inputs.grad
    return output, k, grad_gamma, grad_beta

class layer_norm(object):
    def __init__(self, normalized_shape, elementwise_affine = True, gamma = [], beta = [], adam = False, float32=False, float16=False):
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.axis = None
        self.gamma = np.ones(normalized_shape)
        self.beta  = np.zeros(normalized_shape)
        self.elementwise_affine = elementwise_affine
        self.normalized_shape = normalized_shape
        if elementwise_affine and list(gamma)!=[]:
            self.gamma = gamma

        if elementwise_affine and list(beta)!=[]:
            self.beta = beta

        self.gamma_delta = np.zeros(normalized_shape).astype(np.float64)
        self.beta_delta = np.zeros(normalized_shape).astype(np.float64)
        if float32:
            self.gamma_delta = self.gamma_delta.astype(np.float32)
            self.beta_delta = self.beta_delta.astype(np.float32)
            self.gamma = self.gamma.astype(np.float32)
            self.beta = self.beta.astype(np.float32)
        if float16:
            self.gamma_delta = self.gamma_delta.astype(np.float16)
            self.beta_delta = self.beta_delta.astype(np.float16)
            self.gamma = self.gamma.astype(np.float16)
            self.beta = self.beta.astype(np.float16)
        self.adam = adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsadam = 10**(2-10)
        if elementwise_affine:
            self.moment_g = np.zeros_like(self.gamma)
            self.rmsprop_g = np.zeros_like(self.gamma)
            self.moment_b = np.zeros_like(self.beta)
            self.rmsprop_b = np.zeros_like(self.beta)
            if float32:
                self.moment_g = self.moment_g.astype(np.float32)
                self.rmsprop_g = self.rmsprop_g.astype(np.float32)
                self.moment_b = self.moment_b.astype(np.float32)
                self.rmsprop_b = self.rmsprop_b.astype(np.float32)
            if float16:
                self.moment_g = self.moment_g.astype(np.float16)
                self.rmsprop_g = self.rmsprop_g.astype(np.float16)
                self.moment_b = self.moment_b.astype(np.float16)
                self.rmsprop_b = self.rmsprop_b.astype(np.float16)
        self.t = 1
        self.ep = 1e-5

    def forward(self, inputs):
        self.inputs = deepcopy(inputs)
        self.inshape = inputs.shape
        if self.axis==None:
            self.axis = np.arange(len(self.inshape) - len(self.normalized_shape), len(self.inshape))
            self.axis = tuple(list(self.axis))
            self.gamma_axis_delta = []
            for i in range(len(self.inshape)):
                if i not in self.axis:
                    self.gamma_axis_delta.append(i)
            self.gamma_axis_delta = tuple(self.gamma_axis_delta)

        self.mean = np.mean(inputs, axis = self.axis)
        self.var  = np.var(inputs, axis = self.axis)
        for i in range(len(self.axis)):
            self.mean = np.expand_dims(self.mean, -1)
            self.var = np.expand_dims(self.var, -1)
        if len(self.gamma.shape)!=len(self.inputs.shape):
            for i in range(len(self.inshape) - len(self.axis)):
                self.gamma = np.expand_dims(self.gamma, 0)
                self.beta = np.expand_dims(self.beta, 0)
        
        outputs = (inputs - self.mean) / np.sqrt(self.var + self.ep)
        self.normal = deepcopy(outputs)
        if self.elementwise_affine:
            outputs = outputs * self.gamma + self.beta
        return outputs

    def backward(self, delta):
        # previous layer delta
        normal_shape = np.prod(self.normalized_shape)

        if not self.elementwise_affine:
            self.gamma = np.ones(self.normalized_shape)
        mean = self.mean
        gamma = self.gamma
        var = self.var + self.ep

        partone = gamma * delta / np.sqrt(var)
        parttwo = (1 / np.sqrt(var) / normal_shape) * np.sum(delta * gamma, axis=self.axis, keepdims=True)
        partthree = (1 / np.sqrt(var) / normal_shape) * self.normal * np.sum(delta * self.normal * gamma, axis=self.axis, keepdims=True)
        input_delta = partone - parttwo - partthree

        if self.elementwise_affine:
            self.gamma_delta += np.sum(self.normal * delta, axis = self.gamma_axis_delta)
            self.beta_delta += np.sum(delta, axis = self.gamma_axis_delta)

        return input_delta

    def setzero(self):
        self.gamma_delta[...]  = 0.0
        self.beta_delta[...] = 0.0

    def update(self, lr = 1e-10):
        if self.elementwise_affine:
            if self.adam:
                self.moment_g = self.beta1 * self.moment_g + (1 - self.beta1) * self.gamma_delta
                self.rmsprop_g = self.beta2 * self.rmsprop_g + (1 - self.beta2) * self.gamma_delta**2
                moment_g = self.moment_g / np.sqrt(1 - self.beta1**self.t)
                rmsprop_g = self.rmsprop_g / np.sqrt(1 - self.beta2**self.t)
                self.moment_b = self.beta1 * self.moment_b + (1 - self.beta1) * self.beta_delta
                self.rmsprop_b = self.beta2 * self.rmsprop_b + (1 - self.beta2) * self.beta_delta**2
                moment_b = self.moment_b / np.sqrt(1 - self.beta1**self.t)
                rmsprop_b = self.rmsprop_b / np.sqrt(1 - self.beta2**self.t)
                self.t += 1
                self.gamma -= (moment_g * lr / (np.sqrt(rmsprop_g)+ self.epsadam))
                self.beta -= (moment_b * lr / (np.sqrt(rmsprop_b)+ self.epsadam))
            else:
                self.gamma -= self.gamma_delta * lr
                self.beta -= self.beta_delta * lr

    def save_model(self):
        return [self.gamma, self.beta]

    def restore_model(self, models):
        self.gamma = models[0].reshape(self.gamma.shape)
        self.beta = models[1].reshape(self.beta.shape)

def train_single():
    inputs = np.random.rand(2, 3, 60, 60).astype(np.float64)
    outputs = np.random.rand(2, 3, 60, 60).astype(np.float64)
    elementwise_affine = True
    normalized_shape = (60, 60)
    gamma = np.random.rand(60, 60).astype(np.float64)
    beta = np.random.rand(60, 60).astype(np.float64)

    batchnorm = layer_norm(normalized_shape=normalized_shape, elementwise_affine=elementwise_affine, gamma=gamma, beta=beta, adam=False)
    for i in range(60000):
        out = batchnorm.forward(inputs)
        sum = np.sum((inputs - out) * (inputs - out))
        delta = 2*(out - inputs)
        partial = batchnorm.backward(delta)
        batchnorm.update(lr = 0.001)
        batchnorm.setzero()
        print(sum)

if __name__=="__main__":
    #https://pytorch.org/docs/2.0/generated/torch.nn.LayerNorm.html
    #https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    
    train_single()
    
    inputs = np.random.rand(100, 100, 30, 30).astype(np.float64)
    elementwise_affine = True
    normalized_shape = (100, 30, 30)
    gamma = np.random.rand(100, 30, 30).astype(np.float64) # np.ones(normalized_shape) #np.random.rand(normalized_shape).astype(np.float64)
    beta = np.random.rand(100, 30, 30).astype(np.float64) # np.zeros(normalized_shape) #np.random.rand(normalized_shape).astype(np.float64)

    batchnorm = layer_norm(normalized_shape=normalized_shape, elementwise_affine=elementwise_affine, gamma=gamma, beta=beta)
    output = batchnorm.forward(inputs)
    delta = np.ones(inputs.shape).astype(np.float64)
    # delta = np.random.rand(inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]).astype(np.float64)
    partial = batchnorm.backward(delta)

    output_torch, partial_torch, grad_gamma_torch, grad_beta_torch = torch_compare_layernorm(normalized_shape, inputs, gamma, beta, elementwise_affine, delta)
    assert np.mean(np.abs(output - output_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(output - output_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(partial - partial_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(partial - partial_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(batchnorm.gamma_delta - grad_gamma_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(batchnorm.gamma_delta - grad_gamma_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(batchnorm.beta_delta - grad_beta_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(batchnorm.beta_delta - grad_beta_torch.cpu().detach().numpy()))