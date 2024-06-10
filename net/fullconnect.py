# https://zhuanlan.zhihu.com/p/642043155
import numpy as np
import torch 
from torch import nn

def torch_compare_fc(infeature, outfeature, bias, inputs, params, bias_params):
    network = nn.Linear(infeature, outfeature, bias).requires_grad_(True)
    cnt = 0
    for i in network.parameters():
        if cnt==0:
            i.data = torch.from_numpy(params.T)
            i.retain_grad = True
        else:
            i.data = torch.from_numpy(bias_params)
            i.retain_grad = True
        cnt += 1
            
    inputs = torch.tensor(inputs, requires_grad=True)
    output = network(inputs)
    sum = torch.sum(output) # make sure the gradient is 1
    kk = sum.backward()
    grad_params = 0
    grad_bias   = 0
    cnt = 0
    for i in network.parameters():
        if cnt==0:
            grad_params = i.grad
        else:
            grad_bias = i.grad
        cnt += 1
    inputs.retain_grad()
    k = inputs.grad
    return output, k, grad_params, grad_bias

class fclayer(object):
    def __init__(self, infeature, outfeature, bias=False, params=[], bias_params=[], adam = False, float32=False, float16 = False, name='', init = ''):
        self.infeature = infeature
        self.outfeature = outfeature
        self.bias = bias
        self.float32 = float32
        if list(params)!=[]:
            self.params = params
        else:
            ranges = np.sqrt(1 / infeature)
            self.params = np.random.uniform(-ranges, ranges, (infeature, outfeature))
            if self.float32:
                self.params = self.params.astype(np.float32)
            if float16:
                self.params = self.params.astype(np.float16)
        if bias and list(bias_params)!=[]:
            self.bias_params = bias_params
        else:
            ranges = np.sqrt(1 / infeature )
            self.bias_params = np.random.uniform(-ranges, ranges, (outfeature))
            if self.float32:
                self.bias_params = self.bias_params.astype(np.float32)
            if float16:
                self.bias_params = self.bias_params.astype(np.float16)
        self.params_delta = np.zeros((infeature, outfeature))
        if self.float32:
            self.params_delta = self.params_delta.astype(np.float32)
        if float16:
            self.params_delta = self.params_delta.astype(np.float16)
        self.bias_delta = np.zeros(outfeature)
        if self.float32:
            self.bias_delta = self.bias_delta.astype(np.float32)
        if float16:
            self.bias_delta = self.bias_delta.astype(np.float16)
        
        self.adam = adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsadam = 10**(2-10)
        if self.float32:
            self.moment_p = np.zeros_like(self.params, dtype=np.float32)
        elif float16:
            self.moment_p = np.zeros_like(self.params, dtype=np.float16)
        else:
            self.moment_p = np.zeros_like(self.params)
        if self.float32:
            self.rmsprop_p = np.zeros_like(self.params, dtype=np.float32)
        elif float16:
            self.rmsprop_p = np.zeros_like(self.params, dtype=np.float16)
        else:
            self.rmsprop_p = np.zeros_like(self.params)
        if bias:
            self.moment_b = np.zeros_like(self.bias_params)
            if self.float32:
                self.moment_b = self.moment_b.astype(np.float32)
            elif float16:
                self.moment_b = self.moment_b.astype(np.float16)
            self.rmsprop_b = np.zeros_like(self.bias_params)
            if self.float32:
                self.rmsprop_b = self.rmsprop_b.astype(np.float32)
            elif float16:
                self.rmsprop_b = self.rmsprop_b.astype(np.float16)
        self.t = 1
    
    def forward(self, inputs):
        if self.float32:
            inputs = inputs.astype(np.float32)
        self.inputs = inputs
        output = np.matmul(inputs, self.params)
        if self.bias:
            output = output + self.bias_params[np.newaxis, :]
        return output
    
    def backward(self, delta, inputs=[]):
        if self.float32:
            delta = delta.astype(np.float32)
            if len(inputs):
                inputs = inputs.astype(np.float32)
        #previous layer delta
        input_delta = np.matmul(delta, self.params.T)
        if len(inputs)==0:
            inputs = self.inputs
        #params bias delta
        if len(delta.shape)==3:
            delta__ = np.reshape(delta, (-1, delta.shape[-1]))
            inputs__ = np.reshape(inputs, (-1, inputs.shape[-1]))
            self.params_delta += np.matmul(delta__.T, inputs__).T
            self.bias_delta += np.sum(delta__, axis = 0)
        else:
            self.params_delta += np.matmul(delta.T, inputs).T
            self.bias_delta += np.sum(delta, axis = 0)
        
        return input_delta

    def setzero(self):
        self.params_delta[...] = 0
        if self.bias:
            self.bias_delta[...] = 0

    def update(self, lr = 1e-10):
        # self.params_delta = np.clip(self.params_delta, -6, 6)
        # self.bias_delta = np.clip(self.bias_delta, -6, 6)
        if self.adam:
            self.moment_p = self.beta1 * self.moment_p + (1 - self.beta1) * self.params_delta
            self.rmsprop_p = self.beta2 * self.rmsprop_p + (1 - self.beta2) * self.params_delta**2
            moment_p = self.moment_p / np.sqrt(1 - self.beta1**self.t)
            rmsprop_p = self.rmsprop_p / np.sqrt(1 - self.beta2**self.t)
            self.params -= (moment_p * lr / (np.sqrt(rmsprop_p)+ self.epsadam))
            if self.bias:
                self.moment_b = self.beta1 * self.moment_b + (1 - self.beta1) * self.bias_delta
                self.rmsprop_b = self.beta2 * self.rmsprop_b + (1 - self.beta2) * self.bias_delta**2
                moment_b = self.moment_b / np.sqrt(1 - self.beta1**self.t)
                rmsprop_b = self.rmsprop_b / np.sqrt(1 - self.beta2**self.t)
                self.bias_params -= (moment_b * lr / (np.sqrt(rmsprop_b)+ self.epsadam))
            self.t += 1
        else:
            self.params -= self.params_delta * lr
            if self.bias:
                self.bias_params -= self.bias_delta * lr

    def save_model(self):
        if self.bias:
            return [self.params, self.bias_params]
        else:
            return [self.params]

    def restore_model(self, models):
        self.params = models[0].reshape(self.params.shape)
        if self.bias:
            self.bias_params = models[1].reshape(self.bias_params.shape)

    def __name__(self):
        return "fclayer"

def train_single():
    inputs = np.random.rand(100, 1000)
    outputs = np.random.rand(100, 900)
    infeature = inputs.shape[-1]
    outfeature = 900
    bias = False
    delta = np.ones((inputs.shape[0], outfeature), dtype=np.float64)
    params = np.random.standard_normal((infeature, outfeature)) / np.sqrt(infeature/2)
    if bias:
        bias_params = np.random.standard_normal(outfeature) / np.sqrt(infeature/2)
    else:
        bias_params = []
    
    fc = fclayer(infeature, outfeature, bias, params, bias_params)
    for i in range(1000):
        out = fc.forward(inputs)
        sum = np.sum((outputs - out) * (outputs - out))
        delta = 2*(out - outputs)
        partial = fc.backward(delta, inputs)
        fc.update(0.00001)
        fc.setzero()
        print(sum)

if __name__=="__main__":
    train_single()

    inputs = np.random.rand(3, 1000)
    infeature = inputs.shape[-1]
    outfeature = 900
    bias = True
    delta = np.ones((inputs.shape[0], outfeature), dtype=np.float64)
    params = np.random.standard_normal((infeature, outfeature)) / np.sqrt(infeature/2)
    if bias:
        bias_params = np.random.standard_normal(outfeature) / np.sqrt(infeature/2)
    
    fc = fclayer(infeature, outfeature, bias, params, bias_params)
    output = fc.forward(inputs)
    partial = fc.backward(delta, inputs)
    output_torch, partial_torch, grad_params_torch, grad_bias_torch = torch_compare_fc(infeature, outfeature, bias, inputs, params, bias_params)
    assert np.mean(np.abs(output - output_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(output - output_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(partial - partial_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(partial - partial_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(fc.params_delta.T - grad_params_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(fc.params_delta.T - grad_params_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(fc.bias_delta - grad_bias_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(fc.bias_delta - grad_bias_torch.cpu().detach().numpy()))