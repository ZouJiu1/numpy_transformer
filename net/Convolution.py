# https://zhuanlan.zhihu.com/p/642043155
import numpy as np
import torch 
from torch import nn
from copy import deepcopy

def torch_compare_convolution(in_channel, out_channel, kernel_size, stride, padding, bias, inputs, params, bias_params):
    network = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, \
        bias = bias).requires_grad_(True)
    cnt = 0
    for i in network.parameters():
        if cnt==0:
            i.data = torch.from_numpy(params)
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

class convolution_layer(object):
    def __init__(self, in_channel, out_channel, kernel_size, stride=[1,1], padding=[0,0], bias=True, params=[], bias_params=[]):
        self.in_channel = in_channel
        self.out_channel = out_channel
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        self.kernel_size = kernel_size
        self.bias = bias
        if list(params)!=[]:
            self.params = params
        else:
            ranges = np.sqrt(1 / (in_channel * self.kernel_size[0] * self.kernel_size[1]))
            self.params = np.random.uniform(-ranges, ranges, (out_channel, in_channel, kernel_size[0], kernel_size[1]))

        if bias and list(bias_params)!=[]:
            self.bias_params = bias_params
        else:
            ranges = np.sqrt(1 / (in_channel * self.kernel_size[0] * self.kernel_size[1]))
            self.bias_params = np.random.uniform(-ranges, ranges, (out_channel))

        self.params_delta = np.zeros((out_channel, in_channel, kernel_size[0], kernel_size[1])).astype(np.float64)
        self.bias_delta = np.zeros(out_channel).astype(np.float64)
        if isinstance(stride, int):
            stride = [stride, stride]
        self.stride = stride
        if isinstance(padding, int):
            padding = [padding, padding]
        self.padding = padding

    def im2col(self, kernel_size, outshape, inchannel, pad_input, stride):
        im_col = np.zeros((outshape[0] * outshape[2] * outshape[3], \
                        np.prod(kernel_size) * inchannel)).astype(np.float64) # (ob * oh * ow, kw*kh*ic)
        cnt = 0
        for i in range(outshape[0]):
            for h in range(outshape[2]):
                h_start = h * stride[0]
                for w in range(outshape[3]):
                    w_start = w * stride[1]
                    kernel_size_channel = []
                    for j in range(inchannel): # in_channel
                        slide_window = pad_input[i, j, h_start:h_start + kernel_size[0], \
                                            w_start:w_start + kernel_size[1]]
                        flatten = slide_window.flatten()
                        kernel_size_channel.extend(flatten)
                    im_col[cnt, :] = kernel_size_channel
                    cnt += 1
        return im_col

    def common_calcul(self):
        output = np.zeros(self.outshape).astype(np.float64)
        for i in range(self.outshape[0]):
            for oc in range(self.outshape[1]):
                for h in range(self.outshape[2]):
                    h_start = h*self.stride[0]
                    for w in range(self.outshape[3]):
                        w_start = w*self.stride[1]
                        sum = 0
                        for j in range(self.ishape[1]):
                            slide_window = self.pad_input[i, j, h_start:h_start + self.kernel_size[0], \
                                                w_start:w_start + self.kernel_size[1]]
                            sum += np.sum(slide_window * self.params[oc, j, :, :])
                        output[i, oc, h, w] = sum
        return output

    def forward(self, inputs):
        self.inputs = inputs
        self.ishape = self.inputs.shape
        if isinstance(self.padding, str):
            if self.padding=='same':
                self.padding = [self.kernel_size[0]//2, self.kernel_size[1]//2]
            elif self.padding=='valid':
                self.padding = [0, 0]

        self.oh = (self.ishape[2] + 2 * self.padding[0] - self.kernel_size[0])//self.stride[0] + 1
        self.ow = (self.ishape[3] + 2 * self.padding[1] - self.kernel_size[1])//self.stride[1] + 1
        self.outshape = (self.ishape[0], self.out_channel, self.oh, self.ow)

        if np.sum(self.padding)!=0:
            self.pad_input = np.lib.pad(self.inputs, \
                ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), \
                                mode='constant', constant_values=(0, 0))
        else:
            self.pad_input = self.inputs

        im_col = self.im2col(self.kernel_size, self.outshape, self.ishape[1], self.pad_input, self.stride)
        kernel_col = np.reshape(self.params, (self.params.shape[0], -1)).T
        # (ob * oh * ow, kw*kh*ic) * (kw*kh*ic, oc)    =   (ob * oh * ow, oc)
        output = np.matmul(im_col, kernel_col) 
        output = np.reshape(output, (self.outshape[0], self.outshape[2], self.outshape[3], self.outshape[1]))
        output = np.transpose(output, (0, 3, 1, 2))
        
        # output = self.common_calcul()
        if self.bias:
            output = output + self.bias_params[np.newaxis, :, np.newaxis, np.newaxis]

        return output
    
    def backward_common(self, delta, lr = 1e-10):
        # previous layer delta
        input_delta = np.zeros_like(self.pad_input).astype(np.float64)
        if self.bias:
            self.bias_delta += np.sum(delta, axis=(0, 2, 3))
        for i in range(self.outshape[0]):
            for oc in range(self.outshape[1]):
                for h in range(self.outshape[2]):
                    h_start = h*self.stride[0]
                    for w in range(self.outshape[3]):
                        delta_val = delta[i, oc, h, w]
                        w_start = w*self.stride[1]
                        for j in range(self.ishape[1]):
                            slide_window = self.pad_input[i, j, h_start:h_start + self.kernel_size[0], \
                                                w_start:w_start + self.kernel_size[1]]
                            self.params_delta[oc, j, :, :] += slide_window * delta_val
                            input_delta[i, j, h_start:h_start + self.kernel_size[0], \
                                w_start:w_start + self.kernel_size[1]] += self.params[oc, j, :, :] * delta_val
        ih = self.pad_input.shape[-2]
        iw = self.pad_input.shape[-1]
        input_delta = input_delta[:, :, self.padding[0]:ih-self.padding[0], self.padding[1]:iw-self.padding[1]]
        return input_delta, self.params_delta, self.bias_delta
    
    def backward(self, delta):
        # previous layer delta
        if self.bias:
            self.bias_delta += np.sum(delta, axis=(0, 2, 3))
        N_out, C_out, H_out, W_out = self.outshape
        S_h, S_w = self.stride
        
        # internel pad delta
        internal_pad_H = (H_out - 1) * (S_h - 1) + H_out  #IPH
        internal_pad_W = (W_out - 1) * (S_w - 1) + W_out  #IPW
        pad_delta = np.zeros((N_out, C_out, internal_pad_H, internal_pad_W))
        pad_delta[:, :, ::S_h, ::S_w] = delta

        #calcul params gradient
        H_outshape_params_gradient = (self.ishape[2] + 2 * self.padding[0] - internal_pad_H) + 1  #OH
        W_outshape_params_gradient = (self.ishape[3] + 2 * self.padding[1] - internal_pad_W) + 1  #OW
        outshape_params_gradient = (self.ishape[1], self.outshape[1], H_outshape_params_gradient, \
                                    W_outshape_params_gradient) # ic oc OH OW
        kernel_params_gradient = (internal_pad_H, internal_pad_W)
        pad_input_T = np.transpose(self.pad_input, (1, 0, 2, 3)) #(ic, N, ih, iw)
        im_col = self.im2col(kernel_params_gradient, outshape_params_gradient, pad_input_T.shape[1], pad_input_T, [1, 1])
        pad_delta_T = np.transpose(pad_delta, (1, 0, 2, 3)) #(oc, N, IPH, IPW)
        kernel_col = np.reshape(pad_delta_T, (pad_delta_T.shape[0], -1)).T
        # (ic * OH * OW, IPH * IPH * N) * (IPH * IPH * N, oc) = (ic * OH * OW, oc)
        output = np.matmul(im_col, kernel_col) 
        output = np.reshape(output, (outshape_params_gradient[0], \
            outshape_params_gradient[2], outshape_params_gradient[3], outshape_params_gradient[1]))
        params_delta = np.transpose(output, (3, 0, 1, 2))
        if H_outshape_params_gradient > self.kernel_size[0] and W_outshape_params_gradient > self.kernel_size[1]:
            self.params_delta += params_delta[:, :, :self.kernel_size[0], :self.kernel_size[1]]
        elif H_outshape_params_gradient > self.kernel_size[0]:
            self.params_delta += params_delta[:, :, :self.kernel_size[0], :]
        elif W_outshape_params_gradient > self.kernel_size[1]:
            self.params_delta += params_delta[:, :, :, :self.kernel_size[1]]
        else:
            self.params_delta += params_delta
        
        # calcul externel pad shape and input_delta
        remain_h = (self.ishape[2] + 2 * self.padding[0] - self.kernel_size[0]) % self.stride[0]
        remain_w = (self.ishape[3] + 2 * self.padding[1] - self.kernel_size[1]) % self.stride[1]
        pad_top = self.kernel_size[0] - 1 - self.padding[0]
        pad_bottom = self.kernel_size[0] - 1 - self.padding[0] + remain_h
        pad_left = self.kernel_size[1] - 1 - self.padding[1]
        pad_right = self.kernel_size[1] - 1 - self.padding[1] + remain_w
        pad_delta_external = np.lib.pad(pad_delta, \
                ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), \
                                mode='constant', constant_values=(0, 0)) #(N, oc, EPH, EPW)
        #rotate 180
        cp_params = deepcopy(self.params)
        # cp_params = np.rot90(cp_params, 2, axes=(2, 3))
        cp_params = np.flip(np.flip(cp_params, 2), 3)
        
        params_T = np.transpose(cp_params, (1, 0, 2, 3)) #(ic, oc, kh, kw)
        im_col = self.im2col((params_T.shape[2], params_T.shape[3]), self.ishape, params_T.shape[1], pad_delta_external, [1, 1])
        kernel_col = np.reshape(params_T, (params_T.shape[0], -1)).T
        # (N * ih * iw, kw * kh * oc) * (kw * kh * oc, ic) = (N * ih * iw, ic)
        output = np.matmul(im_col, kernel_col)
        output = np.reshape(output, (self.ishape[0], self.ishape[2], self.ishape[3], self.ishape[1]))
        input_delta = np.transpose(output, (0, 3, 1, 2))
        
        return input_delta

    def getdelta(self):
        if self.bias:
            return []
        return [self.params_delta]
    
    def normdelta(self, maxnorm, l2norm):
        self.delta = self.delta * maxnorm / l2norm

    def setzero(self):
        self.params_delta[...]  = 0.0
        self.bias_delta[...] = 0.0

    def update(self, lr = 1e-10):
        self.params -= self.params_delta * lr
        if self.bias:
            self.bias_params   -= self.bias_delta * lr

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
        return "convolution_layer"

def train_single():
    inputs = np.random.rand(2, 6, 10, 30).astype(np.float64)
    outputs = np.random.rand(2, 10, 10, 30).astype(np.float64)
    batchsize = inputs.shape[0]
    in_channel = inputs.shape[1]
    ih = inputs.shape[2]
    iw = inputs.shape[3]
    out_channel = 10
    kernel_size = [3, 3]
    stride = [1, 1]
    padding = [1, 1]
    bias = False
    params = np.random.standard_normal((out_channel, in_channel, kernel_size[0], kernel_size[1])) / np.sqrt(in_channel/2) / 10
    params = params.astype(np.float64)
    bias_params = None
    if bias:
        bias_params = np.random.standard_normal(out_channel) / np.sqrt(in_channel/2)
        bias_params = bias_params.astype(np.float64)
    convolution = convolution_layer(in_channel, out_channel, kernel_size, stride, padding, bias, params, bias_params)
    for i in range(3000):
        out = convolution.forward(inputs)
        sum = np.sum((outputs - out) * (outputs - out))
        delta = 2*(out - outputs)
        # partial_, = convolution.backward_common(delta)
        partial = convolution.backward(delta)
        convolution.update(0.0001)
        convolution.setzero()
        print(sum)

if __name__=="__main__":
    train_single()
    
    inputs = np.random.rand(2, 6, 100, 300).astype(np.float64)
    batchsize = inputs.shape[0]
    in_channel = inputs.shape[1]
    ih = inputs.shape[2]
    iw = inputs.shape[3]
    out_channel = 10
    kernel_size = [3, 3]
    stride = [3, 3]
    padding = [1, 1]
    bias = True
    params = np.random.standard_normal((out_channel, in_channel, kernel_size[0], kernel_size[1])) / np.sqrt(in_channel/2) / 10
    params = params.astype(np.float64)
    if bias:
        bias_params = np.random.standard_normal(out_channel) / np.sqrt(in_channel/2)
        bias_params = bias_params.astype(np.float64)

    convolution = convolution_layer(in_channel, out_channel, kernel_size, stride, padding, bias, params, bias_params)
    output = convolution.forward(inputs)
    delta = np.ones(convolution.outshape).astype(np.float64)
    # partial_, = convolution.backward_common(delta)
    partial = convolution.backward(delta)

    output_torch, partial_torch, grad_params_torch, grad_bias_torch = torch_compare_convolution(in_channel, out_channel, kernel_size, stride, padding, bias, inputs, params, bias_params)
    assert np.mean(np.abs(output - output_torch.cpu().detach().numpy())) < 1e-3, np.mean(np.abs(output - output_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(partial - partial_torch.cpu().detach().numpy())) < 1e-3, np.mean(np.abs(partial - partial_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(convolution.params_delta - grad_params_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(convolution.params_delta - grad_params_torch.cpu().detach().numpy()))
    if bias:
        assert np.mean(np.abs(convolution.bias_delta - grad_bias_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(convolution.bias_delta - grad_bias_torch.cpu().detach().numpy()))