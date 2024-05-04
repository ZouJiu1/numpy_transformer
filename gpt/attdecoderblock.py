import os
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)

import numpy as np
from net.layernorm import layer_norm
from net.fullconnect import fclayer
from net.activation import Softmax, GELU, ReLU

class attdecoderblock_layer():
    def __init__(self, embed_dim, num_h, adam=False, float32 = False, return_attention = False):
        self.embed_dim = embed_dim
        self.num_h = num_h
        self.len_single =  embed_dim // num_h

        self.norm = layer_norm(self.embed_dim, adam=adam)
        self.norm1 = layer_norm(self.embed_dim, adam=adam)
        self.qkvfc = fclayer(self.embed_dim, self.len_single * num_h * 3, True, adam=adam, float32=float32)
        
        self.fc0 = fclayer(self.embed_dim, self.embed_dim * (2*2), True, adam=adam, float32=float32)
        self.fc1 = fclayer(self.embed_dim * (2*2), self.embed_dim, True, adam=adam, float32=float32)
        self.fc_out = fclayer(self.embed_dim, self.embed_dim, True, adam=adam, float32=float32)
        self.softmax = Softmax()
        self.relu = ReLU()
        self.adam = adam
        self.return_attention = return_attention
    
    def forward(self, inputs, masks = []):
        self.masks = masks
        self.out0 = self.norm.forward(inputs)

        self.out2 = self.fc0.forward(self.out0)
        self.out2 = self.relu.forward(self.out2)
        self.out6 = self.fc1.forward(self.out2)
        
        self.out6 = inputs + self.out6
        
        self.outnorm = self.norm1.forward(self.out6)
        batch, block, _ = self.outnorm.shape
        
        qkv = self.qkvfc.forward(self.outnorm).reshape((batch, block, 2+1, self.num_h, self.len_single))
        result = []
        self.delta_qkv = np.zeros_like(qkv)
        self.batch = batch
        self.block = block
        self.qkv = qkv
        self.atg__ = [[[] for j in range(self.num_h)] for i in range(batch)]
        self.att_col = [[[] for j in range(self.num_h)] for i in range(batch)]
        for n in range(batch):
            tmp = []
            for i in range(self.num_h):
                niq = qkv[n, :, 0, i]
                nik = qkv[n, :, 1, i]
                niv = qkv[n, :, 2, i]
                att = np.matmul(niq, nik.T) / np.sqrt(self.len_single)
                if self.return_attention:
                    self.att_col[n][i] = att
                if len(masks) > 0:
                    att = att + masks
                atg__ = self.softmax.forward(att, axis=-1)
                self.atg__[n][i] = atg__
                rek = np.matmul(atg__, niv)
                tmp.append(rek)
            tmp = np.concatenate(tmp, axis = -1)
            result.append(tmp)
        self.rkk = np.stack(result)

        self.out1 = self.fc_out.forward(self.rkk)
        input1 = self.out6 + self.out1
        if self.return_attention:
            return self.atg__, self.att_col
        return input1

    def backward(self, delta):
        delta0 = self.fc_out.backward(delta, self.rkk)
        for n in range(self.batch):
            n_delta = delta0[n]
            for i in range(self.num_h):
                niq = self.qkv[n, :, 0, i]
                nik = self.qkv[n, :, 1, i]
                niv = self.qkv[n, :, 2, i]
                atg__ = self.atg__[n][i]
                
                now = self.len_single * i
                now_delta = n_delta[:, now : now + self.len_single]

                atg__delta = np.matmul(now_delta, niv.T)
                att_delta = self.softmax.backward(atg__delta, atg__)
                niv_delta = np.matmul(now_delta.T, atg__).T
                
                niq_delta = np.matmul(att_delta, nik) / np.sqrt(self.len_single)
                nik_delta = np.matmul(att_delta.T, niq) / np.sqrt(self.len_single)

                self.delta_qkv[n, :, 2, i] = niv_delta
                self.delta_qkv[n, :, 1, i] = nik_delta
                self.delta_qkv[n, :, 0, i] = niq_delta

        qkvdelta = np.reshape(self.delta_qkv, (self.batch, self.block, -1))
        qkvdelta = self.qkvfc.backward(qkvdelta, self.outnorm)
        qkvdelta = self.norm1.backward(qkvdelta)
        
        qkvdelta = qkvdelta + delta
        d = self.fc1.backward(qkvdelta, self.out2)
        d = self.relu.backward(d)
        d = self.fc0.backward(d, self.out0)
        input_delta = self.norm.backward(d)
        input_delta += qkvdelta
        return input_delta

    def update(self, lr):
        self.norm.update(lr)
        self.norm1.update(lr)
        self.qkvfc.update(lr)
        self.fc0.update(lr)
        self.fc1.update(lr)
        self.fc_out.update(lr)

    def setzero(self):
        self.norm.setzero()
        self.norm1.setzero()
        self.qkvfc.setzero()
        self.fc0.setzero()
        self.fc1.setzero()
        self.fc_out.setzero()

    def save_model(self):
        return [self.norm.save_model(), self.qkvfc.save_model(), self.norm1.save_model(), \
            self.fc0.save_model(), self.fc1.save_model(), self.fc_out.save_model()]

    def restore_model(self, models):
        self.norm.restore_model(models[0])
        self.qkvfc.restore_model(models[1])
        self.norm1.restore_model(models[2])
        self.fc0.restore_model(models[2+1])
        self.fc1.restore_model(models[2*2])
        self.fc_out.restore_model(models[2*2+1])

def create_masks_future(inputs):
    # future
    n, sequence_length = inputs.shape
    input_mask = np.tril(np.ones((sequence_length, sequence_length)))
    # input_mask[input_mask==0] = -np.inf
    input_mask[input_mask==0] = -1e6
    input_mask[input_mask==1] = 0
    return input_mask

if __name__=="__main__":
    batchsize = 10
    embed_dim = 100
    n_patch = 6
    num_h = 2
    mask = create_masks_future(np.random.randn(batchsize, n_patch**2))
    inputs = np.random.randn(batchsize, n_patch**2, embed_dim)
    att = attdecoderblock_layer(embed_dim, num_h)
    
    outputs = np.random.randn(batchsize, n_patch**2, embed_dim)
    for i in range(10000):
        out = att.forward(inputs)
        sum = np.sum((outputs - out) * (outputs - out))
        delta = 2 * (out - outputs) #/ np.prod(outputs.shape)
        partial = att.backward(delta)
        att.update(0.00001)
        att.setzero()
        print(sum)
    k = 0