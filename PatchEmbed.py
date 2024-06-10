import os
from copy import deepcopy
import numpy as np
from gpt.FixedEmbed import Position_Fixed
from net.fullconnect import fclayer
from net.Convolution import convolution_layer
from net.layernorm import layer_norm
from net.Embedding import Embedding_layer

class PatchEmbed_flatten(object):
    def __init__(self, embed_dim, images_shape, n_patch, patchnorm=True) -> None:
        self.embed_dim = embed_dim
        n, c, h, w = images_shape
        self.h_length = h // n_patch
        self.w_length = w // n_patch
        self.n_patch  = n_patch
        self.patchnorm =patchnorm
        self.fullconnect = fclayer(self.h_length * self.w_length * c, self.embed_dim, True)
        if patchnorm:
            self.norm = layer_norm(self.embed_dim)

    def forward(self, images):
        n, c, h, w = images.shape
        h_length = self.h_length
        w_length = self.w_length
        n_patch = self.n_patch
        out = np.zeros((n, n_patch**2, h_length * w_length * c))
        for ni in range(n):
            num_patch = 0
            for i in range(n_patch):
                h_stride = i * h_length
                for j in range(n_patch):
                    w_stride = j * w_length
                    cutimg   = images[ni, :, h_stride:h_stride + h_length, w_stride:w_stride+w_length]
                    out[ni, num_patch, :] = cutimg.flatten()
                    num_patch += 1

        self.inputs = deepcopy(out)
        output = self.fullconnect.forward(out)

        if self.patchnorm:
            output = self.norm.forward(output)
        return output

    def backward(self, delta):
        if self.patchnorm:
            delta = self.norm.backward(delta)
        input_delta = self.fullconnect.backward(delta, self.inputs)
        return input_delta

    def update(self, lr):
        if self.patchnorm:
            self.norm.update(lr)
        self.fullconnect.update(lr)
    
    def setzero(self):
        if self.patchnorm:
            self.norm.setzero()
        self.fullconnect.setzero()

    def save_model(self):
        model = []
        if self.patchnorm:
            model.append(self.norm.save_model())
        model.append(self.fullconnect.save_model())
        return model

    def restore_model(self, models):
        if self.patchnorm:
            self.norm.restore_model(models[0])
        self.fullconnect.restore_model(models[-1])

class PatchEmbed_convolution(object):
    def __init__(self, embed_dim, images_shape, n_patch, patchnorm=True) -> None:
        self.embed_dim = embed_dim
        n, c, h, w = images_shape
        self.batch = n
        self.h_length = h // n_patch
        self.w_length = w // n_patch
        self.n_patch  = n_patch
        self.patchnorm =patchnorm
        self.convolution = convolution_layer(c, embed_dim, kernel_size=self.w_length, stride=self.w_length)
        if patchnorm:
            self.norm = layer_norm(self.embed_dim)

    def forward(self, images):
        out = self.convolution.forward(images)
        out = np.transpose(out, (0, 2, 3, 1))
        out = np.reshape(out, (self.batch, -1, self.embed_dim)) #n, ph*pw, ed
        if self.patchnorm:
            output = self.norm.forward(out)
        return out

    def backward(self, delta):
        if self.patchnorm:
            delta = self.norm.backward(delta)
        delta = np.reshape(delta, (self.batch, self.n_patch, self.n_patch, self.embed_dim))
        delta = np.transpose(delta, (0, 3, 1, 2))
        input_delta = self.convolution.backward(delta)
        return input_delta

    def update(self, lr):
        if self.patchnorm:
            self.norm.update(lr)
        self.convolution.update(lr)

    def setzero(self):
        if self.patchnorm:
            self.norm.setzero()
        self.convolution.setzero()

    def save_model(self):
        model = []
        if self.patchnorm:
            model.append(self.norm.save_model())
        model.append(self.convolution.save_model())
        return model

    def restore_model(self, models):
        if self.patchnorm:
            self.norm.restore_model(models[0])
        self.convolution.restore_model(models[-1])

class Position_Embedding(Embedding_layer):
    def __init__(self, context_length, vocab_size,  embed_dim, adam = False, float32=False, float16 = False):
        self.context_length = context_length
        self.text_embedding = Embedding_layer(vocab_size, embedding_dim = embed_dim, adam = adam, float32=float32, float16=float16)
        # self.pos_embedding  = Position_Fixed(context_length, embed_dim)
        self.pos_embedding = Embedding_layer(context_length, embedding_dim = embed_dim, adam = adam, float32=float32, float16=float16)
        self.adam = adam

    def forward(self, inputs):
        n, sequence_length = inputs.shape
        te = self.text_embedding.forward(inputs) # n, sequence_length, embed_dim
        po = self.pos_embedding.forward(np.arange(sequence_length)) # sequence_length, embed_dim
        if len(po.shape)!=3:
            po = np.expand_dims(po, 0)
        return te + po

    def backward(self, delta):
        input_delta = self.text_embedding.backward(delta)
        delta = np.sum(delta, axis = 0, keepdims=False)
        _ = self.pos_embedding.backward(delta)
        return input_delta

    def update(self, lr):
        self.text_embedding.update(lr)
        self.pos_embedding.update(lr)

    def setzero(self):
        self.text_embedding.setzero()
        self.pos_embedding.setzero()

    def save_model(self):
        return [self.text_embedding.save_model(), self.pos_embedding.save_model()]

    def restore_model(self, models):
        self.text_embedding.restore_model(models[0])
        self.pos_embedding.restore_model(models[1])

if __name__=="__main__":
    batchsize = 1
    lr = 0.0001
    embed_dim = 30
    images_shape = (batchsize, 3, 30-2, 30-2)
    n_patch = 7
    inputs = np.random.randn(batchsize, 3, 30-2, 30-2)
    patchemb = PatchEmbed_flatten(embed_dim, images_shape, n_patch)
    # patchemb = PatchEmbed_convolution(embed_dim, images_shape, n_patch)
    
    context_length = 100
    vocab_size = 300
    embed_dim = 200
    posiemb = Position_Embedding(context_length, vocab_size, embed_dim)

    outputs = np.random.randn(batchsize, context_length, embed_dim)
    inputs = np.random.randint(0, vocab_size, (batchsize, context_length))
    # inputs = np.arange(batchsize * context_length).reshape((batchsize, context_length))
    for i in range(30000):
        out = posiemb.forward(inputs)
        sum = np.sum((outputs - out) * (outputs - out))
        delta = 2 * (out - outputs)
        _ = posiemb.backward(delta)
        posiemb.update(lr = 0.001)
        posiemb.setzero()
        print(sum)