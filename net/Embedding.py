import numpy as np
import torch
from torch import nn

def torch_compare_Embedding(num_embeddings, embedding_dim, delta, inputs, params):
    network = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim).requires_grad_(True)
    network.double()
    for i in network.parameters():
        i.data = torch.tensor(params, dtype=torch.double)
        i.retain_grad = True
    inputs = torch.tensor(inputs, requires_grad=True, dtype=torch.double)
    inputs = inputs.long()
    output = network(inputs)
    delta = torch.tensor(delta)
    output.backward(delta)
    # sum = torch.sum(output) # make sure the gradient is 1
    # kk = sum.backward()
    grad_params = 0
    for i in network.parameters():
        grad_params = i.grad
        params_aft  = i.data
    output.retain_grad()
    return output, grad_params, params_aft

class Embedding_layer(object):
    def __init__(self, num_embeddings, embedding_dim, params=[]):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if list(params)!=[]:
            self.params = params
        else:
            self.params = np.random.normal(0, 1, (num_embeddings, embedding_dim))
        self.delta = np.zeros_like(self.params)

    def forward(self, inputs):
        outshape = list(inputs.shape) + [self.embedding_dim]
        self.flatten = inputs.flatten()
        output = self.params[self.flatten, :]
        output = np.reshape(output, outshape)
        return output

    def backward(self, delta, flatten=[]):

        if len(flatten)!=0:
            isk = tuple([-1, self.delta.shape[-1]])
            delta = np.reshape(delta, isk)
            for i in range(len(flatten)):
                self.delta[flatten[i], :] += delta[i]
        else:
            isk = tuple([-1, self.delta.shape[-1]])
            delta = np.reshape(delta, isk)
            for i in range(len(self.flatten)):
                self.delta[self.flatten[i], :] += delta[i]

        return self.delta

    def setzero(self):
        self.delta[...] = 0

    def update(self, lr=1e-10):
        self.params -= lr * self.delta

    def save_model(self):
        return [self.params.astype(np.float32)]

    def restore_model(self, models):
        self.params = models[0]

def train_single():
    num_embeddings = 1000
    embedding_dim = 100
    params = np.random.normal(0, 1, (num_embeddings, embedding_dim)) * 100

    Embedding = Embedding_layer(num_embeddings, embedding_dim, params.copy())
    outputs = np.random.rand(num_embeddings, embedding_dim)
    for i in range(30000):
        inputs = np.arange(1000).astype(np.int32)
        out = Embedding.forward(inputs)
        sum = np.sum((outputs - out) * (outputs - out))
        delta = 2 * (out - outputs)
        _ = Embedding.backward(delta.copy().reshape(-1, embedding_dim), Embedding.flatten)
        Embedding.update(lr = 0.001)
        Embedding.setzero()
        print(sum)

if __name__=="__main__":
    #https://discuss.pytorch.org/t/how-nn-embedding-trained/32533/11
    train_single()

    num_embeddings = 1000
    embedding_dim = 100
    inputs = np.arange(30).reshape((10, 3)).astype(np.int32)
    params = np.random.normal(0, 1, (num_embeddings, embedding_dim)) * 100

    Embedding = Embedding_layer(num_embeddings, embedding_dim, params.copy())
    output = Embedding.forward(inputs)
    k = output == params.copy()[np.arange(30), :].reshape((list(inputs.shape) + [embedding_dim]))
    assert k.all()
    delta_l = np.ones((list(inputs.shape) + [embedding_dim])).astype(np.float64)
    delta = Embedding.backward(delta_l.copy().reshape(-1, embedding_dim), Embedding.flatten)
    Embedding.update(lr = 1)
    # Embedding.setzero()

    # num_embeddings = 10
    # embedding_dim = 2
    # inputs = np.arange(6).reshape((2, 3)).astype(np.int32)
    # params = np.random.normal(0, 1, (num_embeddings, embedding_dim)) * 100

    # Embedding = Embedding_layer(num_embeddings, embedding_dim, params.copy())
    # output = Embedding.forward(inputs)
    # k = output == params.copy()[np.arange(6), :].reshape((list(inputs.shape) + [embedding_dim]))
    # assert k.all()
    # delta_l = np.ones((list(inputs.shape) + [embedding_dim])).astype(np.float64)
    # delta = Embedding.backward(delta_l.copy().reshape(-1, embedding_dim))
    # Embedding.update(lr = 1)
    # # Embedding.setzero()

    output_torch, grad_params, params_aft = torch_compare_Embedding(num_embeddings, embedding_dim, delta_l, inputs, params.copy())
    assert np.mean(np.abs(output - output_torch.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(output - output_torch.cpu().detach().numpy()))
    assert np.mean(np.abs(Embedding.delta - grad_params.cpu().detach().numpy())) < 1e-6, np.mean(np.abs(Embedding.delta - grad_params.cpu().detach().numpy()))
