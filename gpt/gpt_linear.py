import numpy as np
from net.layernorm import layer_norm
from net.fullconnect import fclayer
from net.activation import ReLU

class gpt_linear_layer():
    def __init__(self, embed_dim, num_classes, expand = 3):
        self.embed_dim = embed_dim
        # self.n_patch = n_patch
        self.fc0 = fclayer(self.embed_dim, self.embed_dim * expand, True)
        self.norm    = layer_norm(self.embed_dim * expand)
        self.fc1 = fclayer(self.embed_dim * expand, num_classes, True)

    def forward(self, inputs):
        self.inputs = inputs
        out0 = self.fc0.forward(inputs)
        self.out1 = self.norm.forward(out0)
        out = self.fc1.forward(self.out1)
        return out

    def backward(self, delta):
        delta = self.fc1.backward(delta, self.out1)
        delta = self.norm.backward(delta)
        delta = self.fc0.backward(delta, self.inputs)
        return delta

    def update(self, lr):
        self.fc1.update(lr)
        self.fc0.update(lr)
        self.norm.update(lr)

    def setzero(self):
        self.fc0.setzero()
        self.fc1.setzero()
        self.norm.setzero()

    def save_model(self):
        return [self.fc0.save_model(), self.fc1.save_model(), self.norm.save_model()]
        # return [self.fc1.save_model()]

    def restore_model(self, models):
        self.fc0.restore_model(models[0])
        self.fc1.restore_model(models[1])
        self.norm.restore_model(models[2])

if __name__=="__main__":
    batchsize = 10
    embed_dim = 30
    n_patch = 7
    num_classes = 10
    cls_token = True
    if not cls_token:
        inputs = np.random.randn(batchsize, (n_patch**2)*embed_dim)
    else:
        inputs = np.random.randn(batchsize, embed_dim)
    posit = classify_layer(embed_dim, batchsize, n_patch, num_classes, cls_token)
    
    outputs = np.random.randn(batchsize, num_classes)
    for i in range(10000):
        out = posit.forward(inputs)
        sum = np.sum((outputs - out) * (outputs - out))
        delta = 2 * (out - outputs) #/ np.prod(outputs.shape)
        partial = posit.backward(delta)
        posit.update(0.00001)
        posit.setzero()
        print(sum)
    k = 0
    # output = posit.forward(inputs)

    # delta = np.ones_like(output)
    # input_delta = posit.backward(delta)
    # k = 0