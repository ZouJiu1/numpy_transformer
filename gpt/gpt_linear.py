import numpy as np
from net.layernorm import layer_norm
from net.fullconnect import fclayer
from net.activation import ReLU

class gpt_linear_layer():
    def __init__(self, embed_dim, num_classes):
        self.embed_dim = embed_dim
        # self.n_patch = n_patch
        # self.fc0 = fclayer(self.embed_dim, self.embed_dim, True)
        # self.relu    = ReLU()
        self.fc1 = fclayer(self.embed_dim, num_classes, True)

    def forward(self, inputs):
        self.inputs = inputs
        # out0 = self.fc0.forward(inputs)
        # self.out1 = self.relu.forward(out0)
        out = self.fc1.forward(inputs)
        return out

    def backward(self, delta):
        delta = self.fc1.backward(delta, self.inputs)
        # delta = self.relu.backward(delta)
        # delta = self.fc0.backward(delta, self.inputs)
        return delta

    def update(self, lr):
        self.fc1.update(lr)
        # self.fc0.update(lr)

    def setzero(self):
        # self.fc0.setzero()
        self.fc1.setzero()

    def save_model(self):
        # return [self.fc0.save_model(), self.fc1.save_model()]
        return [self.fc1.save_model()]

    def restore_model(self, models):
        # self.fc0.restore_model(models[0])
        self.fc1.restore_model(models[0])

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