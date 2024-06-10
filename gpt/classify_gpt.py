import numpy as np
from net.layernorm import layer_norm
from net.fullconnect import fclayer
from net.activation import ReLU

class classify_layer():
    def __init__(self, embed_dim, batch, n_patch, num_classes, cls_token = True, adam=False, relu=True, float32=False, float16=False):
        self.batch = batch
        self.embed_dim = embed_dim
        self.n_patch = n_patch
        self.cls_token = cls_token
        if cls_token:
            self.fc0 = fclayer(self.embed_dim, self.embed_dim, True, adam=adam, float32=float32, float16=float16)
        else:
            self.fc0 = fclayer(self.embed_dim * int(n_patch**2), self.embed_dim, True, adam=adam, float32=float32, float16=float16)
        self.relu    = ReLU()
        self.fc1 = fclayer(self.embed_dim, num_classes, True, adam=adam, float32=float32, float16=float16)
        self.reluact = relu

    def forward(self, inputs):
        self.inputs = inputs
        out0 = self.fc0.forward(inputs)
        if self.reluact:
            self.out1 = self.relu.forward(out0)
        else:
            self.out1 = out0.copy()
        out = self.fc1.forward(self.out1)
        return out

    def backward(self, delta):
        delta = self.fc1.backward(delta, self.out1)
        if self.reluact:
            delta = self.relu.backward(delta)
        delta = self.fc0.backward(delta, self.inputs)
        if self.cls_token:
            zeros = np.zeros((self.batch, self.n_patch**2, self.embed_dim))
            zeros[:, 0] = delta
            return zeros
        else:
            return delta

    def update(self, lr):
        self.fc1.update(lr)
        self.fc0.update(lr)

    def setzero(self):
        self.fc0.setzero()
        self.fc1.setzero()

    def save_model(self):
        return [self.fc0.save_model(), self.fc1.save_model()]

    def restore_model(self, models):
        self.fc0.restore_model(models[0])
        self.fc1.restore_model(models[1])

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