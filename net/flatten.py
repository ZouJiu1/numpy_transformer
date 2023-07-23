import numpy as np

class flatten_layer():
    def forward(self, inputs):
        self.shape = inputs.shape
        return np.stack([i.flatten() for i in inputs])
    
    def backward(self, delta, lr = ''):
        return np.reshape(delta, self.shape)
    
    def update(self, lr = ''):
        pass

    def setzero(self):
        pass