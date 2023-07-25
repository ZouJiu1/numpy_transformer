import os
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)

from net.loss import cross_entropy_loss
import numpy as np
import pickle
from net.layernorm import layer_norm
from PatchEmbed import Position_Embedding
from attention import attention_layer
from gpt.gpt_linear import gpt_linear_layer
from gpt.gpt_train import getdata
from net.layernorm import layer_norm

from copy import deepcopy
import json

def predict():
    vocab_size, id2char, char2id, input_texts = getdata()

    epoch = 30
    batchsize = 1 #100
    lr = 0.01
    embed_dim = 90
    n_patch = 7
    num_layer = 6
    num_h = [3] * num_layer #[3, 6, 12, 3, 6, 12]
    context_length = 26 - 2 #300

    patchemb = Position_Embedding(context_length, vocab_size, embed_dim)
    layers = [patchemb]
    layers += [attention_layer(embed_dim, num_h[i]) for i in range(num_layer)]
    norm = layer_norm(embed_dim)
    cll = gpt_linear_layer(embed_dim, batchsize, n_patch, vocab_size)
    layers += [norm, cll]
    
    inputs = np.random.randint(0, vocab_size, (1, 1))
    output = deepcopy(inputs)
    for ij in range(context_length - 1):
        text = deepcopy(inputs)
        for l in range(len(layers)):
            inputs = layers[l].forward(inputs)
        inputs = np.reshape(inputs, (-1, vocab_size))
        out = inputs - np.max(inputs, axis = -1)[..., np.newaxis]   # avoid too large in exp 
        softmax = np.exp(out) / np.sum(np.exp(out), axis = -1)[:, np.newaxis]
        out = np.argmax(softmax, axis = -1)
        out = np.expand_dims(out, (-1))
        inputs = out.copy()
        output = np.concatenate([output, out], axis = -1)

    output = [id2char[int(ij)] for ij in output[0]]
    return ''.join(output[:200])

if __name__ =="__main__":
    savepath = abspath
    pretrained_model = r'C:\Users\10696\Desktop\access\numpy_transformer\gpt\model'
    output = predict()
    output