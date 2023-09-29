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
from gpt.attdecoderblock import attdecoderblock_layer
from gpt.gpt_linear import gpt_linear_layer
from gpt_character.gpt_train_english_char import getdata
from net.layernorm import layer_norm
from net.fullconnect import fclayer
from copy import deepcopy
import json

def predict():
    vocab_size, id2char, char2id, input_texts = getdata()

    batchsize = 1
    embed_dim = 27
    num_layer = 1
    num_h = [3] * num_layer #[3, 6, 12, 3, 6, 12]
    context_length = 10

    patchemb = Position_Embedding(context_length, vocab_size, embed_dim)
    layers = [patchemb]
    
    at0 = attdecoderblock_layer(embed_dim, num_h[0])

    layers += [at0]
    norm = layer_norm(embed_dim)
    cll = fclayer(embed_dim, vocab_size, True)
    layers += [norm, cll]

    if os.path.exists(pretrained_model):
        with open(pretrained_model, 'rb') as obj:
            models = pickle.load(obj)
        cnt = 0
        for l in layers:
            k = dir(l)
            if 'restore_model' in k and 'save_model' in k:
                l.restore_model(models[cnt])
                cnt += 1
        del models
    else:
        exit(-1)

    ret = []
    for num in range(10):
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
        output = [output[0]] + [": "] + output[1:]
        ret.append(''.join(output[:200]))
    return ret

if __name__ =="__main__":
    savepath = abspath
    pretrained_model = r'C:\Users\10696\Desktop\access\numpy_transformer\gpt_character\model\gpt_english_epoch_10000_loss_0.09217.pkl'
    output = predict()
    output