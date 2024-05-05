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
from gpt.attdecoderblock import attdecoderblock_layer
from gpt.gpt_linear import gpt_linear_layer
from gpt_character.gpt_train_english_char import getdata, create_masks_future
from net.layernorm import layer_norm
from net.fullconnect import fclayer
from classify import classify_layer
from net.flatten import flatten_layer
from copy import deepcopy
from logging import Logger
import json

def predict():
    vocab_size, id2char, char2id, input_texts = getdata()

    all_steps = 3000 - 1000
    batchsize = 63 + 1
    learning_rate = 0.003                         #   batchsize
    embed_dim = 192 ## vocab_size if vocab_size%3==0 else (vocab_size//3) * 3 + 3 # 192
    num_layer = 10 + 1 + 1
    num_h = [3] * num_layer
    context_length = 5

    ADAM = False
    cls_token = True
    float32 = True

    patchemb = Position_Embedding(context_length, vocab_size, embed_dim, adam=ADAM)
    layers = [patchemb]
    
    at0 = attdecoderblock_layer(embed_dim, num_h[0], adam=ADAM, float32=float32)
    # at1 = attdecoderblock_layer(embed_dim, num_h[1], adam=ADAM, float32=float32)
    # at2 = attdecoderblock_layer(embed_dim, num_h[2], adam=ADAM, float32=float32)
    # at3 = attdecoderblock_layer(embed_dim, num_h[3], adam=ADAM, float32=float32)
    # at4 = attdecoderblock_layer(embed_dim, num_h[4], adam=ADAM, float32=float32)
    # at5 = attdecoderblock_layer(embed_dim, num_h[5], adam=ADAM, float32=float32)
    # at6 = attdecoderblock_layer(embed_dim, num_h[6], adam=ADAM, float32=float32)
    # at7 = attdecoderblock_layer(embed_dim, num_h[7], adam=ADAM, float32=float32)
    # at8 = attdecoderblock_layer(embed_dim, num_h[8], adam=ADAM, float32=float32)
    # at9 = attdecoderblock_layer(embed_dim, num_h[9], adam=ADAM, float32=float32)
    # at10 = attdecoderblock_layer(embed_dim, num_h[10], adam=ADAM, float32=float32)
    # at11 = attdecoderblock_layer(embed_dim, num_h[11], adam=ADAM, float32=float32)
    # at12 = attdecoderblock_layer(embed_dim, num_h[12], adam=ADAM, float32=float32)
    # at13 = attdecoderblock_layer(embed_dim, num_h[13], adam=ADAM, float32=float32)

    # layers += [at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11, at12]
    layers += [at0] #, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11]
    # layers += [at0, at1, at2, at3, at4, at5, at6]

    norm = layer_norm(embed_dim, adam=ADAM)
    flatten     = flatten_layer()
    # if not cls_token:
    cll = classify_layer(embed_dim, batchsize, np.sqrt(context_length), vocab_size, cls_token=False, adam=ADAM, relu=False, float32=float32)
    # else:
        # cll = fclayer(embed_dim, vocab_size, True, adam=ADAM, float32=float32)
    layers += [norm, flatten, cll]

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
    
    inputs = 'cdefg'#r'床前明月光，'#满树桃花映日开 床前明月光
    # inputs = r'独客长安觉月遥'
    inputs = [char2id[ci] for ci in inputs]
    inputs = np.array([inputs])
    # inputs = np.random.randint(0, vocab_size, (1, 1))
    output = deepcopy(inputs)
    for ij in range(inputs.shape[1] * 6):
        text = deepcopy(inputs)
        input_mask_fut = create_masks_future(inputs)
        # input_mask_fut[...] = 0
        for l in range(len(layers)):
            if isinstance(layers[l], attdecoderblock_layer):
                inputs = layers[l].forward(inputs, input_mask_fut)
            else:
                inputs = layers[l].forward(inputs)
        inputs = np.reshape(inputs, (-1, vocab_size))[-1, :]
        out = inputs - np.max(inputs, axis = -1, keepdims = True)  # avoid too large in exp 
        softmax = np.exp(out) / np.sum(np.exp(out), axis = -1, keepdims = True)
        
        # rng1 = np.random.default_rng()
        # out = rng1.multinomial(1, softmax, 1).argmax()   ## 投掷一次骰子的

        out = np.argmax(softmax)

        out = np.expand_dims(out, (0))
        output = np.concatenate([output, np.array([out])], axis = -1)
        inputs = output.copy()[:, -context_length:]
        # if output.shape[1] >= context_length:
        #     break

    output = [id2char[int(ij)] for ij in output[0]]
    return ''.join(output)

'''
https://numpy.org/doc/stable/reference/random/generated/numpy.random.multinomial.html#numpy.random.multinomial
Throw a dice 20 times:

np.random.multinomial(20, [1/6.]*6, size=1)
array([[4, 1, 7, 5, 2, 1]]) # random
It landed 4 times on 1, once on 2, etc.

Now, throw the dice 20 times, and 20 times again:

np.random.multinomial(20, [1/6.]*6, size=2)
array([[3, 4, 3, 3, 4, 3], # random
       [2, 4, 3, 4, 0, 7]])
For the first run, we threw 3 times 1, 4 times 2, etc. For the second, we threw 2 times 1, 4 times 2, etc.
'''
if __name__ =="__main__":
    savepath = abspath
    pretrained_model = r'C:\Users\10696\Desktop\Numpy\numpy_transformer\gpt_character\model\gpt_english_last.pkl'
    output = predict()
    output