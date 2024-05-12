import os
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)

from net.loss import cross_entropy_loss
import numpy as np
import pickle
import torch
from copy import deepcopy
import json
from net.layernorm import layer_norm
from attention import attention_layer
from PatchEmbed import Position_Embedding
from attdecoderblock import attdecoderblock_layer
from gpt.gpt_linear import gpt_linear_layer
from gpt.gpt_train_english import getdata, create_masks_future, getinputs
from net.layernorm import layer_norm
from net.fullconnect import fclayer
from classify import classify_layer
from net.flatten import flatten_layer

def predict():
    vocab_size, id2char, char2id, input_texts = getdata()

    all_steps = 6000 - 1000
    batchsize = 63 + 1
    learning_rate = 0.0003                         #   batchsize
    embed_dim = 192*2  # vocab_size if vocab_size%3==0 else (vocab_size//3) * 3 + 3 # 192
    num_layer = 10 + 1 + 1
    num_h = [3*2] * num_layer
    context_length = 256

    inputid = [0]
    character = list(r''',.? !;、，。？！；''')
    for i in range(1, len(input_texts)):
        if input_texts[i - 1] in character:
            inputid.append(i)
    inputid = np.array(inputid, dtype = np.int64)
    while inputid[-1] + context_length + 1 > len(input_texts):
        inputid = inputid[:-1]

    ADAM = False
    cls_token = True
    single_word = False
    float32 = True
    
    patchemb = Position_Embedding(context_length, vocab_size, embed_dim, adam=ADAM)
    layers = [patchemb]
    
    at0 = attdecoderblock_layer(embed_dim, num_h[0], adam=ADAM)
    at1 = attdecoderblock_layer(embed_dim, num_h[1], adam=ADAM)
    at2 = attdecoderblock_layer(embed_dim, num_h[2], adam=ADAM)
    at3 = attdecoderblock_layer(embed_dim, num_h[3], adam=ADAM)
    at4 = attdecoderblock_layer(embed_dim, num_h[4], adam=ADAM)
    at5 = attdecoderblock_layer(embed_dim, num_h[5], adam=ADAM)
    at6 = attdecoderblock_layer(embed_dim, num_h[6], adam=ADAM)
    at7 = attdecoderblock_layer(embed_dim, num_h[7], adam=ADAM)
    at8 = attdecoderblock_layer(embed_dim, num_h[8], adam=ADAM)
    at9 = attdecoderblock_layer(embed_dim, num_h[9], adam=ADAM)
    at10 = attdecoderblock_layer(embed_dim, num_h[10], adam=ADAM)
    at11 = attdecoderblock_layer(embed_dim, num_h[11], adam=ADAM)
    # at12 = attdecoderblock_layer(embed_dim, num_h[12], adam=ADAM)
    # at13 = attdecoderblock_layer(embed_dim, num_h[13], adam=ADAM)

    # layers += [at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11, at12]
    layers += [at0]#, at1, at2, at3, at4, at5]#, at6, at7, at8, at9, at10, at11]

    norm = layer_norm(embed_dim, adam=ADAM)
    flatten     = flatten_layer()
    # if not cls_token:
    if not single_word:
        cll = classify_layer(embed_dim, batchsize, 1, vocab_size, cls_token=False, adam=ADAM, relu=False, float32=float32)
        layers += [norm, cll]
    else:
        cll = classify_layer(embed_dim, batchsize, np.sqrt(context_length), vocab_size, cls_token=False, adam=ADAM, relu=False, float32=float32)
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

    num_sample = 2
    example = []
    start_length = 60
    for _ in range(num_sample):
        inputs, input_mask, label_single, id_start = getinputs(inputid, start_length, 1, input_texts, char2id, id2char)
        out = list(inputs.reshape(-1))
        for ij in range(context_length * 2):
            input_mask = create_masks_future(inputs)
            for l in range(len(layers)):
                if isinstance(layers[l], attdecoderblock_layer):
                    inputs = layers[l].forward(inputs, input_mask)
                else:
                    inputs = layers[l].forward(inputs)
            inputs = np.reshape(inputs, (-1, vocab_size))
            p_shift = inputs - np.max(inputs, axis = -1)[..., np.newaxis]   # avoid too large in exp 
            predict = np.exp(p_shift) / np.sum(np.exp(p_shift), axis = -1)[:, np.newaxis]
            # p = np.argmax(predict, axis=-1).reshape(-1)
            p = torch.multinomial(torch.from_numpy(predict), 1).cpu().numpy().flatten()
            out.append(p[-1])
            inputs = np.array(out[-context_length:])[np.newaxis, :]
        output = ''.join([id2char[int(ij)] for ij in out]) + "\n"
        example.append(output)
    character = list(r''',.? !;、，。？！；''')
    file_path = os.path.join(abspath, 'gpt', 'model', 'example_english.txt')
    file = open(file_path, "w")
    fileshow = ""
    for i, sample in enumerate(example):
        file.write(f"################ SAMPLE { i + 1 } ################\n SAMPLE {i+1} INPUT: ")
        ln = 0
        for j in range(len(sample)):
            if j <= start_length - 2:
                file.write(sample[j])
                fileshow += sample[j]
                continue
            elif j == start_length-1:
                file.write(sample[i] + '\n\n\n Model Output: ')
                fileshow += sample[i] + '\n\n\n Model Output: '
                continue
            if ln < 60 and ln > 50 and sample[j] == ' ':
                file.write('\n ')
                fileshow += "\n "
                ln = 0
            file.write(sample[j])
            fileshow += sample[j]
            ln += 1
        # file.write("".join(sample))
        file.write("\n\n\n")

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
    pretrained_model = os.path.join(abspath, r'gpt', r'model', r'gpt_english_last.pkl')
    output = predict()