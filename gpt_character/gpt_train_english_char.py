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
from net.layernorm import layer_norm
from net.fullconnect import fclayer

from copy import deepcopy
import json

# https://en.wikipedia.org/wiki/AlexNet
# https://pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html#alexnet
# https://github.com/l5shi/Image-Recognition-on-MNIST-dataset/blob/master/AlexNet.ipynb

def getdata():
    dataset = os.path.join(abspath, 'dataset')
    os.makedirs(dataset, exist_ok=True)
    id2char_char2id = os.path.join(abspath, 'dataset', r"alphabet.json")
    # inpath = os.path.join(abspath, 'dataset', r"alphabet.txt")
    # with open(inpath, 'r', encoding='utf-8') as obj:
    #     readcontent = obj.read()
    readcontent = \
    '''
abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz  abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz 
abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz 
abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz 
abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz 
abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz 
abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz 
abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz 
abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz 
abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz 
abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz 
abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz 
abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz 
abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz 
    '''
    kk = [i if i!='\n' else " " for i in readcontent]
    unique = np.unique(kk)
    length = len(unique)
    id2char = {i:char for i, char in enumerate(unique)}
    char2id = {char:i for i, char in enumerate(unique)}
    if not os.path.exists(id2char_char2id):
        with open(id2char_char2id, 'w', encoding='utf-8') as obj:
            json.dump({"id2char":id2char, 'char2id':char2id}, obj, indent=2, separators=(",", ":"), ensure_ascii=False)
    else:
        with open(id2char_char2id, 'r', encoding='utf-8') as obj:
            jsonfile = json.load(obj)
        id2chark = jsonfile["id2char"]
        char2id = jsonfile["char2id"]
        length = len(id2char)
        id2char = {}
        for key, value in id2chark.items():
            id2char[int(key)] = value
    return length, id2char, char2id, kk

def create_masks(input_mask):
    
    #pad
    input_mask = np.array(input_mask)
    n, sequence_length = input_mask.shape
    k1 = input_mask[:, None, :]
    k2 = np.ones_like(input_mask)[:, :, None]
    k = k1 * k2
    k = (1.0 - k) * (-1e6)

    #future
    input_mask = np.tril(np.ones_like(k))
    input_mask[input_mask==0] = -np.inf
    input_mask[input_mask==1] = 0
    return input_mask

def transformer_image_train(num_classes):
    vocab_size, id2char, char2id, input_texts = getdata()

    epoch = 10000
    batchsize = 10
    learning_rate = 0.0002
    # embed_dim = 210
    embed_dim = 27
    # num_layer = 12
    num_layer = 1
    num_h = [3] * num_layer
    # context_length = 260
    context_length = 10

    logfile = os.path.join(logdir, 'log_gpt_alphabet.txt')
    fpwrite = open(logfile, 'w', encoding='utf-8')

    patchemb = Position_Embedding(context_length, vocab_size, embed_dim)
    layers = [patchemb]
    
    at0 = attdecoderblock_layer(embed_dim, num_h[0])
    # at1 = attdecoderblock_layer(embed_dim, num_h[1])
    # at2 = attdecoderblock_layer(embed_dim, num_h[2])
    # at3 = attdecoderblock_layer(embed_dim, num_h[3])
    # at4 = attdecoderblock_layer(embed_dim, num_h[4])
    # at5 = attdecoderblock_layer(embed_dim, num_h[5])
    # at6 = attdecoderblock_layer(embed_dim, num_h[6])
    # at7 = attdecoderblock_layer(embed_dim, num_h[7])
    # at8 = attdecoderblock_layer(embed_dim, num_h[8])
    # at9 = attdecoderblock_layer(embed_dim, num_h[9])
    # at10 = attdecoderblock_layer(embed_dim, num_h[10])
    # at11 = attdecoderblock_layer(embed_dim, num_h[11])
    # at12 = attdecoderblock_layer(embed_dim, num_h[12])
    # at13 = attdecoderblock_layer(embed_dim, num_h[13])
    layers += [at0]
    # layers += [at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11, at12, at13]

    norm = layer_norm(embed_dim)
    cll = fclayer(embed_dim, vocab_size, True)
    layers += [norm, cll]

    datapath = os.path.join(abspath, 'dataset')
    os.makedirs(datapath, exist_ok=True)
    modelpath = os.path.join(abspath, 'gpt_character', 'model')
    os.makedirs(modelpath, exist_ok=True)

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

    alliter = 0
    pre = 0
    lr = learning_rate
    start_epoch = 1
    try:
        if os.path.exists(pretrained_model):
            start_epoch = int(pretrained_model.split(os.sep)[-1].split("_")[3]) + 1
    except:
        start_epoch = 1
    for i in range(start_epoch, epoch + 1):
        meanloss = 0
        # if i!=0:
            # lr = lr * dot
        if i==20*epoch//30:
            lr = learning_rate * 0.1
        elif i==26*epoch//30:
            lr = learning_rate * 0.1 * 0.1
        number = 0
        jk = 0
        pre_col = []
        while True:
            # if alliter < 99:
            #     lr = 0.0001
            # elif alliter == 100:
            #     lr = learning_rate
            jk += 1
            inputs = []
            label = []
            input_mask = []
            status = -6
            for ij in range(batchsize):
                if number + context_length+1 >= len(input_texts):
                    status = 6
                    break
                tmp = [char2id[input_texts[ci + number]] for ci in range(context_length+1)]
                inputchar = "".join([id2char[ci]  for ci in tmp])
                # input_mask.append([1 for ci in range(context_length-1)])
                # input_mask[-1].extend([0])
                inputs.append(tmp[:-1])
                label.append(tmp[1:])
                number += context_length + 1
            if status > 0:
                break
            alliter += 1
            inputs = np.array(inputs)
            if len(input_mask)==0:
                input_mask = np.ones_like(inputs)
            
            input_mask = create_masks(input_mask)
            label_single = np.array(label).reshape(-1)
            for l in range(len(layers)):
                if isinstance(layers[l], attdecoderblock_layer):
                    inputs = layers[l].forward(inputs, input_mask)
                else:
                    inputs = layers[l].forward(inputs)

            ishape = inputs.shape
            inputs = np.reshape(inputs, (-1, vocab_size))
            labels = np.zeros_like(inputs)
            labels[np.arange(len(inputs)), label_single] = 1
            # k = np.sum(labels, axis = -1)
            loss, delta, predict = cross_entropy_loss(inputs, labels)
            if loss < 0.06:
                k = 0
            delta = np.reshape(delta, ishape)
            meanloss += loss
            p = np.argmax(predict, axis=-1)
            precision = np.sum(label_single==p) / len(label_single)
            pre_col.append(precision)

            fpwrite.write("epoch:{}, lr: {:.6f}, loss: {:.6f}, iters: {}, precision: {:.6f}\n". \
                    format(i, lr, loss, str(jk) +"_"+ str(alliter), precision))
            fpwrite.flush()
            for l in range(len(layers)-1, -1, -1):
                delta = layers[l].backward(delta)
                layers[l].update(lr)
                layers[l].setzero()
        meanloss /= jk
        premean = np.mean(pre_col)
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
        fpwrite.write("epoch: {},  {}\n\n".format(i, ''.join(output[:200])))
        fpwrite.flush()

    # savemodel
    allmodel = []
    for l in layers:
        k = dir(l)
        if 'restore_model' in k and 'save_model' in k:
            allmodel.append(l.save_model())
    name = "gpt_english_epoch_"+str(i)+"_loss_"+str(round(meanloss, 6))+".pkl"

    with open(os.path.join(modelpath, name), 'wb') as obj:
        pickle.dump(allmodel, obj)
    fpwrite.close()

if __name__ =="__main__":
    savepath = abspath
    pretrained_model = r''
    logdir = os.path.join(savepath, 'gpt', 'log')
    os.makedirs(logdir, exist_ok=True)
    transformer_image_train(10)

'''
https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
https://github.com/UdbhavPrasad072300/Transformer-Implementations/blob/main/notebooks/MNIST%20Classification%20-%20ViT.ipynb
https://github.com/s-chh/PyTorch-Vision-Transformer-ViT-MNIST/tree/main
https://itp.uni-frankfurt.de/~gros/StudentProjects/WS22_23_VisualTransformer/
https://jamesmccaffrey.wordpress.com/2023/01/10/a-naive-transformer-architecture-for-mnist-classification-using-pytorch/
https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
https://github.com/BrianPulfer/PapersReimplementations/blob/main/vit/vit_torch.py
https://github.com/microsoft/Swin-Transformer
https://huggingface.co/docs/transformers/v4.27.0/model_doc/vit
'''
    