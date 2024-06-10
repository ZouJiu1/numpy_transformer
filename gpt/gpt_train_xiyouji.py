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
from attdecoderblock import attdecoderblock_layer
from net.layernorm import layer_norm
from net.fullconnect import fclayer
import re
from gpt.classify_gpt import classify_layer
from net.flatten import flatten_layer
from copy import deepcopy
import json

# https://en.wikipedia.org/wiki/AlexNet
# https://pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html#alexnet
# https://github.com/l5shi/Image-Recognition-on-MNIST-dataset/blob/master/AlexNet.ipynb

def getdata():
    dataset = os.path.join(abspath, 'dataset')
    os.makedirs(dataset, exist_ok=True)
    id2char_char2id = os.path.join(abspath, 'dataset', r"xiyouji_wuchengen.json")
    inpath = os.path.join(abspath, 'dataset', r"xiyouji_wuchengen.txt")
    with open(inpath, 'r', encoding='utf-8') as obj:
        readcontent = obj.read().lower()
    # kk = [i if i!='\n' else " " for i in readcontent]
    # kk = [i if i not in al else " " for i in readcontent]
    kk = []
    chinese_character = list(r'''，。：？《》{}【】、‘；’“”：、·~！@￥%……&*（）-=□''')
    character = [',', '.', " ", '?', '"', ";", '{', "}", ")", "(", ">", "<", '\n'] + chinese_character

    character = ['\n']

    for i in readcontent:
        i = i.lower()
        if i in character:
            if kk[-1]!=" ":
                kk.append(" ")
        else:
            kk.append(i)

    kk = "".join(kk)
    kk = re.sub(r' +', " ", kk)
    kk = re.sub(r'\n+', " ", kk)
    # kk = re.sub(r',', "", kk)
    # kk = re.sub(r'\.', "", kk)
    # kk = re.sub(r'，', "", kk)
    # kk = re.sub(r'。', "", kk)
    # kk = re.sub(r'？', "", kk)
    # kk = re.sub(r'\?', "", kk)
    # kk = re.sub(r'、', "", kk)
    # kk = re.sub(r'》', "", kk)
    # kk = re.sub(r'《', "", kk)
    # kk = re.sub(r'；', "", kk)
    # kk = re.sub(r'！', "", kk)
    while kk[-1]==' ':
        kk = kk[:-1]
    kk = list(kk)
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

def create_masks_future(inputs):
    # future
    n, sequence_length = inputs.shape
    input_mask = np.tril(np.ones((sequence_length, sequence_length)))
    input_mask[input_mask==0] = -np.inf
    # input_mask[input_mask==0] = -1e6
    input_mask[input_mask==1] = 0
    return input_mask

def create_masks_pad(input_mask):
    # pad
    input_mask = np.array(input_mask)
    n, sequence_length = input_mask.shape
    k1 = input_mask[:, None, :]
    k2 = np.ones_like(input_mask)[:, :, None]
    k = k1 * k2
    k = (1.0 - k)
    k[k==1.0] = -np.inf
    return k

# k = create_masks_pad([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0]])

def getinputs(inputid, context_length, batchsize, input_texts, char2id, id2char):
    inputs = []
    label = []
    input_mask = []
    # id_start = np.random.randint(0, len(input_texts) - context_length -1, (batchsize))
    id_start = np.random.choice(inputid, batchsize, replace=False)

    markedchar = [',', '. ']
    for id in id_start:
        tmp = [char2id[ci] for ci in input_texts[id : id + context_length + 1]]
        inputchar = "".join([id2char[ci]  for ci in tmp])
        # input_mask.append([1 for ci in range(context_length-1)])
        # input_mask[-1].extend([0])
        inputs.append(tmp[:-1])
        label.append(tmp[1:])
    inputs = np.array(inputs)
    if len(input_mask)==0:
        input_mask = np.ones_like(inputs)
            
    input_mask_fut = create_masks_future(inputs)
    # input_mask_pad = create_masks_pad(input_mask)
    input_mask = input_mask_fut
    # if not single_word:
    label_single = np.array(label)   #[:, -1, np.newaxis] #.reshape(-1)
    # else:
    #     label_single = np.array(label)[:, -1, np.newaxis] #.reshape(-1)
    
    return inputs, input_mask, label_single, id_start

def transformer_image_train():
    global savename
    vocab_size, id2char, char2id, input_texts = getdata()
    
    inputid = [0]
    character = list(r''',.? !;、，。？！；''')
    for i in range(1, len(input_texts)):
        if input_texts[i - 1] in character:
            inputid.append(i)
    inputid = np.array(inputid, dtype = np.int64)
    # all_steps = 6000 - 1000 + 10000
    # batchsize = 10 #63+1
    # learning_rate = 0.0003 #/ batchsize
    # embed_dim = vocab_size if vocab_size%3==0 else (vocab_size + 2)//3 # 192
    # num_layer = 10 + 1 + 1
    # num_h = [3] * num_layer
    # context_length = 30 # 260 - 2*2
    
    epoch = 10
    batchsize = 64
    learning_rate = 0.0003      #  batchsize
    embed_dim = 192*2           #  vocab_size//3
    num_layer = 10 + 1 + 1
    num_h = [ 3 * 2 ] * num_layer
    context_length = 256
    all_steps = 6600            #  epoch * (len(input_texts)//batchsize//context_length) #169000

    while inputid[-1] + context_length + 1 > len(input_texts):
        inputid = inputid[:-1]

    ADAM = True
    cls_token = True
    float32 = True

    logfile = os.path.join(logdir, 'log_gpt_xiyouji.txt')
    fpwrite = open(logfile, 'a+', encoding='utf-8')

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
    layers += [at0, at1, at2, at3] #, at4, at5]#, at6, at7, at8, at9, at10, at11]

    norm = layer_norm(embed_dim, adam=ADAM)
    flatten     = flatten_layer()
    # if not cls_token:
    if not single_word:
        cll = classify_layer(embed_dim, batchsize, 1, vocab_size, cls_token=False, adam=ADAM, relu=False, float32=float32)
        layers += [norm, cll]
    else:
        cll = classify_layer(embed_dim, batchsize, np.sqrt(context_length), vocab_size, cls_token=False, adam=ADAM, relu=False, float32=float32)
        layers += [norm, flatten, cll]
    # else:
        # cll = fclayer(embed_dim, vocab_size, True, adam=ADAM, float32=float32)
    jk = 0
    num_attention = 0
    if os.path.exists(pretrained_model):
        with open(pretrained_model, 'rb') as obj:
            models = pickle.load(obj)
        if isinstance(models[-1], int):
            jk = models[-1]
        cnt = 0
        for l in layers:
            k = dir(l)
            nam = l.__module__
            if 'attdecoderblock' in nam:
                num_attention += 1
            if 'restore_model' in k and 'save_model' in k:
                try:
                    l.restore_model(models[cnt])
                except:
                    jk = 0
                    l.restore_model(models[cnt - 1])
                    continue
                cnt += 1
        del models
    else:
        exit(-1)

    nam = savename.split('last')[0] + 'last'
    savename = nam + "%d.pkl"%num_attention

    lr = learning_rate
    try:
        while jk < all_steps:
            meanloss = 0
            pre_col = []
            while True:
                if jk > all_steps:
                    break
                # if all_steps <= 100:
                #     lr = learning_rate * all_steps / 100
                # if all_steps==23*all_steps//30:
                #     lr = learning_rate * 0.1
                # elif all_steps==28*all_steps//30:
                #     lr = learning_rate * 0.1 * 0.1
                jk += 1
                inputs, input_mask, label_single, _ = getinputs(inputid, context_length, batchsize, input_texts, char2id, id2char)
                if single_word:
                    input_mask[...] = 0
                for l in range(len(layers)):
                    if isinstance(layers[l], attdecoderblock_layer):
                        inputs = layers[l].forward(inputs, input_mask)
                    else:
                        inputs = layers[l].forward(inputs)
                # inputs = inputs[:, -1, :]
                # label_single = label_single[:, -1]
                ishape = inputs.shape
                inputs = np.reshape(inputs, (-1, vocab_size))
                labels = np.zeros_like(inputs)
                labels[np.arange(len(inputs)), label_single.reshape(-1)] = 1
                loss, delta, predict = cross_entropy_loss(inputs, labels)
                # loss = loss * batchsize
                # delta = delta * batchsize
                delta = np.reshape(delta, ishape)
                
                # delta = np.zeros_like(inputs)
                # loss = 0
                # predict = np.zeros_like(inputs[0])
                # for ik in range(batchsize):
                #     labels = np.zeros_like(inputs[ik])
                #     labels[np.arange(len(inputs[ik])), label_single[ik]] = 1
                #     losskkk, deltakkk, predictkkk = cross_entropy_loss(inputs[ik], labels)
                #     delta[ik, :, :] = deltakkk
                #     loss += losskkk
                #     predict = np.concatenate([predict, predictkkk], axis = 0)
                # predict = predict[32*16//2:, :]
                # delta *= batchsize
                # loss *= batchsize
                for l in range(len(layers)-1, -1, -1):
                    delta = layers[l].backward(delta)
                    layers[l].update(lr)
                    layers[l].setzero()

                p = np.argmax(predict, axis=-1)
                if single_word:
                    precision = np.sum(label_single.reshape(-1)==p) / len(p)
                else:
                    p = np.reshape(p, label_single.shape)
                    precision = np.sum(p[:, -1]==label_single[:, -1]) / len(p)
                
                pre_col.append(precision)
                meanloss += loss
                i = jk * context_length * batchsize // len(input_texts)
                if jk%50==0:
                    inputs, input_mask, label_single, id_start = getinputs(inputid, 30, 1, input_texts, char2id, id2char)
                    label_sin = input_texts[id_start[0] + 30:id_start[0]+context_length*3+1]
                    # shape = label_single.shape
                    out = list(inputs.reshape(-1))
                    valprecision = []
                    if single_word:
                        input_mask[...] = 0
                    # ink = inputs.copy()
                    for ij in range(context_length):
                        label_single = np.array([[char2id[label_sin[ij]]]])
                        for l in range(len(layers)):
                            if isinstance(layers[l], attdecoderblock_layer):
                                inputs = layers[l].forward(inputs, input_mask)
                            else:
                                try:
                                    inputs = layers[l].forward(inputs)
                                except:
                                    kl = 0
                        ishape = inputs.shape
                        inputs = np.reshape(inputs, (-1, vocab_size))
                        labels = np.zeros_like(inputs)
                        labels[np.arange(len(inputs)), label_single.reshape(-1)] = 1
                        # k = np.sum(labels, axis = -1)
                        _, _, predict = cross_entropy_loss(inputs, labels)
                        p = np.argmax(predict, axis=-1)
                        if single_word:
                            valpre = np.sum(label_single.reshape(-1)==p) / len(p)
                        else:
                            p = p.reshape(-1)
                            valpre = np.sum(label_single[:, -1]==p[-1]) / len(p)
                            # p = p.reshape(-1)
                        valprecision.append(valpre)
                        out.append(p[-1])
                        inputs = np.array(out[-context_length:])[np.newaxis, :]
                        input_mask = create_masks_future(inputs)
                        if single_word:
                            input_mask[...] = 0
                    valpre = np.mean(valprecision)
                    out.insert(context_length, 0)
                    out.insert(context_length, 0)
                    out.insert(context_length, 0)
                    output = ''.join([id2char[int(ij)] for ij in out]) + "\n"
                else:
                    output = ""
                    valpre = 0
            
                fpwrite.write("epoch:{}, lr: {:.6f}, loss: {:.6f}, iters: {}, precision: {:.6f}, valpre: {:.6f}\n{}". \
                        format(i, lr, loss, str(jk) +"_"+ str(all_steps), precision, valpre, output))
                fpwrite.flush()

                # savemodel
                if (jk + 1) % 50==0:
                    savemodel(layers, jk)

            meanloss /= jk

            fpwrite.write("epoch: {},  {}\n\n".format(i, ''.join(output[:200])))
            fpwrite.flush()
        fpwrite.close()
    except KeyboardInterrupt:
        savemodel(layers, jk)

def savemodel(layers, jk):
    global savename
    allmodel = []
    for l in layers:
        k = dir(l)
        if 'restore_model' in k and 'save_model' in k:
            allmodel.append(l.save_model())
    allmodel.append(jk)
    with open(os.path.join(modelpath, savename), 'wb') as obj:
        pickle.dump(allmodel, obj)

if __name__ =="__main__":
    single_word = False
    savename = f"gpt_xiyouji_last3.pkl"
    modelpath = os.path.join(abspath, 'gpt', 'model')
    os.makedirs(modelpath, exist_ok=True)
    pretrained_model = os.path.join(abspath, 'gpt', 'model', savename)
    logdir = os.path.join(abspath, 'gpt', 'log')
    os.makedirs(logdir, exist_ok=True)
    transformer_image_train()

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