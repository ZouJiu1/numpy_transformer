import os
from net.Convolution import convolution_layer
from net.loss import cross_entropy_loss, mean_square_loss
from net.fullconnect import fclayer
from net.activation import ReLU
from net.flatten import flatten_layer
import numpy as np
import pickle
from net.layernorm import layer_norm
from PatchEmbed import PatchEmbed_flatten, PatchEmbed_convolution
from Position_add import Position_learnable
from attention import attention_layer
from classify import classify_layer
from net.layernorm import layer_norm

from torchvision import datasets
from PIL import Image
import pandas as pd
from copy import deepcopy

abspath = os.path.abspath(__file__)
filename = abspath.split(os.sep)[-1]
abspath = abspath.replace(filename, "")

# https://en.wikipedia.org/wiki/AlexNet
# https://pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html#alexnet
# https://github.com/l5shi/Image-Recognition-on-MNIST-dataset/blob/master/AlexNet.ipynb

def transformer_image_train(num_classes):
    epoch = 36
    batchsize = 100
    lr = 0.001
    embed_dim = 96
    images_shape = (batchsize, 1, 30-2, 30-2)
    n_patch = 7
    patchnorm = True
    # [0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]
    fixed     = 1 #False
    cls_token = 0 #True
    num_h = [2*2] * 6 #[3, 6, 12, 3, 6, 12]
    patch_convolu = 0 #False

    if patch_convolu:
        choose = "_pc"
    else:
        choose = '_pf'
    if patchnorm:
        choose += "_pn"
    if fixed:
        choose += "_fixed"
    if cls_token:
        choose += "_clstoken"
    logfile = os.path.join(logdir, 'log_transformer_of_image_%s.txt'%choose)
    fpwrite = open(logfile, 'w', encoding='utf-8')

    if patch_convolu:
        patchemb = PatchEmbed_convolution(embed_dim, images_shape, n_patch, patchnorm = patchnorm)
    else:
        patchemb = PatchEmbed_flatten(embed_dim, images_shape, n_patch, patchnorm = patchnorm)
    positionL = Position_learnable(n_patch, embed_dim, fixed = fixed)
    att1 = attention_layer(embed_dim, num_h[0])
    att2 = attention_layer(embed_dim, num_h[1])
    att3 = attention_layer(embed_dim, num_h[2])
    layers = [patchemb, positionL, att1, att2, att3]

    att4 = attention_layer(embed_dim, num_h[3])
    att5 = attention_layer(embed_dim, num_h[4])
    att6 = attention_layer(embed_dim, num_h[5])
    layers += [att4, att5, att6]

    norm = layer_norm(embed_dim)
    flatten     = flatten_layer()
    cll = classify_layer(embed_dim, batchsize, n_patch, num_classes, cls_token)
    if not cls_token:
        layers += [norm, flatten, cll]
    else:
        layers += [norm, cll]
    
    datapath = os.path.join(abspath, 'dataset')
    os.makedirs(datapath, exist_ok=True)
    modelpath = os.path.join(abspath, 'model')
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

    datatest = datasets.MNIST(root = datapath, train=False, download=True)
    datatrain = datasets.MNIST(root = datapath, train=True, download=True)
    testdata, testlabel = datatest._load_data()
    datas, labels = datatrain._load_data()
    # */255
    testdata, testlabel = testdata.cpu().numpy() / 255, testlabel.cpu().numpy()
    datas, labels = datas.cpu().numpy() / 255, labels.cpu().numpy()
    #one-hot
    test_label = np.zeros((len(testlabel), 10))
    test_label[range(len(testlabel)), testlabel] = 1
    test_l = testlabel.copy()
    testlabel = test_label.copy()
    train_label = np.zeros((len(labels), 10))
    train_label[range(len(labels)), labels] = 1
    train_l = labels.copy()
    labels = train_label.copy()
    del test_label, train_label

    number_image = datas.shape[0]
    # for i in range(number_image):
    #     img = datas[i, :, :]
    #     Image.fromarray(img.cpu().numpy()).save(os.path.join(abspath, 'dataset', str(i) + ".jpg"))

    loss = 999999
    iters = number_image//batchsize + (0 if number_image%batchsize == 0 else 1)
    dot = np.power(0.001, 1/epoch)
    for i in range(25, epoch):
        meanloss = 0
        # if i!=0:
            # lr = lr * dot
        if i==20:
            lr = lr * 0.1
        elif i==33:
            lr = lr * 0.1
        k = np.arange(len(train_l))
        np.random.shuffle(k)
        datas = datas[k]
        labels = labels[k]
        
        train_l = train_l[k]
        for j in range(iters):
            images = datas[j*batchsize:(j+1)*batchsize, :, :]
            label = labels[j*batchsize:(j+1)*batchsize, :]
            label_single = train_l[j*batchsize:(j+1)*batchsize]
            images = images[:, np.newaxis, :, :]
            if len(images)==0:
                continue
            for l in range(len(layers)):
                if isinstance(layers[l], classify_layer):
                    if cls_token:
                        images = layers[l].forward(images[:, 0])
                    else:
                        images = layers[l].forward(images)
                else:
                    images = layers[l].forward(images)
            loss, delta, predict = cross_entropy_loss(images, label)
            meanloss += loss
            p = np.argmax(predict, axis=-1)
            precision = np.sum(label_single==p) / len(label_single)
                
            fpwrite.write("epoch:{}, lr: {:.6f}, loss: {:.6f}, iters: {}, precision: {:.6f}\n".format(i, lr, loss, j, precision))
            fpwrite.flush()
            for l in range(len(layers)-1, -1, -1):
                delta = layers[l].backward(delta)
                layers[l].update(lr)
                layers[l].setzero()
        acc = 0
        length = 0
        k = np.arange(len(testdata))
        # np.random.seed(999999666)
        np.random.shuffle(k)
        testdata = testdata[k]
        test_l = test_l[k]
        testlabel = testlabel[k]
        # if i==epoch-1:
        #     num = len(testdata)
        # else:
        num = len(testdata)//(1000)
        dic = {i:0 for i in range(10)}
        for j in range(num):
            images = testdata[j*batchsize:(j+1)*batchsize, :, :]
            images = images[:, np.newaxis, :, :]
            if images.shape[0]==0:
                continue
            label = testlabel[j * batchsize:(j+1) * batchsize, :]
            label_single = test_l[j * batchsize:(j+1) * batchsize]
            for l in range(len(layers)):
                if isinstance(layers[l], classify_layer):
                    if cls_token:
                        images = layers[l].forward(images[:, 0])
                    else:
                        images = layers[l].forward(images)
                else:
                    images = layers[l].forward(images)
            loss, delta, predict = cross_entropy_loss(images, label)
            p = np.argmax(predict, axis=-1)
            length += len(label_single)
            acc += np.sum(label_single==p)
            
            for ij in range(len(p)):
                if p[ij]==label_single[ij]:
                    dic[p[ij]] += 1
            
        precision = acc / length
        meanloss = meanloss / iters
        # savemodel
        allmodel = []
        for l in layers:
            k = dir(l)
            if 'restore_model' in k and 'save_model' in k:
                allmodel.append(l.save_model())
        name = "epoch_"+str(i)+"_loss_"+str(round(meanloss, 6))+"_pre_"+str(round(precision, 6))+"_%s.pkl"%choose
        
        with open(os.path.join(modelpath, name), 'wb') as obj:
            pickle.dump(allmodel, obj)
            
        # dic['precision'] = precision
        # df = pd.DataFrame(dic, index=np.arange(1)).T
        # df.to_csv(os.path.join(abspath, name.replace(".pkl", ".csv")), index=True)

        fpwrite.write("epoch: {}, testset precision: {}\n\n".format(i, precision))
        fpwrite.flush()
    fpwrite.close()

if __name__ =="__main__":
    savepath = abspath
    pretrained_model = r'C:\Users\10696\Desktop\access\numpy_transformer\model\epoch_24_loss_0.211765_pre_0.937__pf_pn_fixed.pkl'
    logdir = os.path.join(savepath, 'log')
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
    