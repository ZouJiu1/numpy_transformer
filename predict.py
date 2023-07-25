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

def loading_model(num_classes):
    epoch = 30
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
    
    if os.path.exists(pretrained_model):
        with open(pretrained_model, 'rb') as obj:
            models = pickle.load(obj)
        cnt = 0
        for l in layers:
            k = dir(l)
            if 'restore_model' in k and 'save_model' in k:
                l.restore_model(models[cnt])
                cnt += 1
    for l in layers:
        k = l.__class__.__name__
        if k=="layer_batchnorm":
            l.train = False
    return layers

def predict_evaluate(layers):
    batchsize = 100
    datapath = os.path.join(abspath, 'dataset')
    os.makedirs(datapath, exist_ok=True)
    
    datatest = datasets.MNIST(root = datapath, train=False, download=True)
    testdata, testlabel = datatest._load_data()
    # */255
    testdata, testlabel = testdata.cpu().numpy() / 255, testlabel.cpu().numpy()
    #one-hot
    test_label = np.zeros((len(testlabel), 10))
    test_label[range(len(testlabel)), testlabel] = 1
    test_l = testlabel.copy()
    testlabel = test_label.copy()

    if predict_or_evaluate:
        cvshow = os.path.join(abspath, 'cvshow')
        os.makedirs(cvshow, exist_ok = True)
        for i in os.listdir(cvshow):
            os.remove(os.path.join(cvshow, i))
        for i in range(len(testlabel)):
            img = testdata[i, :, :]
            ori = (deepcopy(img)[:, :]*255).astype(np.uint8)
            img = img[np.newaxis, np.newaxis, :, :]
            truth = test_l[i]
            for l in range(len(layers)):
                img = layers[l].forward(img)
            p_shift = img - np.max(img, axis = -1)[:, np.newaxis]   # avoid too large in exp 
            predict = np.exp(p_shift) / np.sum(np.exp(p_shift), axis = -1)[:, np.newaxis]
            p = np.argmax(predict, axis=-1)[0]
            # plt.imshow(ori)
            # plt.title("Predict:"+str(p)+", Truth:"+str(truth))
            # plt.savefig(os.path.join(cvshow, str(i)+"_p_"+str(p)+"_t_"+str(truth)+ ".jpg"), bbox_inches='tight')
            image = Image.fromarray(ori).convert("L")
            image.save(os.path.join(cvshow, str(i)+"_Predict_"+str(p)+"_Truth_"+str(truth)+ ".jpg"))
            if i > 10:
                break
    else:
        dic = {i:0 for i in range(10)}
        acc = 0
        length = 0
        for j in range(len(test_l)):
            images = testdata[j*batchsize:(j+1)*batchsize, :, :]
            images = images[:, np.newaxis, :, :]
            label = testlabel[j*batchsize:(j+1)*batchsize, :]
            label_single = test_l[j*batchsize:(j+1)*batchsize]
            if len(images)==0:
                break
            for l in range(len(layers)):
                kl = dir(layers[l])
                if '__name__' in kl and 'layer_batchnorm' in layers[l].__name__():
                    layers[l].train = False
                images = layers[l].forward(images)
            loss, delta, predict = cross_entropy_loss(images, label)
            p = np.argmax(predict, axis=-1)
            length += len(label_single)
            acc += np.sum(label_single==p)
            
            for ij in range(len(p)):
                if p[ij]==label_single[ij]:
                    dic[p[ij]] += 1
            if j %1==0:
                print(j) 
        print(dic)
        dickk = {}
        for key, value in dic.items():
            label_g = np.array(test_l, dtype = np.int32)
            dickk[key] = value / np.sum(label_g==int(key))
        precision = acc / length
        name = pretrained_model.replace(".pkl", "_evalall.csv")
        dickk['precision'] = precision
        df = pd.DataFrame(dickk, index=np.arange(1)).T
        df.to_csv(os.path.join(abspath, name), index=True)

if __name__ =="__main__":
    savepath = abspath
    pretrained_model = r'C:\Users\10696\Desktop\access\numpy_transformer\model\epoch_33_loss_0.085326_pre_0.972__pf_pn_fixed.pkl'
    layers = loading_model(10)
    predict_or_evaluate = False
    predict_evaluate(layers)