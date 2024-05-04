import numpy as np
import matplotlib.pyplot as plt
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
from attdecoderblock import attdecoderblock_layer
from gpt.gpt_linear import gpt_linear_layer
from gpt.gpt_train_potry3000 import getdata, create_masks_future
from net.layernorm import layer_norm
from net.fullconnect import fclayer
from classify import classify_layer

from copy import deepcopy
import json
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

def predict(inputs):
    pretrained_model = r'C:\Users\10696\Desktop\Numpy\numpy_transformer\gpt\model\gpt_poetry3000_iters1999_1_loss_3259.634242.pkl'
    vocab_size, id2char, char2id, input_texts = getdata()

    all_steps = 3000 - 1000
    batchsize = 63 + 1
    learning_rate = 0.003                         #   batchsize
    embed_dim = 192 ## vocab_size if vocab_size%3==0 else (vocab_size//3) * 3 + 3 # 192
    num_layer = 10 + 1 + 1
    num_h = [3] * num_layer
    context_length = 100

    ADAM = False
    cls_token = True
    float32 = True

    patchemb = Position_Embedding(context_length, vocab_size, embed_dim, adam=ADAM)
    layers = [patchemb]
    
    at0 = attdecoderblock_layer(embed_dim, num_h[0], adam=ADAM, float32=float32, return_attention=True)
    at1 = attdecoderblock_layer(embed_dim, num_h[1], adam=ADAM, float32=float32)
    at2 = attdecoderblock_layer(embed_dim, num_h[2], adam=ADAM, float32=float32)
    at3 = attdecoderblock_layer(embed_dim, num_h[3], adam=ADAM, float32=float32, return_attention=True)
    at4 = attdecoderblock_layer(embed_dim, num_h[4], adam=ADAM, float32=float32)
    at5 = attdecoderblock_layer(embed_dim, num_h[5], adam=ADAM, float32=float32)
    at6 = attdecoderblock_layer(embed_dim, num_h[6], adam=ADAM, float32=float32)
    at7 = attdecoderblock_layer(embed_dim, num_h[7], adam=ADAM, float32=float32)
    at8 = attdecoderblock_layer(embed_dim, num_h[8], adam=ADAM, float32=float32)
    at9 = attdecoderblock_layer(embed_dim, num_h[9], adam=ADAM, float32=float32)
    at10 = attdecoderblock_layer(embed_dim, num_h[10], adam=ADAM, float32=float32)
    at11 = attdecoderblock_layer(embed_dim, num_h[11], adam=ADAM, float32=float32)
    # at12 = attdecoderblock_layer(embed_dim, num_h[12], adam=ADAM, float32=float32)
    # at13 = attdecoderblock_layer(embed_dim, num_h[13], adam=ADAM, float32=float32)

    # layers += [at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11, at12]
    layers += [at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11]
    # layers += [at0, at1, at2, at3, at4, at5, at6]

    norm = layer_norm(embed_dim, adam=ADAM)
    # if not cls_token:
    #     cll = classify_layer(embed_dim, batchsize, 1, vocab_size, cls_token, adam=ADAM, relu=False, float32=float32)
    # else:
    cll = fclayer(embed_dim, vocab_size, True, adam=ADAM, float32=float32)
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
    
    inputs = [char2id[ci] for ci in inputs]
    inputs = np.array([inputs])
    # inputs = np.random.randint(0, vocab_size, (1, 1))
    output = deepcopy(inputs)
    for ij in range(context_length - 1):
        text = deepcopy(inputs)
        input_mask_fut = create_masks_future(inputs)
        input_mask_fut[...] = 0
        for l in range(len(layers)):
            if isinstance(layers[l], attdecoderblock_layer):
                inputs = layers[l].forward(inputs, input_mask_fut)
                if layers[l].return_attention==True:
                    return inputs
            else:
                inputs = layers[l].forward(inputs)
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        # Note also that we must extrapolate beyond vmin/vmax
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.]
        return np.ma.masked_array(np.interp(value, x, y,
                                            left=-np.inf, right=np.inf))

    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)

def plotattention():
    # https://matplotlib.org/stable/users/explain/text/fonts.html
    # trigger core fonts for PDF backend
    # from matplotlib.font_manager import _get_win32_installed_fonts, FontProperties, get_font, findSystemFonts
    # k = findSystemFonts()
    # fp = FontProperties()
    # fam = fp.get_family()
    # fp.set_family()
    plt.rcParams['font.family'] = ['DengXian']
    # plt.rcParams['font.family'] = ['SimHei']
    # trigger core fonts for PS backend
    # plt.rcParams["ps.useafm"] = True
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    inputs = r'床前明月光'#满树桃花映日开 床前明月光 山高江水深
    # inputs = r'独客长安觉月遥'
    after_softmax, before_softmax = predict(inputs)
    after_softmax = after_softmax[0]
    before_softmax = before_softmax[0]
    fig, ax = plt.subplots(2, 3)
    cmap = "viridis" #"PuOr"
    ax_col = []
    for i in range(len(after_softmax)):
        att = np.array(after_softmax[i])
        att[att<0] = 0
        for j in range(len(att)):
            att[j, j] = 0
        ax[i//3, i%3].imshow(att, cmap = cmap)
        ax[i//3, i%3].label_outer()
        # Show all ticks and label them with the respective list entries
        ax[i//3, i%3].set_xticks(np.arange(len(inputs)), labels=inputs)
        ax[i//3, i%3].set_yticks(np.arange(len(inputs)), labels=inputs)
        ax[i//3, i%3].tick_params(top=True, bottom=False,
                    labeltop=True, labelbottom=False)
        ax[i//3, i%3].spines[:].set_visible(False)
        # ax[i//3, i%3].set_xticks(np.arange(len(inputs)+1)-.5, minor=True)
        # ax[i//3, i%3].set_yticks(np.arange(len(inputs)+1)-.5, minor=True)
        ax[i//3, i%3].grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax[i//3, i%3].tick_params(which="minor", bottom=False, left=False)

    for i in range(len(before_softmax)):
        att = np.array(before_softmax[i])
        att[att<0] = 0
        for j in range(len(att)):
            att[j, j] = 0
        ax_col.append(ax[(i+3)//3, (i+3)%3].imshow(att, cmap=cmap))
        ax[i//3, i%3].label_outer()
        # Show all ticks and label them with the respective list entries
        ax[(i+3)//3, (i+3)%3].set_xticks(np.arange(len(inputs)), labels=inputs)
        ax[(i+3)//3, (i+3)%3].set_yticks(np.arange(len(inputs)), labels=inputs)
        ax[(i+3)//3, (i+3)%3].tick_params(top=True, bottom=False,
                                            labeltop=True, labelbottom=False)
        ax[(i+3)//3, (i+3)%3].spines[:].set_visible(False)
        # ax[(i+3)//3, (i+3)%3].set_xticks(np.arange(len(inputs)+1)-.5, minor=True)
        # ax[(i+3)//3, (i+3)%3].set_yticks(np.arange(len(inputs)+1)-.5, minor=True)
        ax[(i+3)//3, (i+3)%3].grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax[(i+3)//3, (i+3)%3].tick_params(which="minor", bottom=False, left=False)

    # Find the min and max of all colors for use in setting the color scale.
    # vmin = min(image.get_array().min() for image in images)
    # vmax = max(image.get_array().max() for image in images)
    # norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # for im in images:
    #     im.set_norm(norm)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "6%", pad="3%")
    # plt.colorbar(im, cax=cax)
    plt.colorbar(ax_col[-1], ax=ax, cax = cax, orientation='vertical', fraction=.09, location="right")
    # fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    # cax = fig.axes((0.85, 0.1, 0.07, 0.8))
    # fig.colorbar(cax=cax)
    fig.tight_layout()
    # Create colorbar
    plt.show()
    plt.close()

    # k = np.array([[-0.01635286, -0.11992685,  0.55947667,  0.92379665, -0.51663721,
    #     -0.15404126, -0.00184499],
    #    [-0.2325137 ,  0.07962042, -0.21066462,  0.26676813,  0.07328191,
    #      0.05249183, -0.0999634 ],
    #    [-0.56667554, -0.40644667, -0.02308455,  0.33910039, -0.07462835,
    #      0.32599962, -0.07650095],
    #    [-0.17541669, -0.11392358,  0.09849485,  0.29542559, -0.46458262,
    #     -0.17263578,  0.33820251],
    #    [-0.31038493,  0.1939393 , -0.33804718,  0.22234213, -0.46945375,
    #      0.07869679,  0.31734848],
    #    [ 0.04049709, -0.38160264, -0.0219599 , -0.28630418, -0.98243105,
    #     -0.64073545, -0.28534561],
    #    [-0.6606887 ,  0.15611888, -0.48505753, -0.24753423, -0.10505721,
    #     -0.26085418,  0.90769702]])
    # k[k<0] = 0
    # for i in range(len(k)):
    #     k[i, i] = 0
    # plt.imshow(k)
    # plt.show()
    # plt.close()

if __name__=="__main__":
    plotattention()