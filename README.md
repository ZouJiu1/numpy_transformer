# Transformer in numpy
## VIT vision transformer
I write a VIT network in numpy fully, including forward and backpropagation.<br>
including those layers, **multi attention**, **PatchEmbed**, **Position_add**, **convolution**, **Fullconnect**, **flatten**, **Relu**, **layer_norm**, **Cross Entropy loss** and **MSE loss**<br>
In training, it use cpu and slowly, so I use different settings<br>

Training it with MNIST dataset, **it’s precision can reach to 97.2%**, it's setting is <br>
```
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
```

this codes provide functions to save model and restore model to train<br>
you can find those models in model dir<br><br>
Train with command<br>
```
python transformer_of_image.py
```

### predict

```
python predict.py
```

### precision
train in MacBook Pro 2020 Intel

| classes | precision |
| ------ | ------ |
| 0 | 0.9908163265306122 |
| 1 | 0.9903083700440528 |
| 2 | 0.9748062015503876 |
| 3 | 0.9831683168316832 |
| 4 | 0.9674134419551935 |
| 5 | 0.9708520179372198 |
| 6 | 0.9739039665970772 |
| 7 | 0.9630350194552529 |
| 8 | 0.9517453798767967 |
| 9 | 0.9544103072348861 |
| all precision | 0.972 |

## gpt character numpy
in directory `gpt_character`

**Train and predict**
```
python gpt_character\gpt_train_english_char.py
python gpt_character\gpt_charpredict.py
```

**Result**
```
'm: nopqrstuv',
'p: qrstuvwxy',
'w: xyz abcde',
'w: xyz abcde',
'x: yz abcdef',
'f: ghijklmno',
't: uvwxyz ab',
'p: qrstuvwxy',
'y: z abcdefg',
'w: xyz abcde'
```

## gpt in numpy

now I'm training the model

## blogs
[numpy实现VIT vision transformer在MNIST-https://zhuanlan.zhihu.com/p/645326689](https://zhuanlan.zhihu.com/p/645326689)<br>


总共实现了这几个层：

[numpy实现vision transformer图像输入的patch-https://zhuanlan.zhihu.com/p/645318207](https://zhuanlan.zhihu.com/p/645318207)

[numpy实现vision transformer的position embedding-https://zhuanlan.zhihu.com/p/645320199](https://zhuanlan.zhihu.com/p/645320199)

[numpy实现multi-attention层的前向传播和反向传播-https://zhuanlan.zhihu.com/p/645311459](https://zhuanlan.zhihu.com/p/645311459)

[全连接层的前向传播和反向传播-https://zhuanlan.zhihu.com/p/642043155](https://zhuanlan.zhihu.com/p/642043155)

[损失函数的前向传播和反向传播-https://zhuanlan.zhihu.com/p/642025009](https://zhuanlan.zhihu.com/p/642025009)

## Reference
[https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py](https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py)<br>
[https://github.com/UdbhavPrasad072300/Transformer-Implementations/blob/main/notebooks/MNIST%20Classification%20-%20ViT.ipynb](https://github.com/UdbhavPrasad072300/Transformer-Implementations/blob/main/notebooks/MNIST%20Classification%20-%20ViT.ipynb)<br>
[https://github.com/s-chh/PyTorch-Vision-Transformer-ViT-MNIST/tree/main](https://github.com/s-chh/PyTorch-Vision-Transformer-ViT-MNIST/tree/main)<br>
[https://itp.uni-frankfurt.de/~gros/StudentProjects/WS22_23_VisualTransformer/](https://itp.uni-frankfurt.de/~gros/StudentProjects/WS22_23_VisualTransformer/)<br>
[https://jamesmccaffrey.wordpress.com/2023/01/10/a-naive-transformer-architecture-for-mnist-classification-using-pytorch/](https://jamesmccaffrey.wordpress.com/2023/01/10/a-naive-transformer-architecture-for-mnist-classification-using-pytorch/)<br>
[https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c)<br>
[https://github.com/BrianPulfer/PapersReimplementations/blob/main/vit/vit_torch.py](https://github.com/BrianPulfer/PapersReimplementations/blob/main/vit/vit_torch.py)<br>
[https://github.com/microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)<br>
[https://huggingface.co/docs/transformers/v4.27.0/model_doc/vit](https://huggingface.co/docs/transformers/v4.27.0/model_doc/vit)<br>