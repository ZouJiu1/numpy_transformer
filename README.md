# Transformer in numpy
## VIT vision transformer
I write a VIT network in numpy fully, including forward and backpropagation.<br>
including those layers, **multi attention**, **PatchEmbed**, **Position_add**, **convolution**, **Fullconnect**, **flatten**, **Relu**, **layer_norm**, **Cross Entropy loss** and **MSE loss**<br>
In training, it use cpu and slowly, so I use different settings<br>
Training it with MNIST dataset, **it’s precision can > 90%**<br>
this codes provide functions to save model and restore model to train<br>
you can find those models in model dir<br><br>
Train with command<br><br>
```
python transformer_of_image.py
```

### predict

```
python predict.py
```

### precision

| classes | precision |
| ------ | ------ |
| 0 | 0.9887755102040816 |
| 1 | 0.9876651982378855 |
| 2 | 0.9689922480620154 |
| 3 | 0.9633663366336633 |
| 4 | 0.9592668024439919 |
| 5 | 0.9495515695067265 |
| 6 | 0.9728601252609603 |
| 7 | 0.938715953307393 |
| 8 | 0.9332648870636551 |
| 9 | 0.9534192269573836 |
| all precision | 0.90 |

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