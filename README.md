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

## the gpt poetry3000 with the numpy
train precision is 96%, you can find it in the file log_gpt_poetry3000.txt.

##### train
```
python gpt_train_potry3000.py
```

##### run 
the predict model is the gpt_poetry3000_iters1999_1_loss_3259.634242.pkl
```
python gpt_predict_poetrythree.py
```
##### blogs 
[https://zhuanlan.zhihu.com/p/659018819 numpy实现GPT的decoder来产生旧诗词的](https://zhuanlan.zhihu.com/p/659018819)

**Result**
```
拂马似尘飞。 叶浓知柳密，花尽觉梅疏。兰生未可握，蒲小不堪书。 梅含今春树，还临先日池。人怀前岁忆，
花发故年枝。 池平生已合，林花发稍稠。风入花枝动，日照水光浮。 空庭高楼月，非复三五圆。

一戍鸣烟直，平沙落日迟。 露寒金掌重，天近玉绳低。 惊蝉移古柳，斗雀堕寒庭。 鹤传沧海信，僧和白云诗。 
鸟暝风沉角，天清月上旗。 多年不道姓，几日旋移家。 喧风生木末，迟景入泉心。 湘云随雁断，

其如旅病牵。抱琴传此意，栖岳计何年。暮倚中流楫，闻歌徒自怜。 侍史趋清禁，承恩下直庐。瓶馀赊得酒，
架积赐来书。刻凤才何有，雕虫习未除。由来少尘事，寂寞意何如。 锦石带寒英，秋光澹客情。色增

一天渠终更谁先，聊复怜渠与酒钱。富贵不愁天不管，不应丘壑也关天。 雨涨平池绿似淮，半扉春水眼慵开。
无钱得买扁舟去，莫道旧来今不来。相望千家信不通，悬知春在雨声中。蹇驴欲去愁泥滑，安得西飞六尺

谁寞经旬见此枝。 花开犹未报人知，花下行吟漫自思。花若能言应笑我，年年无酒只题诗。 晚发西山舟欲北，
天风吹我复还家。十年一到非容易，独立平原看稻花。 新开竹径贮秋多，携酒烦公每见过。月出未高公已

去愁架子之酒楼。 危桥当古寺，闲倚喜同僧。极浦霁秋雨，扁舟明夜灯。风沈人语远，潮涨月华升。万事空凝念，
其如总未能。 松间灯夕过，顾影在天涯。雪暝迷归鹤，春寒误早花。艰难知世味，贫病厌年华。故国

忽孙共读书。 云沈秋驿雨，鸡送晓窗灯。 门当车马道，帘隔利名尘。 云开千里月，风动一天星。 绿涨他山雨，
青浮近市烟。 月色四时好，人心此夜偏。 春水有秀色，野云无俗姿。 出处自有时，人生安得偕

归弱。 问落莫空山里，唤入诗人几案来。 云欲开时又不开，问天觅阵好风催。雨无多落泥偏滑，溪不胜深岸故颓。
添尽红炉著尽衣，一杯方觉暖如痴。人言霜后寒无奈，春在瓮中渠不知。 梅不嫌疏杏要繁，主人何

黯其将雨。嗟我怀人，道修且阻。眷此区区，俯仰再抚。良辰过鸟，逝不我伫。 意不若义，义不若利。利之使人，
能忘生死。利不若义，义不若意。意之使人，能动天地。 居暗观明，居静观动。居简观繁，居轻

更长烛屡花。一轮观浴兔，两部听鸣蛙。 诚能得初心，何必返初服。有以固中扃，不须防外逐。 野驿人稀到，
空庭草自生。霜清殊未觉，雨细更含晴。 老屋愁风破，空林过雨乾。飘零黄叶满，寂寞野香残

月在画楼西。 妾如江边花，君如江上水。花落随水流，东风吹不起。 妾家横塘东，与郎乍相逢。郎来不须问，
门外植梧桐。 昼静暖风微，帘垂客到稀。画梁双燕子，不敢傍人飞。 水抱孤村远，山通一

何山起暮馀。当庭波始阔，峡水月常阴。魂梦犹难到，愁君白发侵。 水如树欲静，滩如风不宁。百里断肠声，
当年游子听。一往不可复，此行安所欲。千古流水心，耿耿在幽独。 疏放难违性，苔荒野巷深。到门黄叶雨

片处处云生。 零落雪霜后，犹含千载春。一株化为石，谁是种时人。 佛心随处见，层出更分明。不用催灯火，
天高月自生。 乾坤皆数五，日月正符同。但仰重离照，难名厚载功。 水畔幡竿险，分符得异恩。

仰彼苍苍可奈何。浊酒一杯愁未解，唾壶击碎不成歌。 木犀香透越山云，记得根从海上分。恨杀西风夜来恶，
一枝摧处正愁君。 天意于人有浅深，人于天意岂容心。一行一止惟时耳，此道堂堂古到今。 丹鼎刀圭炼

镜日上。我怀前岁忆，花发故年枝。 池平生已合，林花发稍稠。风入花枝动，日照水光浮。 空庭高楼月，
非复三五圆。何须照床里，终是一人眠。 别怨凄歌响，离啼湿舞衣。愿假乌栖曲，翻从南向飞。 三洲断江口

月上秋来醉，空斋夜落声。隔床惊昨梦，隐几话平生。灯净书还读，香销句忽成。他年相望处，吾亦用吾情。 
羡棹吴松曲，来寻独冷盟。误听对床雨，唤作打篷声。漏缓更筹滴，春从水驿生。晓云驱宿翳，我欲趁新晴
```

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