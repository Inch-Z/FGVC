https://tool.lu/markdown/

## 在Standford Dogs上的实验

### 1：注意力模块对比实验

***Standford Dogs数据集简介：***

训练集: 12000

测试集：8580

Classes：120

数据集图片示例：

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml5584\wps1.png)![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml5584\wps2.png)![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\ksohtml5584\wps3.png) 

***\*Implementation Details:\****

使用ImageNet-1K预训练权重

Input-size：图片缩放至256x256，随机翻转15度，随机裁剪至224x224

Batch-size：64

Optimizer：SGD (moment=0.9)

Weight-Decay：1e-4

Epochs：150

Init_learning_rate: 1e-4 (每50个epoch衰减为原学习率的1/10)

### 1.1传统安插注意力模块对比实验

|  resnet50  |  +SE   | +CBAM(sa)  |  +Eca  | + Fca  | +Fca&Eca |     备注     |
| :--------: | :----: | :--------: | :----: | :----: | :------: | :----------: |
| **87.09%** | 85.19% | **86.77%** | 85.40% | 85.86% |  85.06%  | init_lr=5e-4 |

***\*实验结果分析：\****

加入SOTA的attention block，相比baseline，测试集准确率不升反降。预计是由于在每一个Bottleneck中加入attention block后易对预训练模型的初始化权重产生较大扰动，导致效果不佳。

***\*后续改进思路：\****

参照Semantic Segmentation领域的做法(例CCNet、DANet)，仅将attention block安插在网络的分类层前，以减少对网络backbone中预训练参数的扰动。



### 1.2 加在最后一层(全局平均池化压缩前)的对比实验

#### 实验日期(2.07-2.14)

|           resnet50            |                    +SE                    |             +SE(残差连接)             |             +CBAM(sa)             |  +CBAM(sa残差连接)   |
| :---------------------------: | :---------------------------------------: | :-----------------------------------: | :-------------------------------: | :------------------: |
|            87.09%             |                  87.04%                   |                87.89%                 |              87.54%               |      **88.35%**      |
|      **+Fca(残差连接)**       |            **+Eca(残差连接)**             |      **+DCT_CBAM(sa 残差连接)**       | **+CBAM(upsample=4，sa残差连接)** | **+SKNet(M=3，G=8)** |
|            87.91%             |                  88.07%                   |                88.02%                 |            **88.35%**             |        87.77%        |
| **+SKNet(M=3，G=8,残差连接)** | **+SKNet(upsample=4，M=4，G=16残差连接)** | **+CBAM(sa，《插入第一层》残差连接)** |     **+CBAM(ca+sa,残差连接)**     |                      |
|            88.24%             |                  87.60%                   |                87.17%                 |              88.04%               |                      |

##### 总结：

1. 使用预训练模型时，attention block插入每一个res_block效果不佳，加在最后一层效果较好。
2. attention block在最后使用残差连接(加权结果Feature map+原始Feature map)效果比直接用加权结果Feature map更好。
3. 轻量级空间注意力CBAM_spatial效果最好，达到88.35%，对比baseline提升了约1.3%。

##### 后续实验改进经验：

1. 针对“总结”中的第1点，可以在训练中针对不同的层分配不同的学习率，在预训练好的basebone分配低学习率，在attention block处分配大学习率。



### 2：分层学习率实验

#### 实验日期(2.17-2.19)

| +AC(k=3,keep_size)(FC层<br/>-init_lr=1e-3)，(backbone<br/>-init_lr=1e-4)) |                      +AC(k=3,keep_size)                      |                 +AC(k=3,keep_size,残差连接)                  | +AC(k=3,keep_size,残差连接)<br/>(FC与AC层-init_lr=1e-2)，(backbone<br/>-init_lr=1e-4)) | +Fca<br/>(input_size=448,all_layer) (FC与AC层<br/>-init_lr=1e-3)，(backbone<br/>-init_lr=1e-4)) |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                          88.17%(21)                          |                            88.33%                            |                            88.42%                            |                            85.65%                            |                            88.39%                            |
| **+Fca<br/>(input_size=224,all_layer) (FC层<br/>-init_lr=1e-3)，(backbone<br/>-init_lr=1e-4))** | **resnet50<br>(FC层<br/>-init_lr=1e-3)，(backbone<br/>-init_lr=1e-4))** | **+Fca<br/>(input_size=224) (FC层<br/>-init_lr=1e-3)，(backbone<br/>-init_lr=1e-4))** | **+CBAM_sa<br/>(all_layer) <br>(FC层-init_lr=1e-3)，(backbone<br/>-init_lr=1e-4))** | **+CBAM_sa<br/>(FC层-init_lr=1e-3)，(backbone<br/>-init_lr=1e-4))** |
|                            88.42%                            |                          **88.73%**                          |                          **88.58%**                          |                            87.93%                            |                            88.48%                            |

***\*实验结果分析：\****

FC层学习率提升十倍后，准确率明显上升，Baseline提升了1.64%，达到目前最优的88.73%。除ACNet外，添加其余attention block对比统一学习率时，都有一定的提升。



***\*注：下阶段实验开始， Backbone的初始学习率(init_\*******\*lr)\*******\*均采用1e-4，F\*******\*C\*******\*分类层初始学习率均采用1e-\*******\*4\*******\*。每5\*******\*0\*******\*epoch衰减为原学习率的1/\*******\*10\*******\*。其余\*******\*Implementation Details保持不变。\****





### 3：分类层全局池化(GAP)替换实验

#### 实验日期(2.18-2.19)


|   resnet50<br>ACNet instead of GAP   |  resnet50<br/>DCT(top16) instead of GAP  |  resnet50 <br/>DCT(low16) instead of GAP   | resnet50 <br/>DCT(low16) instead of GAP<br>lr统一为1e-4 |                  备注：                  |
| :----------------------------------: | :--------------------------------------: | :----------------------------------------: | :-----------------------------------------------------: | :--------------------------------------: |
|                86.40%                |                  86.41%                  |                   86.35%                   |                         86.62%                          | 默认FC层init_lr=1e-3<br>backbone是其1/10 |
| **resnet50 <br/>GMP instead of GAP** | **resnet50 <br/> s_GMP(49)+c_GAP(2048)** | **resnet50 <br/> (2048)c_GMP+c_GAP(2048)** |     **resnet50 <br/>s_GMP(49)+s_GAP(49)+s_DCT(49)**     |                                          |
|                4.29%                 |                **88.61%**                |                 **87.12%**                 |                         72.39%                          |                                          |

| mobilenet-v2 | SE_mobilenet-v2 | CBAM_sa_mobilenet-v2 |
| :----------: | :-------------: | :------------------: |
|    82.01%    |     81.38%      |        82.03%        |

***\*实验结果分析：\****

尝试在最后FC层前的全局平均池化(GAP)采用了多种替代方案，结果不尽理想，还是原有仅使用GAP效果最佳。





### 4 ：backbone后置新网络实验(自设计)

#### 4.1 ：Multi-Kernal-Network(MK-Net)（自设计，参考SKNet、CAP]）

![image-20210330154904778](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210330154904778.png)

#### 实验日期(2.20-2.21)

**(Multi-Kernal默认为 3x3 & 5x5 & 7x7 )**

|   Multi-Kernal(out=512, G=8)<br>fc&mk(1e-3),backbone(1e-4)   |   Multi-Kernal(out=512, G=8)<br>fc(1e-3),mk&backbone(1e-4)   |  Multi-Kernal(out=2048, G=8)<br>fc&mk(1e-3),backbone(1e-4)   |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                        **88.40%(16)**                        |                        **88.26%(57)**                        |                          88.25%(54)                          |
| **Multi-Kernal(out=2048, G=8)<br/>fc&mk(1e-3),backbone(1e-5)** | **Multi-Kernal(out=2048, G=2)<br/>fc&mk(1e-3),backbone(1e-4)** | **Multi-Kernal(out=128, G=8)<br/>fc&mk(1e-3),backbone(1e-4)** |
|                          86.93%(26)                          |                         87.79%(105)                          |                          87.97%(20)                          |
| **Multi-Kernal-FcaNet(out=512, G=8)<br/>fc&mk(1e-3),backbone(1e-4)** | **Multi-Kernal-FcaPool(out=512, G=8)<br/>fc&mk(1e-3),backbone(1e-4)** |  **AC-Multi-Kernal(out=512, G=8)<br/>fc&mk(1e-3),bb(1e-4)**  |
|                          85.26%(50)                          |                         85.36%(133)                          |                         86.81%(123)                          |

***\*实验结果分析：\****

1. 使用MK-Net后，采用原512的通道数输出效果最佳，达到88.40%的准确率，低于Baseline的88.73%。

2. 后续在MK-Net基础上再次尝试《第三阶段》的GAP替换实验，效果同样未能提升。



#### 实验日期(2.21-2.22)

#### 4.2 ：Multi-Kernal-Interaction(MKI-Net，自设计，参考API-Net)

![image-20210330154856051](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210330154856051.png)

|                   Model                    |  Test acc(%)   |
| :----------------------------------------: | :------------: |
|                CC-Net(r=1)                 | **88.00**%(17) |
|                CC-Net(r=2)                 |   86.97%(11)   |
|       Multi-Kernal-API-V1(2048输出)        |   87.30%(16)   |
| Multi-Kernal-API-V1(2048输出，学习率不*10) |   87.51(55)    |
|        Multi-Kernal-API-V1(512输出)        |   87.30(18)    |
|       Multi-Kernal-API-V1(1536输出)        |   87.45(15)    |
|   Multi-Kernal-API-V2(weight-decay=1e-4)   |                |
|   Multi-Kernal-API-V2(weight-decay=5e-4)   |   88.08(39)    |





### 5 ：dropout功效实验

#### 实验日期(2.22-2.23)

|                          Model                           |  Test acc(%)  |
| :------------------------------------------------------: | :-----------: |
|                   Baseline(resnet-50)                    |   88.73(41)   |
|             Baseline + 训练&测试Dropout(0.5)             | **88.80(85)** |
|               Baseline + 训练Dropout(0.5)                | **88.86(49)** |
|          Baseline + FcaNet(最后一层，残差连接)           |   88.58(39)   |
| Baseline + FcaNet(最后一层，残差连接) + 训练Dropout(0.5) |   88.76(42)   |

####  **是否添加注意力模块对比实验(以下实验训练时FC层均采用Dropout(0.5))**

|                        Model                         |  Test acc(%)  |
| :--------------------------------------------------: | :-----------: |
|                 Baseline(resnet-50)                  | **88.73(41)** |
|    Baseline + 训练&测试FcaNet(最后一层，残差连接)    | **88.76(42)** |
|      Baseline + 训练FcaNet(最后一层，残差连接)       |   88.60(42)   |
|       Baseline + 训练&测试Multi-Kernal-API-V1        |   87.25(16)   |
| Baseline + 训练&测试Multi-Kernal-API-V1(学习率不*10) |   87.51(55)   |
|                                                      |               |

***\*实验结果分析：\****

1. 在通常情况下，dropout是有效的，能在一定程度上反正过拟合，但不可能成为主角。





### 6 ：优化器更换实验

#### 实验日期(3.2-3.4)

#### resnet——优化器&学习率调参实验

|                Model                 |  Test acc(%)   |
| :----------------------------------: | :------------: |
|    baseline (Adam, init_lr=1e-4)     |    75.63(2)    |
|    baseline (Adam, init_lr=1e-5)     |   84.77(13)    |
|    baseline (Adam, init_lr=1e-6)     | **85.36(134)** |
| baseline (Adam, init_lr=1e-6(fc*10)) | **87.89(107)** |

***\*实验结果分析：\****

1. 不太行，SGD永远的神。





### 7：Efficient-Net

#### 7.1 baseline调参实验

#### 实验日期(2.27-3.1)

|                            Model                             |     Test acc(%)     |
| :----------------------------------------------------------: | :-----------------: |
|                     以下input-size = 224                     |                     |
| Efficient-Net-B0(batch=24, dropout=0.5, init_lr=1e-4(fc*10)) |     85.87(129)      |
| Efficient-Net-B4(batch=38, dropout=0.5, init_lr=1e-4(fc*10)) |     90.92(140)      |
| Efficient-Net-B0(batch=23, dropout=0.2, init_lr=1e-4(fc*10)) |     85.65(139)      |
| Efficient-Net-B4(batch=23, dropout=0.2, init_lr=1e-4(fc*10)) |    **90.93(90)**    |
| Efficient-Net-B0(batch=23, dropout=0.2, init_lr=1e-3(fc*10)) |      84.77(20)      |
| Efficient-Net-B4(batch=23, dropout=0.2, init_lr=1e-3(fc*10)) |      90.62(10)      |
| Efficient-Net-B7(batch=16, dropout=0.5, init_lr=1e-4(fc*10)) |    **91.29(70)**    |
|                                                              |                     |
|                     以下input-size = 380                     |                     |
| Efficient-Net-B4(batch=14, dropout=0.5, init_lr=1e-4(fc*10)) |    **94.02(47)**    |
| Efficient-Net-B4(batch=14, dropout=0.5, 余弦退火-init_lr=1e-4(T_max=10))) | 93.58(133) - 欠拟合 |
| Efficient-Net-B4(batch=14, dropout=0.5, 余弦退火-init_lr=1e-4(fc*10， T_max=10))) |    **94.07(39)**    |
| Efficient-Net-B4(batch=14, dropout=0.5, 余弦退火-init_lr=1e-5(fc*100， T_max=10))) | 93.60(145) -欠拟合  |
|                                                              |                     |
|                     以下input-size = 600                     |                     |
| Efficient-Net-B7(batch=4, dropout=0.5, 余弦退火-init_lr=1e-4(fc*10， T_max=10))) |    **92.62(34)**    |

***\*实验结果分析：\****

1. Efficient-Net是真的猛，尤其Efficient-Net-B4，达到了94.07%的准确率，对比resnet-50超出5.2个点。
2. 输入图像的大小太重要了。



#### 7.2**Efficient-Net**注意力模块对比实验

#### 实验日期(2.28-3.2)

|                            Model                             |  Test acc(%)   |
| :----------------------------------------------------------: | :------------: |
|                     以下input-size = 224                     |                |
| Efficient-Net-B4(batch=38, dropout=0.5, init_lr=1e-4(fc*10)) | **90.92(140)** |
| SE+Efficient-Net-B4(batch=23, dropout=0.5, init_lr=1e-4(fc*10)) | **90.68(73)**  |
| CBAM_sa+Efficient-Net-B4(batch=36, dropout=0.5, init_lr=1e-4(fc*10)) |   90.36(145)   |
| Fca+Efficient-Net-B4(batch=36, dropout=0.5, init_lr=1e-4(fc*10)) |   90.65(148)   |

***\*实验结果分析：\****

1. 老样子，加了注意力模块一坨屎。



### 8：RepVgg

#### 实验日期(3.2-3.5)

#### （baseline调参实验）

|                            Model                             |       Test acc(%)       |
| :----------------------------------------------------------: | :---------------------: |
|    RepVgg-B0(batch=48, dropout=0.5, init_lr=1e-4(fc*10))     |        84.43(51)        |
|    RepVgg-B1(batch=64, dropout=0.5, init_lr=1e-4(fc*10))     |        88.65(27)        |
|    RepVgg-B1(batch=64, dropout=0.5, init_lr=1e-5(fc*10))     |   88.48(144)- 欠拟合    |
|    RepVgg-B1(batch=64, dropout=0.5, init_lr=1e-5(fc*100))    |      **88.92(97)**      |
|                                                              |                         |
| RepVgg-B1(batch=64, dropout=0.5, init_lr=1e-4(fc*10)) + SENet(last layer) |       88.73(103)        |
| RepVgg-B1(batch=64, dropout=0.5, init_lr=1e-4(fc*10)) + SENet(layer 1-4) |   86.41(143) - 欠拟合   |
| RepVgg-B1(batch=64, dropout=0.5, init_lr=1e-5(fc*100)) + SENet(layer 1-4) | **88.80(146)** - 欠拟合 |

***\*实验结果分析：\****

1. RepVgg还蛮不错的，显存占用小，准确率也能达到resnet-50的baseline。





### 9：resnet-50上的疯狂调参实验(学习率、学习率衰减、添加attention后的学习率衰减)

#### 实验日期(3.3-3.10)

#### 9.1：resnet-50+SENet_Bottle_neck（调参实验）

|                         Model                          |     Test acc(%)     |
| :----------------------------------------------------: | :-----------------: |
|                    ( init_lr=1e-3)                     |      85.14(19)      |
|                    ( init_lr=1e-4)                     |      85.19(22)      |
|           (dropout=0.5, init_lr=1e-3(fc*10))           |      86.31(11)      |
|           (dropout=0.5, init_lr=1e-4(fc*10))           |     87.38(108)      |
|          (dropout=0.5, init_lr=1e-4(fc*100))           |      87.33(86)      |
|           (dropout=0.5, init_lr=1e-5(fc*10))           | 81.01(146) - 欠拟合 |
|          (dropout=0.5, init_lr=1e-5(fc*100))           | 85.90(147) - 欠拟合 |
| (dropout=0.5, init_lr=1e-4(fc*10)) + 每10epoch衰减0.94 |    **87.54(96)**    |
|        余弦退火-init_lr=1e-4(fc*10， T_max=10)         |      87.45(69)      |
|        余弦退火-init_lr=1e-5(fc*100， T_max=10)        | 86.71(143) - 欠拟合 |



#### 9.2：resnet-50（退火学习率调参实验）

|              Model              |     Test acc(%)     |
| :-----------------------------: | :-----------------: |
| init_lr=1e-2(fc*10， T_max=10)  | 75.28(137) - 过拟合 |
| init_lr=1e-3(fc*10， T_max=10)  | 85.78(107) - 过拟合 |
| init_lr=1e-4(fc*10， T_max=10)  |      88.78(54)      |
| init_lr=1e-5(fc*10， T_max=10)  |     87.67(140)      |
| init_lr=1e-5(fc*100， T_max=10) |   **88.89(108)**    |
| init_lr=1e-5(fc*100， T_max=5)  |     88.74(139)      |



#### 9.3：resnet-50+SENet_Last_Layer（调参实验）

|                        Model                         |        Test acc(%)        |
| :--------------------------------------------------: | :-----------------------: |
|       余弦退火-init_lr=1e-4(fc*10， T_max=10)        |       **88.28(61)**       |
|       余弦退火-init_lr=1e-5(fc*100， T_max=10)       |   88.75(144) - 略欠拟合   |
|  余弦退火-init_lr=1e-4(se x 10, fc x 10， T_max=10)  |    88.89(44) - 较稳定     |
| 余弦退火-init_lr=1e-5(se x 10, fc x 100， T_max=10)  | **88.94(115) - 略欠拟合** |
| 余弦退火-init_lr=1e-5(se x 100, fc x 100， T_max=10) |        88.82(145)         |



#### 9.4 使用调参总结后的预训练方式（余弦退火：init_lr=1e-5(attention x 10, fc x 100， T_max=10)）

|         Model         |  Test acc(%)   |
| :-------------------: | :------------: |
| Resnet-50（Baseline） |   88.89(108)   |
|        +SENet         | **88.94(115)** |
|        +EcaNet        |   88.39(74)    |
|        +FcaNet        |   88.38(107)   |
|    +CBAM(spatial)     |   87.96(141)   |

***\*实验结果分析：\****

1. 余弦退火感觉还是蛮好的，后期有上升空间。





### 10：改进PMG实验（余弦退火学习率）

#### 实验日期(3.10-3.18)

#### 10.1：仅concat层训练的调参实验

|                            Model                             |     Test acc(%)     |
| :----------------------------------------------------------: | :-----------------: |
|   Resnet-50（**Baseline**）init_lr=1e-5(fc*100， T_max=10)   |   **88.75**(144)    |
|              L3+L4+L5 —— init_lr=1e-4(T_max=10)              |    **87.95**(70)    |
|          L3+L4+L5 —— init_lr=1e-4(pmg*10，T_max=10)          |  87.33(27)- 过拟合  |
|              L3+L4+L5 —— init_lr=1e-5(T_max=10)              | 86.55(125) -欠拟合  |
|          L3+L4+L5 —— init_lr=1e-5(pmg*10，T_max=10)          | 87.62(141) - 过拟合 |
|         L3+L4+L5 —— init_lr=1e-5(pmg*100，T_max=10)          | 86.39(19) - 过拟合  |
| input_size * 2 (448) L3+L4+L5 —— init_lr=1e-4(pmg*10，T_max=10) | 87.59(55) - 过拟合  |

|                          Model                           |      Test acc(%)       |
| :------------------------------------------------------: | :--------------------: |
| Resnet-50（**Baseline**）init_lr=1e-5(fc*100， T_max=10) |     **88.75**(144)     |
|      L3+L4+L5 —— init_lr=1e-4(**fc***10，T_max=10)       | **87.37**(27) - 过拟合 |
|     L3+L4+L5+SENet —— init_lr=1e-4(fc*10，T_max=10)      |   86.88(8) - 过拟合    |

|                            Model                             |       Test acc(%)       |
| :----------------------------------------------------------: | :---------------------: |
|          L3+L4+L5 —— init_lr=1e-4(fc*10，T_max=10)           |      **87.95**(70)      |
|             L3 —— init_lr=1e-4(pmg*10，T_max=10)             |  62.28(126) -  过拟合   |
|             L4 —— init_lr=1e-4(pmg*10，T_max=10)             |   78.78(136) - 过拟合   |
|             L5 —— init_lr=1e-4(pmg*10，T_max=10)             | **87.41**(145) - 过拟合 |
| 0.1 * L3+ 0.2 * L4+ 0.7 * L5 —— init_lr=1e-4(fc*10，T_max=10) | **87.56**(21) - 过拟合  |



#### 10.2 以下去处额外CONV层，单层对比实验

|                          Model                           |      Test acc(%)       |
| :------------------------------------------------------: | :--------------------: |
| Resnet-50（**Baseline**）init_lr=1e-5(fc*100， T_max=10) |     **88.75**(144)     |
|        L3 —maxpool— init_lr=1e-4(fc*10，T_max=10)        | 19.85(66) - 严重欠拟合 |
|        L4 —maxpool— init_lr=1e-4(fc*10，T_max=10)        | 71.70(77) - 严重欠拟合 |
|        L5 —maxpool— init_lr=1e-4(fc*10，T_max=10)        |       87.39(57)        |
|        L5 —avgpool— init_lr=1e-4(fc*10，T_max=10)        |     **88.71**(25)      |



#### 10.3 完全复现实验

|         Model(input 448) - 4损失函数         |  Test acc(%)   |
| :------------------------------------------: | :------------: |
|               resnet(Baseline)               | **88.75**(144) |
| ECCV-PMG- 完全一模一样复现(<u>Baseline2</u>) |   83.58(24)    |

***\*实验结果分析：\****

1. 很奇怪，似乎Dogs数据集更适合224作为输入。
2. 先认真看论文，否则容易做很多无意义的探索。





### 11：MKI-V3（余弦退火学习率）

#### 实验日期(4.2-4.3)

#### 11.1：仅concat层训练的调参实验

|           Model           | Test acc(%) |
| :-----------------------: | :---------: |
| Resnet-50（**Baseline**） | 88.78(144)  |
|     MKI-V3(no concat)     |  86.85(9)   |
|          MKI-V3           |  87.70(38)  |
|                           |             |
|                           |             |
|                           |             |
|                           |             |

