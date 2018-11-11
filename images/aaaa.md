# Enet

运行命令
======================================

训练:

python train_jingjian.py 

tensorboard --logdir=./tboard


继续训练:

python train_jingjian.py --weights_path model/culane_lanenet/culane_lanenet_enet_2018-10-30-10-18-10.ckpt-3800

测试
python test_jingjian.py  --dataset_dir ./data --weights_path model/culane_lanenet/culane_lanenet_enet_2018-10-29-16-18-40.ckpt-7000

进展：
==========================================

1.找到最小的精简Enet结构

![](https://alioss.tower.im/712512%2Fc71c7518b95b4221a660d38a44b000df_large?Expires=1541035392&OSSAccessKeyId=NCnAvvXb096Ats57&Signature=UCOlJuQqqH6YkOu4iFuHS%2Bjli0E%3D&response-content-disposition=inline%3Bfilename%3D%22%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20181030173142.jpg%22&response-content-type=image%2Fjpeg)

2.尝试找到合适的结构（未完成）
   
   a. 11-05-10-15-10 : E2 保留所有空洞卷积

3.尝试将bottleneck换成shufflenet（完成并测试中）：模型大小38MB,精简后11MB

4.可加SENet（但是还没加，代码在下面找，因为在训练模型，没有GPU跑）

5.现在可能是因为学习率过小，导致模型精确度上升较慢

6.当模型难训练时，如何提升训练速度：

a、先用最快的优化函数如RMSprop

b、当训练差不多时，再用收敛效果最好的优化函数：如adm


下一步方向：
===========================================

更快

更小

更准


one shot 语义分割？？？？？？？

[](http://www.vision.ee.ethz.ch/~cvlsegmentation/osvos/)

[](https://blog.csdn.net/zdyueguanyun/article/details/78204802)

OSVOS？？？？？？？？？？？
[项目代码](https://github.com/scaelles/OSVOS-TensorFlow)

deeplab???????????????????


video segmentation:
[video segmentation综述](https://zhuanlan.zhihu.com/p/32247505)

[论文《Efficient Video Object Segmentation via Network Modulation》](https://zhuanlan.zhihu.com/p/36139460)

网络结构的优化:

![](https://pic3.zhimg.com/v2-42a00ee136b4ebca8c56408858d74a16_b.jpg)

卷积运算方式,bottleneck设计,添加或删除不同的层

shuffle net V2

![](https://img-blog.csdn.net/20180731224347374?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQzODAxNjU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

channel shuffle
```
def channel_shuffle(name, x, num_groups):
    with tf.variable_scope(name) as scope:
        n, h, w, c = x.shape.as_list()
        x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
        x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
        output = tf.reshape(x_transposed, [-1, h, w, c])
        return output
```
depthwise_conv2d
```
tf.nn.depthwise_conv2d(input,filter,strides,padding,rate=None,name=None,data_format=None)
```

砍层


mobilenet V2


attention 机制

IBN模块


BiSeNet

![](http://5b0988e595225.cdn.sohucs.com/images/20180807/c46b74b61684414db59a8e8923232960.jpeg)

简单来讲就是空间分支加上内容分支

SENet


shufflenet V2 

时间降到20ms

模型控制在20mb以下

四个指导方针：

1. 卷积核数量尽量与输入通道数相同（即输入通道数等于输出通道数）；

2. 谨慎使用group convolutions，注意group convolutions的效率；

3. 降低网络碎片化程度；

4. 减少元素级运算

现在在做实验,将浅层的Enet转化为深层的encoder,用relu,效果还可以

下一步想将每个浅层的网络赋予权重,并训练



研究资料:
----------------------

[深度学习各种笔记](https://sgfin.github.io/learning-resources/)

[CNN网络优化学习总结——从MobileNet到ShuffleNet](https://blog.csdn.net/sun_28/article/details/78170878)

[各种卷积总结](https://github.com/vdumoulin/conv_arithmetic)

![](https://pic4.zhimg.com/80/v2-f3c061509b3b9e96f5fc400a9ea26522_hd.jpg)

![](https://pic2.zhimg.com/80/v2-15e27880d3b1c562af7b4080dfac3739_hd.jpg)

[卷积神经网络结构演变（form Hubel and Wiesel to SENet）——学习总结，文末附参考论文](https://zhuanlan.zhihu.com/p/34621135)

高效模型设计： 

CNNs在CV任务中取得了极大的成功，在嵌入式设备上运行高质量深度神经网络需求越来越大，这也促进了对高效模型的探究。

例如，与单纯的堆叠卷积层，GoogleNet增加了网络的宽度，复杂度降低很多；

SqueezeNet在保持精度的同时大大减少参数和计算量；

ResNet利用高效的bottleneck结构实现惊人的效果。Xception中提出深度可分卷积概括了Inception序列。

MobileNet利用深度可分卷积构建的轻量级模型获得了先进的成果；

ShuffleNet的工作是推广群卷积(group convolution)和深度可分卷积(depthwise separable convolution)。

![](https://pic4.zhimg.com/80/v2-9d9bdea7ca7039165ca875d774aefa4e_hd.jpg)




[轻量化网络ShuffleNet MobileNet v1/v2 解析](https://zhuanlan.zhihu.com/p/35405071)


通过这几篇论文的创新点，得出以下可认为是发 (Shui) 论文的 idea：

1. 采用 depth-wise convolution，再设计一个方法解决「信息流通不畅」问题，然后冠以美名 XX-Net。（看看 ShuffleNet 就是）

2. 针对 depth-wise convolution 作文章，卷积方式不是千奇百怪么？各种卷积方式可参考 Github（https://github.com/vdumoulin/conv_arithmetic），挑一个或者几个，结合起来，只要参数量少，实验效果好，就可以发 (Shui) 论文。

3. 接着第 2，如果设计出来一个新的卷积方式，如果也存在一些「副作用」，再想一个方法解决这个副作用，再美其名曰 XX-Net。就是自己「挖」个坑，自己再填上去。

ResNet的核心：
一堆网络的集合，有浅有深

![](https://pic3.zhimg.com/v2-1081c92e6f695c20df034cff68e565dc_r.jpg)


从宽网络角度来分析 ResNet，的确解释了 [1] 中 lesion study，而且指明为什么 ResNet 在层数较深时能有较好的收敛率 —— 总有部分短路径远小于模型真正的深度。[2] 在 [1] 之后，进行了近一步的分析，指出带 Residual Connection 的深网络可以“近似地”看作宽模型，BUT，跟真正的宽模型还是不同的！Forward 时两者可以看做相同，但 backward 时有部分 gradient 路径无法联通。也就是说， ResNet 在回传 gradient 时，尚有提升的空间，这就是为什么 ResNeXt，Wide-ResNet 等文章能有提升的原因——因为 ResNet 并不是真正的宽模型 。

![](https://pic2.zhimg.com/80/v2-0e44e8acf72d30ab961ca8b5ecb23c6d_hd.png)

![](https://pic2.zhimg.com/80/v2-a37582245c4085c3e2df89b4c4f888ed_hd.png)
###### ResNext:

![](https://pic3.zhimg.com/80/v2-ee942d228efdaaff277c8a9a8b96a131_hd.jpg)

左边是ResNet的基本结构，右边是ResNeXt的基本结构

通过在通道上对输入进行拆分，进行分组卷积，每个卷积核不用扩展到所有通道，可以得到更多更轻量的卷积核，并且，卷积核之间减少了耦合，用相同的计算量，可以得到更高的精度。


###### Xception：

![](https://pic4.zhimg.com/80/v2-8bc91f4aa78e77119eaa960c93c0d3b0_hd.jpg)


！【】（https://img-blog.csdn.net/20180608211237818?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0pOaW5nV2Vp/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70）

简单来讲：就是先用1* 1 融合各通道，然后用3 × 3 来解析单个channel

这个模块就是先每个通道卷积再对所有通道的每个像素点卷积，简单来讲，就是先散开卷积，然后再深度卷积。


好处：减少计算力的浪费

缺点：底层支持并不太好

Xception基于一个假设，水平和竖直方向的空间卷积（比如第一步的3x3卷积）和深度方向的通道卷积（比如第二步的1x1卷积）可以完全独立进行，这样减少了不同操作间的耦合，可以有效利用计算力。实验证明，相同的计算量，精度有明显的提升。（不过现在对于分组卷积的底层支持还不够好，实际速度并没有理论计算的那么好，需要底层库进行更好的支持）


###### shuffleNet:

[轻量级网络--ShuffleNet论文解读](https://blog.csdn.net/u011974639/article/details/79200559)

核心：
featuremap通道分组进行卷积，减少计算量。

![](https://pic3.zhimg.com/80/v2-c64110d20c8204affb88d79ec67b75f7_hd.jpg)

![](http://owv7la1di.bkt.clouddn.com/blog/180129/lj9IK1cm9H.png)

![](https://pic2.zhimg.com/80/v2-de3847209089b598bf2a227924f77f5f_hd.jpg)

上面的a是普通的ResNet结构，而b是ShuffleNet结构，可以看到，具体实现上，是先经过1x1通道分组卷积，然后通道重排，再3x3空间上的depthwise convolution，然后再经过一个1x1的通道分组卷积。图c是要进行降维时候的结构，stride为2，旁边的shortcut也变为stride 2的平均池化。

![](https://pic1.zhimg.com/80/v2-5a81d1bb186b32aad32b638b26a17043_hd.jpg)

###### SENET

[项目代码-tensorflow](https://github.com/taki0112/SENet-Tensorflow)

[项目代码](https://github.com/hujie-frank/SENet)

将featuremaps各通道赋予权重并训练

SENet的Squeeze-Excitation模块在普通的卷积（单层卷积或复合卷积）由输入X得到输出U以后，对U的每个通道进行全局平均池化得到通道描述子（Squeeze），再利用两层FC得到每个通道的权重值，对U按通道进行重新加权得到最终输出（Excitation），这个过程称之为feature recalibration，通过引入attention重新加权，可以得到抑制无效特征，提升有效特征的权重，并很容易地和现有网络结合，提升现有网络性能，而计算量不会增加太多。

![](https://pic2.zhimg.com/v2-e02ddf911b562e18f8b6edf527c8fb75_b.jpg)

![](https://pic1.zhimg.com/80/v2-65edde2384540885d71b94de135b50e4_hd.jpg)

···
def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :
        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])

        scale = input_x * excitation

        return scale
···

###### BiSENET

[论文阅读：BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://www.jianshu.com/p/877a3f59f483?open_source=weibo_search)

[论文](https://arxiv.org/abs/1808.00897v1)

###### IBN-Net

[批量标准化(BN)、实例标准化(IN)、特征标准化(FN)](https://blog.csdn.net/mzpmzk/article/details/80043076)

[BN与IN的对比](https://www.zhihu.com/question/68730628/answer/277339783)


[IBN-Net论文笔记](https://blog.csdn.net/sunyao_123/article/details/81294724)


batch norm是对一个batch里所有的图片的所有像素求均值和标准差。而instance norm是对单个图片的所有像素求均值和标准差。

IBN-Net能够学习捕获和消除外观的变换，而保持学到的特征的区别性。 

浅层的外观不同的feature divergence较大，到深层就变小；而内容不同的feature divergence在浅层较小，在深层较大。所以浅层使用IN，深层使用BN，但是由于BN的巨大的作用，所以浅层也保留使用BN。

IBN 模块：消除图像的风格信息，仅保留其内容信息


为了充分发挥IN和BN的各自优势，作者提出了IBN-Net的两条基本构建原则：1、为了防止网络在高层的语义判别性被破坏，IN只加在网络低层中；2、为了保留低层中的语义信息，网络低层中也保留BN。根据这两条原则，作者提出了如下两个IBN block：

![](https://pic4.zhimg.com/80/v2-75e65684cf5ca97381dea76cef27ec45_hd.jpg)

为何如此设计，作者给出了三点解释：（1）在ResNet的原论文中已经证明，identity path不加任何东西更有利于优化ResNet，所以IN不应该加在identity path中；（2）IN置于第一个normalization层，是为了保证与identity path的结果的一致性；（3）在第一个normalization层，IN和BN各占一半的通道数，一方面保证了不增加网络参数和计算量，另一方面也是为了在IN过滤掉反映外观变化的信息的同时用BN保留语义信息。此外，作者还展示了在实验中用到的其它几个IBN block，其核心思想依然符合上述两个设计原则，在此不再赘述。

###### Mobilenet

核心就是先分层卷积，然后在用1* 1卷积进行深度卷积，减少运算量。

![](https://pic1.zhimg.com/v2-2f939c1fbb6ba6a10a38b599223a002c_b.jpg)

![](https://pic3.zhimg.com/v2-2fb755fbd24722bcb35f2d0d291cee22_b.jpg)

###### MobileNet V2

![](https://pic1.zhimg.com/80/v2-25b6c783dbb5412119200696f02f3018_hd.jpg)

![](http://mmbiz.qpic.cn/mmbiz_png/a9UoojghtAECXTQ4RVhEVLLYcOuvmpyBhME3SBibkuxBJAUqj4yOzupiapPmEALebR0o4txC9kKTnHpib3Qpg4WVQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)


Enet论文阅读
======================================


[Semantic Segmentation--ENet:A Deep Neural Network Architecture for Real-Time Semantic..论文解读](https://blog.csdn.net/u011974639/article/details/78956380)

![](https://pic2.zhimg.com/80/v2-b7f8d494d2777b64a999faff07effe4e_hd.jpg)

![](https://pic4.zhimg.com/80/v2-870fd241940d9a8d2cb1d82537f37778_hd.jpg)


因为此代码是基于lanenet网络改进的，所以需要先了解lanenet网络的一些基本参数：


![](https://pic3.zhimg.com/80/v2-0a7016d069c25a1aa65384349d9d24ee_hd.png)

![](http://owv7la1di.bkt.clouddn.com/blog/180102/Hb10AFHcJI.png)


精确率与召回率等预测准确度计算:
------------------------------------------------

TP: true positive ;

TN: true negative;

FP: false positive;

FN: false negative;

这四个参数的误差用于计算许多种性能评估指标，用来评价分类器的性能（classifier）

![](https://pic3.zhimg.com/80/v2-d9875e0306dae16aaea850afe9636745_hd.jpg)

[如何解释召回率与准确率？](https://www.zhihu.com/question/19645541)

![](https://pic3.zhimg.com/80/d701da76199148837cfed83901cea99e_hd.png)

![](https://pic1.zhimg.com/80/v2-459293ee827f3467fab1179742b2a188_hd.jpg)

lossfunction：
------------------------------------------------

[TensorFlow四种Cross Entropy算法实现和应用 ](http://dataunion.org/26447.html)


交叉熵损失函数：

二分类

在二分的情况下，模型最后需要预测的结果只有两种情况，对于每个类别我们的预测得到的概率为p和1-p。此时表达式为：

![](http://www.zhihu.com/equation?tex=L%3D-%5Bylog%5C+%5Chat+y%2B%281-y%29log%5C+%281-%5Chat+y%29%5D)

其中：

- y——表示样本的label，正类为1，负类为0

- p——表示样本预测为正的概率

多分类
多分类的情况实际上就是对二分类的扩展：

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7DJ+%3D+-%5Csum_%7Bc%3D1%7D%5EMy_%7Bc%7D%5Clog%28p_%7Bc%7D%29%5Cend%7Balign%7D+%5C%5C)

其中：
- M——类别的数量；
- y——指示变量（0或1）,如果该类别和样本的类别相同就是1，否则是0；
- p——对于观测样本属于类别c的预测概率。


使用交叉熵损失函数，不仅可以很好的衡量模型的效果，又可以很容易的的进行求导计算。

优缺点

优点

在用梯度下降法做参数更新的时候，模型学习的速度取决于两个值：

一、学习率；

二、偏导值。

其中，学习率是我们需要设置的超参数，所以我们重点关注偏导值。从上面的式子中，我们发现，偏导值的大小取决于 x_i 和 [sigma(y_i)-y_i] ，我们重点关注后者，后者的大小值反映了我们模型的错误程度，该值越大，说明模型效果越差，但是该值越大同时也会使得偏导值越大，从而模型学习速度更快。所以，使用逻辑函数得到概率，并结合交叉熵当损失函数时，在模型效果差的时候学习速度比较快，在模型效果好的时候学习速度变慢。

空洞卷积
------------------------------------

空洞卷积:既不缩小数据大小,还能够增大其感受野,之前只有downpooling的方法

[Tensorflow】tf.nn.atrous_conv2d如何实现空洞卷积？](https://blog.csdn.net/mao_xiao_feng/article/details/78003730)

tf.nn.atrous_conv2d(value,filters,rate,padding,name=None）

除去name参数用以指定该操作的name，与方法有关的一共四个参数：

    value：
    指需要做卷积的输入图像，要求是一个4维Tensor，具有[batch, height, width, channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]

    filters：
    相当于CNN中的卷积核，要求是一个4维Tensor，具有[filter_height, filter_width, channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，同理这里第三维channels，就是参数value的第四维

    rate：
    要求是一个int型的正数，正常的卷积操作应该会有stride（即卷积核的滑动步长），但是空洞卷积是没有stride参数的，这一点尤其要注意。取而代之，它使用了新的rate参数，那么rate参数有什么用呢？它定义为我们在输入图像上卷积时的采样间隔，你可以理解为卷积核当中穿插了（rate-1）数量的“0”，把原来的卷积核插出了很多“洞洞”，这样做卷积时就相当于对原图像的采样间隔变大了。具体怎么插得，可以看后面更加详细的描述。此时我们很容易得出rate=1时，就没有0插入，此时这个函数就变成了普通卷积。

    padding：
    string类型的量，只能是”SAME”,”VALID”其中之一，这个值决定了不同边缘填充方式。


rate = 2 的 3*3空洞卷积

![](https://upload-images.jianshu.io/upload_images/207577-4ba7cf60bf5476f5.gif?imageMogr2/auto-orient/strip%7CimageView2/2/w/395)

非对称卷积Asymmetric Convolutions:
-------------------------------

![](https://pic3.zhimg.com/v2-e03eb40cd8d82ad40e943ad26644fc5a_b.jpg)

![](https://pic2.zhimg.com/80/v2-f0b49e7c24119ae8b4f5dd6d9170bf05_hd.jpg)

[为什么非对称卷积（Asymmetric Convolution）减少了运算量?](https://www.zhihu.com/question/270055683)

![](https://github.com/greenfishflying/Enet/blob/master/images/%E9%9D%9E%E5%AF%B9%E7%A7%B0%E5%8D%B7%E7%A7%AF%E8%AE%A1%E7%AE%97.jpg)

```
import tensorflow as tf
x = tf.Variable(tf.ones([1, 4, 4, 1]))
w = tf.Variable(tf.ones([3, 1, 1, 1]))
w2 = tf.Variable(tf.ones([1, 3, 1, 1]))
w3 = tf.Variable(tf.ones([3, 3, 1, 1]))
output = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
output2 = tf.nn.conv2d(output, w2, strides=[1, 1, 1, 1], padding='SAME')
output3 = tf.nn.conv2d(x, w3, strides=[1, 1, 1, 1], padding='SAME')
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print("output",sess.run(output))
print("output2",sess.run(output2))
print("output3",sess.run(output3))
print("w",sess.run(w))
print("w2",sess.run(w2))
print("w3",sess.run(w3))
```

pool层(效果等同与步长为2的卷积)
--------------------------------

[Pool层及其公式推导](https://blog.csdn.net/qq_29381089/article/details/80688255)

[卷积神经网络_（1）卷积层和池化层学习](https://www.cnblogs.com/zf-blog/p/6075286.html)


[pooling层汇总](https://mp.weixin.qq.com/s/ISvHyUrXpxGTCMVib-ptnw)


pooling层作用，可以增加卷积感受野，减少参数。


downsampling方法常用：Maxpooling + conv

新方法：空洞卷积，好处，不会丢弃feature。



![](https://images2015.cnblogs.com/blog/1062917/201611/1062917-20161117195428888-895158719.png)

strides=[1, 1, 1, 1]参数解释:

conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')

这是一个常见的卷积操作，其中strides=【1,1,1,1】表示滑动步长为1，padding=‘SAME’表示填0操作

当我们要设置步长为2时，strides=【1,2,2,1】，很多同学可能不理解了，这四个参数分别代表了什么，查了官方函数说明一样不明不白，今天我来解释一下。

strides在官方定义中是一个一维具有四个元素的张量，其规定前后必须为1，这点大家就别纠结了，所以我们可以改的是中间两个数，中间两个数分别代表了水平滑动和垂直滑动步长值，于是就很好理解了。

在卷积核移动逐渐扫描整体图时候，因为步长的设置问题，可能导致剩下未扫描的空间不足以提供给卷积核的，大小扫描 比如有图大小为5*5,卷积核为2*2,步长为2,卷积核扫描了两次后，剩下一个元素，不够卷积核扫描了，这个时候就在后面补零，补完后满足卷积核的扫描，这种方式就是same。如果说把刚才不足以扫描的元素位置抛弃掉，就是valid方式。


![](https://images2015.cnblogs.com/blog/1062917/201611/1062917-20161117211920029-1784506227.png)


MaxPooling:

Maxpooling 同时会传出一个最大值位置的矩阵，为了Maxunpooling的运算

![](https://mmbiz.qpic.cn/mmbiz_png/a9UoojghtAGcVGUrTe7nibyyxaeLgwuCohbsSiamwSRGU75owtpyTczsHiakZYFHsYw6CibhflqGBh8doaiamXOtYIQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Average Pooling：

将每个窗口计算其平均值，反向传播时均匀分配到每个元素。

Global Average Pooling：

[global pooling 详解](https://zhuanlan.zhihu.com/p/37683646)

实现方法，就是用平均pooling但是把核的大小改成featuremap的长宽。

简单讲：就是将featuremaps 每层求平均，这样减少参数量，可以防止过拟合。

一般用在全连接层前

unpooling:
---------------------------------

左边为pooling 右边为unpooling

![](https://pic4.zhimg.com/v2-d39bbce090ad22a219b0e6b91953cd7b_b.png)

![](https://img-blog.csdn.net/20180127154813206?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQV9hX3Jvbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

unsampling上采样:
-------------------------

[反卷积(Deconvolution)、上采样(UNSampling)与上池化(UnPooling)](https://blog.csdn.net/A_a_ron/article/details/79181108?utm_source=blogxgwz0)

![](https://img-blog.csdn.net/20180127154813206?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQV9hX3Jvbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


反卷积:
----------------------------

[理解tf.nn.conv2d和tf.nn.conv2d_transpose](https://www.jianshu.com/p/a897ed29a8a0)

[](https://blog.csdn.net/panglinzhuo/article/details/75207855)

![](http://deeplearning.net/software/theano_versions/dev/_images/same_padding_no_strides.gif)

代码中:
-------------------

##### [lanenet]

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=decode_logits, labels=tf.squeeze(label, axis=[3]),
                name='entropy_loss')
                
[TensorFlow关于tf.nn.sparse_softmax_cross_entropy_with_logits（）](https://blog.csdn.net/ZJRN1027/article/details/80199248)

[	tf.nn.softmax_cross_entropy_with_logits中的“logits”到底是个什么意思？](https://blog.csdn.net/yhily2008/article/details/80262321)

参数logits，logit本身就是是一种函数，它把某个概率p从[0,1]映射到[-inf,+inf]（即正负无穷区间）。这个函数的形式化描述为：logit=ln(p/(1-p))。
我们可以把logist理解为原生态的、未经缩放的，可视为一种未归一化的log 概率，如是[4, 1, -2],简单来讲,就是得到预测的概率每归一化


第一步：Softmax:

输出每个类占总类的概率

![](https://pic4.zhimg.com/75938cc54604077d2ed193e97a5302bb_b.jpg)

[Softmax 函数的特点和作用是什么？ - 忆臻的回答 - 知乎](https://www.zhihu.com/question/23765351/answer/240869755)



![](https://img-blog.csdn.net/20180504192857468?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1pKUk4xMDI3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

第二步：计算Cross-Entropy:

神经网络的输出层经过Softmax函数作用后，接下来就要计算我们的loss了，这个这里是使用了Cross-Entropy作为了loss function。由于tf.nn.sparse_softmax_cross_entropy_with_logits()输入的label格式为一维的向量，所以首先需要将其转化为one-hot格式的编码，例如如果分量为3，代表该样本属于第三类，其对应的one-hot格式label为[0，0，0，1，.......0]，而如果你的label已经是one-hot格式，则可以使用tf.nn.softmax_cross_entropy_with_logits()函数来进行softmax和loss的计算。


![](https://img-blog.csdn.net/20180504201531230?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1pKUk4xMDI3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

##### [Enet]

loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, weights=weights)



输出道路点的位置在:

out_logits = compute_ret['binary_seg_logits'] #判断为道路的点,shape(4,256,512,2)

out_logitss = tf.nn.softmax(logits=out_logits)#进行softmax变换,得到每个点的是道路和背景的概率,shape(4,256,512,2)

out_logits_out = tf.argmax(out_logitss, axis=-1)#shape(4,256,512,1) [最后一位表示最后找到的最大的标签的位置是0 还是1 如果是0 就是背景,如果是1就是道路,正好会显示为二值图]

print('out_logits',out_logits[0][0][0])

print('out_logitss',out_logitss[0][0][0])

print('out_logits_out',binary_seg_img[0][0][0])


输出:

out_logits [0.1682797 0.346079 ]

out_logitss [0.4556669 0.5443331]

out_logits_out 1




[tf.argmax()以及axis解析](https://blog.csdn.net/qq575379110/article/details/70538051)

按行或列输出矩阵最大值的坐标,0是按列,1是按行,

##### -1 代表什么???????????????????

个人猜测,找最后一个维度的最大的标签

###### 代码阅读

整个网络的参数变化:

```
build modelllllllll inggggggggggggggggggggggggg
################ initial block
===============initial block
======x [4, 256, 512, 3]
======conv_branch [4, 128, 256, 13]
======pool_branch [4, 128, 256, 3]#行列各缩一倍，但是深度不变
======concat [4, 128, 256, 16]
======output [4, 128, 256, 16]
===================================
initial_output = [4, 128, 256, 16]
################ Encoder_1
inputs_stage_1 = [4, 128, 256, 16]
===============encoder_bottleneck_regular
======input_shape [4, 128, 256, 16]
======conv_branch (downsampling)  [4, 64, 128, 4]#利用4个卷积，每个卷积为[2*2*16]（长，宽，深度），故，输出4个featuremaps。
======convolution [4, 64, 128, 4]
======1x1 expansion [4, 64, 128, 64]
======main_branch(downsampling) [4, 64, 128, 64]
======add the branchs [4, 64, 128, 64]
======output [4, 64, 128, 64]
output_1st_downsample = [4, 64, 128, 64]
pooling_indeces_1 = [4, 64, 128, 16]
===============encoder_bottleneck_regular
======input_shape [4, 64, 128, 64]
======conv_branch [4, 64, 128, 16]
======convolution [4, 64, 128, 16]
======1x1 expansion [4, 64, 128, 64]
======add the branchs [4, 64, 128, 64]
======output [4, 64, 128, 64]
===============encoder_bottleneck_regular
======input_shape [4, 64, 128, 64]
======conv_branch [4, 64, 128, 16]
======convolution [4, 64, 128, 16]
======1x1 expansion [4, 64, 128, 64]
======add the branchs [4, 64, 128, 64]
======output [4, 64, 128, 64]
===============encoder_bottleneck_regular
======input_shape [4, 64, 128, 64]
======conv_branch [4, 64, 128, 16]
======convolution [4, 64, 128, 16]
======1x1 expansion [4, 64, 128, 64]
======add the branchs [4, 64, 128, 64]
======output [4, 64, 128, 64]
===============encoder_bottleneck_regular
======input_shape [4, 64, 128, 64]
======conv_branch [4, 64, 128, 16]
======convolution [4, 64, 128, 16]
======1x1 expansion [4, 64, 128, 64]
======add the branchs [4, 64, 128, 64]
======output [4, 64, 128, 64]
################ Encoder_2
inputs_stage_2 = [4, 64, 128, 64]
===============encoder_bottleneck_regular
======input_shape [4, 64, 128, 64]
======conv_branch (downsampling)  [4, 32, 64, 16]
======convolution [4, 32, 64, 16]
======1x1 expansion [4, 32, 64, 128]
======main_branch(downsampling) [4, 32, 64, 128]
======add the branchs [4, 32, 64, 128]
======output [4, 32, 64, 128]
output_2nd_downsample = [4, 32, 64, 128]
pooling_indices_2 = [4, 32, 64, 64]
===============encoder_bottleneck_regular
======input_shape [4, 32, 64, 128]
======conv_branch [4, 32, 64, 32]
======convolution [4, 32, 64, 32]
======1x1 expansion [4, 32, 64, 128]
======add the branchs [4, 32, 64, 128]
======output [4, 32, 64, 128]
encoder_bottleneck_dilated
======input_shape [4, 32, 64, 128]
======1x1 projection [4, 32, 64, 32]
======dilated conv [4, 32, 64, 32]
======1x1 projection [4, 32, 64, 128]
======main branch [4, 32, 64, 128]
======merge the branches [4, 32, 64, 128]
======output [4, 32, 64, 128]
encoder_bottleneck_asymmetric
======input_shape [4, 32, 64, 128]
======1x1 projection [4, 32, 64, 32]
======asymmetric conv [4, 32, 64, 32]
======asymmetric conv2 [4, 32, 64, 32]
======1x1 projection [4, 32, 64, 128]
======merge the branches [4, 32, 64, 128]
======output [4, 32, 64, 128]
encoder_bottleneck_dilated
======input_shape [4, 32, 64, 128]
======1x1 projection [4, 32, 64, 32]
======dilated conv [4, 32, 64, 32]
======1x1 projection [4, 32, 64, 128]
======main branch [4, 32, 64, 128]
======merge the branches [4, 32, 64, 128]
======output [4, 32, 64, 128]
===============encoder_bottleneck_regular
======input_shape [4, 32, 64, 128]
======conv_branch [4, 32, 64, 32]
======convolution [4, 32, 64, 32]
======1x1 expansion [4, 32, 64, 128]
======add the branchs [4, 32, 64, 128]
======output [4, 32, 64, 128]
encoder_bottleneck_dilated
======input_shape [4, 32, 64, 128]
======1x1 projection [4, 32, 64, 32]
======dilated conv [4, 32, 64, 32]
======1x1 projection [4, 32, 64, 128]
======main branch [4, 32, 64, 128]
======merge the branches [4, 32, 64, 128]
======output [4, 32, 64, 128]
encoder_bottleneck_asymmetric
======input_shape [4, 32, 64, 128]
======1x1 projection [4, 32, 64, 32]
======asymmetric conv [4, 32, 64, 32]
======asymmetric conv2 [4, 32, 64, 32]
======1x1 projection [4, 32, 64, 128]
======merge the branches [4, 32, 64, 128]
======output [4, 32, 64, 128]
encoder_bottleneck_dilated
======input_shape [4, 32, 64, 128]
======1x1 projection [4, 32, 64, 32]
======dilated conv [4, 32, 64, 32]
======1x1 projection [4, 32, 64, 128]
======main branch [4, 32, 64, 128]
======merge the branches [4, 32, 64, 128]
======output [4, 32, 64, 128]
####### Encoder_3_seg
===============encoder_bottleneck_regular
======input_shape [4, 32, 64, 128]
======conv_branch [4, 32, 64, 32]
======convolution [4, 32, 64, 32]
======1x1 expansion [4, 32, 64, 128]
======add the branchs [4, 32, 64, 128]
======output [4, 32, 64, 128]
encoder_bottleneck_dilated
======input_shape [4, 32, 64, 128]
======1x1 projection [4, 32, 64, 32]
======dilated conv [4, 32, 64, 32]
======1x1 projection [4, 32, 64, 128]
======main branch [4, 32, 64, 128]
======merge the branches [4, 32, 64, 128]
======output [4, 32, 64, 128]
encoder_bottleneck_asymmetric
======input_shape [4, 32, 64, 128]
======1x1 projection [4, 32, 64, 32]
======asymmetric conv [4, 32, 64, 32]
======asymmetric conv2 [4, 32, 64, 32]
======1x1 projection [4, 32, 64, 128]
======merge the branches [4, 32, 64, 128]
======output [4, 32, 64, 128]
encoder_bottleneck_dilated
======input_shape [4, 32, 64, 128]
======1x1 projection [4, 32, 64, 32]
======dilated conv [4, 32, 64, 32]
======1x1 projection [4, 32, 64, 128]
======main branch [4, 32, 64, 128]
======merge the branches [4, 32, 64, 128]
======output [4, 32, 64, 128]
===============encoder_bottleneck_regular
======input_shape [4, 32, 64, 128]
======conv_branch [4, 32, 64, 32]
======convolution [4, 32, 64, 32]
======1x1 expansion [4, 32, 64, 128]
======add the branchs [4, 32, 64, 128]
======output [4, 32, 64, 128]
encoder_bottleneck_dilated
======input_shape [4, 32, 64, 128]
======1x1 projection [4, 32, 64, 32]
======dilated conv [4, 32, 64, 32]
======1x1 projection [4, 32, 64, 128]
======main branch [4, 32, 64, 128]
======merge the branches [4, 32, 64, 128]
======output [4, 32, 64, 128]
encoder_bottleneck_asymmetric
======input_shape [4, 32, 64, 128]
======1x1 projection [4, 32, 64, 32]
======asymmetric conv [4, 32, 64, 32]
======asymmetric conv2 [4, 32, 64, 32]
======1x1 projection [4, 32, 64, 128]
======merge the branches [4, 32, 64, 128]
======output [4, 32, 64, 128]
encoder_bottleneck_dilated
======input_shape [4, 32, 64, 128]
======1x1 projection [4, 32, 64, 32]
======dilated conv [4, 32, 64, 32]
======1x1 projection [4, 32, 64, 128]
======main branch [4, 32, 64, 128]
======merge the branches [4, 32, 64, 128]
======output [4, 32, 64, 128]
#######  # # Decoder_1_seg
decoder_bottleneck
======input_shape [4, 32, 64, 128]
======# 1x1 projection (upsampling) [4, 32, 64, 64]
======# max unpooling [4, 64, 128, 64]
======# conv branch  [4, 32, 64, 128]
======# # 1x1 projection  [4, 32, 64, 32]
======# # conv upsampling  [4, 64, 128, 32]
====== # # 1x1 expansion   [4, 64, 128, 64]
====== # #add the batches   [4, 64, 128, 64]
====== # # output   [4, 64, 128, 64]
decoder_bottleneck
======input_shape [4, 64, 128, 64]
======# conv branch  [4, 64, 128, 64]
======# # 1x1 projection  [4, 64, 128, 16]
======# # conv   [4, 64, 128, 16]
====== # # 1x1 expansion   [4, 64, 128, 64]
====== # #add the batches   [4, 64, 128, 64]
====== # # output   [4, 64, 128, 64]
decoder_bottleneck
======input_shape [4, 64, 128, 64]
======# conv branch  [4, 64, 128, 64]
======# # 1x1 projection  [4, 64, 128, 16]
======# # conv   [4, 64, 128, 16]
====== # # 1x1 expansion   [4, 64, 128, 64]
====== # #add the batches   [4, 64, 128, 64]
====== # # output   [4, 64, 128, 64]
#######  # # Decoder_2_seg
decoder_bottleneck
======input_shape [4, 64, 128, 64]
======# 1x1 projection (upsampling) [4, 64, 128, 16]
======# max unpooling [4, 128, 256, 16]
======# conv branch  [4, 64, 128, 64]
======# # 1x1 projection  [4, 64, 128, 16]
======# # conv upsampling  [4, 128, 256, 16]
====== # # 1x1 expansion   [4, 128, 256, 16]
====== # #add the batches   [4, 128, 256, 16]
====== # # output   [4, 128, 256, 16]
decoder_bottleneck
======input_shape [4, 128, 256, 16]
======# conv branch  [4, 128, 256, 16]
======# # 1x1 projection  [4, 128, 256, 4]
======# # conv   [4, 128, 256, 4]
====== # # 1x1 expansion   [4, 128, 256, 16]
====== # #add the batches   [4, 128, 256, 16]
====== # # output   [4, 128, 256, 16]
################ total output = [4, 256, 512, 2]

```

减去encode3

```
build modelllllllll inggggggggggggggggggggggggg
################ initial block
===============initial block
======x [4, 256, 512, 3]
======conv_branch [4, 128, 256, 13]
======pool_branch [4, 128, 256, 3]
======concat [4, 128, 256, 16]
======output [4, 128, 256, 16]
===================================
initial_output = [4, 128, 256, 16]
################ Encoder_1
inputs_stage_1 = [4, 128, 256, 16]
===============encoder_bottleneck_regular
======input_shape [4, 128, 256, 16]
======conv_branch (downsampling)  [4, 64, 128, 4]
======convolution [4, 64, 128, 4]
======1x1 expansion [4, 64, 128, 64]
======main_branch(downsampling) [4, 64, 128, 64]
======add the branchs [4, 64, 128, 64]
======output [4, 64, 128, 64]
output_1st_downsample = [4, 64, 128, 64]
pooling_indeces_1 = [4, 64, 128, 16]
===============encoder_bottleneck_regular
======input_shape [4, 64, 128, 64]
======conv_branch [4, 64, 128, 16]
======convolution [4, 64, 128, 16]
======1x1 expansion [4, 64, 128, 64]
======add the branchs [4, 64, 128, 64]
======output [4, 64, 128, 64]
===============encoder_bottleneck_regular
======input_shape [4, 64, 128, 64]
======conv_branch [4, 64, 128, 16]
======convolution [4, 64, 128, 16]
======1x1 expansion [4, 64, 128, 64]
======add the branchs [4, 64, 128, 64]
======output [4, 64, 128, 64]
===============encoder_bottleneck_regular
======input_shape [4, 64, 128, 64]
======conv_branch [4, 64, 128, 16]
======convolution [4, 64, 128, 16]
======1x1 expansion [4, 64, 128, 64]
======add the branchs [4, 64, 128, 64]
======output [4, 64, 128, 64]
===============encoder_bottleneck_regular
======input_shape [4, 64, 128, 64]
======conv_branch [4, 64, 128, 16]
======convolution [4, 64, 128, 16]
======1x1 expansion [4, 64, 128, 64]
======add the branchs [4, 64, 128, 64]
======output [4, 64, 128, 64]
################ Encoder_2
inputs_stage_2 = [4, 64, 128, 64]
===============encoder_bottleneck_regular
======input_shape [4, 64, 128, 64]
======conv_branch (downsampling)  [4, 32, 64, 16]
======convolution [4, 32, 64, 16]
======1x1 expansion [4, 32, 64, 128]
======main_branch(downsampling) [4, 32, 64, 128]
======add the branchs [4, 32, 64, 128]
======output [4, 32, 64, 128]
output_2nd_downsample = [4, 32, 64, 128]
pooling_indices_2 = [4, 32, 64, 64]
===============encoder_bottleneck_regular
======input_shape [4, 32, 64, 128]
======conv_branch [4, 32, 64, 32]
======convolution [4, 32, 64, 32]
======1x1 expansion [4, 32, 64, 128]
======add the branchs [4, 32, 64, 128]
======output [4, 32, 64, 128]
encoder_bottleneck_dilated
======input_shape [4, 32, 64, 128]
======1x1 projection [4, 32, 64, 32]
======dilated conv [4, 32, 64, 32]
======1x1 projection [4, 32, 64, 128]
======main branch [4, 32, 64, 128]
======merge the branches [4, 32, 64, 128]
======output [4, 32, 64, 128]
encoder_bottleneck_asymmetric
======input_shape [4, 32, 64, 128]
======1x1 projection [4, 32, 64, 32]
======asymmetric conv [4, 32, 64, 32]
======asymmetric conv2 [4, 32, 64, 32]
======1x1 projection [4, 32, 64, 128]
======merge the branches [4, 32, 64, 128]
======output [4, 32, 64, 128]
encoder_bottleneck_dilated
======input_shape [4, 32, 64, 128]
======1x1 projection [4, 32, 64, 32]
======dilated conv [4, 32, 64, 32]
======1x1 projection [4, 32, 64, 128]
======main branch [4, 32, 64, 128]
======merge the branches [4, 32, 64, 128]
======output [4, 32, 64, 128]
===============encoder_bottleneck_regular
======input_shape [4, 32, 64, 128]
======conv_branch [4, 32, 64, 32]
======convolution [4, 32, 64, 32]
======1x1 expansion [4, 32, 64, 128]
======add the branchs [4, 32, 64, 128]
======output [4, 32, 64, 128]
encoder_bottleneck_dilated
======input_shape [4, 32, 64, 128]
======1x1 projection [4, 32, 64, 32]
======dilated conv [4, 32, 64, 32]
======1x1 projection [4, 32, 64, 128]
======main branch [4, 32, 64, 128]
======merge the branches [4, 32, 64, 128]
======output [4, 32, 64, 128]
encoder_bottleneck_asymmetric
======input_shape [4, 32, 64, 128]
======1x1 projection [4, 32, 64, 32]
======asymmetric conv [4, 32, 64, 32]
======asymmetric conv2 [4, 32, 64, 32]
======1x1 projection [4, 32, 64, 128]
======merge the branches [4, 32, 64, 128]
======output [4, 32, 64, 128]
encoder_bottleneck_dilated
======input_shape [4, 32, 64, 128]
======1x1 projection [4, 32, 64, 32]
======dilated conv [4, 32, 64, 32]
======1x1 projection [4, 32, 64, 128]
======main branch [4, 32, 64, 128]
======merge the branches [4, 32, 64, 128]
======output [4, 32, 64, 128]
####### Encoder_3_seg
#######  # # Decoder_1_seg
decoder_bottleneck
======input_shape [4, 32, 64, 128]
======# 1x1 projection (upsampling) [4, 32, 64, 64]
======# max unpooling [4, 64, 128, 64]
======# conv branch  [4, 32, 64, 128]
======# # 1x1 projection  [4, 32, 64, 32]
======# # conv upsampling  [4, 64, 128, 32]
====== # # 1x1 expansion   [4, 64, 128, 64]
====== # #add the batches   [4, 64, 128, 64]
====== # # output   [4, 64, 128, 64]
decoder_bottleneck
======input_shape [4, 64, 128, 64]
======# conv branch  [4, 64, 128, 64]
======# # 1x1 projection  [4, 64, 128, 16]
======# # conv   [4, 64, 128, 16]
====== # # 1x1 expansion   [4, 64, 128, 64]
====== # #add the batches   [4, 64, 128, 64]
====== # # output   [4, 64, 128, 64]
decoder_bottleneck
======input_shape [4, 64, 128, 64]
======# conv branch  [4, 64, 128, 64]
======# # 1x1 projection  [4, 64, 128, 16]
======# # conv   [4, 64, 128, 16]
====== # # 1x1 expansion   [4, 64, 128, 64]
====== # #add the batches   [4, 64, 128, 64]
====== # # output   [4, 64, 128, 64]
#######  # # Decoder_2_seg
decoder_bottleneck
======input_shape [4, 64, 128, 64]
======# 1x1 projection (upsampling) [4, 64, 128, 16]
======# max unpooling [4, 128, 256, 16]
======# conv branch  [4, 64, 128, 64]
======# # 1x1 projection  [4, 64, 128, 16]
======# # conv upsampling  [4, 128, 256, 16]
====== # # 1x1 expansion   [4, 128, 256, 16]
====== # #add the batches   [4, 128, 256, 16]
====== # # output   [4, 128, 256, 16]
decoder_bottleneck
======input_shape [4, 128, 256, 16]
======# conv branch  [4, 128, 256, 16]
======# # 1x1 projection  [4, 128, 256, 4]
======# # conv   [4, 128, 256, 4]
====== # # 1x1 expansion   [4, 128, 256, 16]
====== # #add the batches   [4, 128, 256, 16]
====== # # output   [4, 128, 256, 16]
################ total output = [4, 256, 512, 2]
build modelllllllll endddddddddd
```


train_jingjian.py

```
  
   net = lanenet_merge_model.LaneNet(early_drop_prob=early_drop_prob, later_drop_prob=later_drop_prob,
                                      net_flag=net_flag, phase=phase)
                                      
   compute_ret = net.compute_loss(input_tensor=input_tensor, binary_label=binary_label,
                                   instance_label=instance_label, name='lanenet_loss')
```
     
lanenet_merge_model.py

```
def compute_loss(self, input_tensor, binary_label, instance_label, name):
    seg = LaneNetBinarySeg(early_drop_prob=self.early_drop_prob_ph, later_drop_prob=self.later_drop_prob_ph,
                                   phase=self._phase, net_flag=self._net_flag)
                                   
    seg_loss = seg.compute_loss(input_tensor=input_tensor, label=binary_label, name=name+'_seg_loss')
                                   
```

lanenet_binary_segmentation.py

```

class LaneNetBinarySeg(cnn_basenet.CNNBaseModel):
    def __init__(self, early_drop_prob, later_drop_prob, phase, net_flag='enet'):
        elif self._net_flag == "enet":
            self._encoder = enet_encoder.Enet_encoder(phase=phase)
            
        # choose decode model according to encode model
                elif self._net_flag == 'enet':
            self._decoder = enet_decoder.Enet_decoder()

        self.early_drop_prob_ph = early_drop_prob
        self.later_drop_prob_ph = later_drop_prob

      def compute_loss(self, input_tensor, label, name):
      
          inference_ret = self.build_model(input_tensor=input_tensor, name='inference') #开始搭建网络\
      
      def build_model(self, input_tensor, name):
          # first encode
            # 当选择其他encoder时，修改传递参数
            encode_ret, network,\
            inputs_shape_1, \
            pooling_indices_1, \
            inputs_shape_2, \
            pooling_indices_2 = self._encoder.encode(input_tensor=input_tensor,
                                                     early_drop_prob=self.early_drop_prob_ph,
                                                     later_drop_prob=self.later_drop_prob_ph,
                                                     scope='encode')
                                                     
              elif self._net_flag.lower() == 'enet':
                decode_ret = self._decoder.decode_seg(input_tensor=network,
                                                      later_drop_prob=self.later_drop_prob_ph,
                                                      pooling_indices_1=pooling_indices_1,
                                                      pooling_indices_2=pooling_indices_2, scope="decode")

```
enet_encoder.py
```
def encode(self, input_tensor, early_drop_prob, later_drop_prob, scope):

    network = self.initial_block(x=input_tensor, scope="initial")
    
    # # Encoder_1
    ....
    # # Encoder_2
    ....


```
enet_decoder.py

```
def decode_seg(self, input_tensor, later_drop_prob,
                   pooling_indices_1, pooling_indices_2, scope):
             # Encoder_3_seg
            network_seg = self.encoder_bottleneck_regular(x=input_tensor, output_depth=128,
                                                          drop_prob=later_drop_prob,
                                                          scope="seg_bottleneck_3_1")
    
            network_seg = self.encoder_bottleneck_dilated(x=network_seg, output_depth=128,
                                                          drop_prob=later_drop_prob,
                                                          scope="seg_bottleneck_3_2", dilation_rate=2)
    
            network_seg = self.encoder_bottleneck_asymmetric(x=network_seg, output_depth=128,
                                                             drop_prob=later_drop_prob,
                                                             scope="seg_bottleneck_3_3")
    
            network_seg = self.encoder_bottleneck_dilated(x=network_seg, output_depth=128,
                                                          drop_prob=later_drop_prob,
                                                          scope="seg_bottleneck_3_4", dilation_rate=4)
    
            network_seg = self.encoder_bottleneck_regular(x=network_seg, output_depth=128,
                                                          drop_prob=later_drop_prob,
                                                          scope="seg_bottleneck_3_5")
    
            network_seg = self.encoder_bottleneck_dilated(x=network_seg, output_depth=128,
                                                          drop_prob=later_drop_prob,
                                                          scope="seg_bottleneck_3_6", dilation_rate=8)
    
            network_seg = self.encoder_bottleneck_asymmetric(x=network_seg, output_depth=128,
                                                             drop_prob=later_drop_prob,
                                                             scope="seg_bottleneck_3_7")
    
            network_seg = self.encoder_bottleneck_dilated(x=network_seg, output_depth=128,
                                                          drop_prob=later_drop_prob,
                                                          scope="seg_bottleneck_3_8", dilation_rate=16)
            ret['stage3_seg'] = dict()
            ret['stage3_seg']['data'] = network_seg
            ret['stage3_seg']['shape'] = network_seg.get_shape().as_list()
    
            # decoder
            # # Decoder_1_seg
            network_seg = self.decoder_bottleneck(x=network_seg, output_depth=64,
                                                  scope="seg_bottleneck_4_0", upsampling=True,
                                                  pooling_indices=pooling_indices_2)
    
            network_seg = self.decoder_bottleneck(x=network_seg, output_depth=64,
                                                  scope="seg_bottleneck_4_1")
    
            network_seg = self.decoder_bottleneck(x=network_seg, output_depth=64,
                                                  scope="seg_bottleneck_4_2")
    
            ret['stage4_seg'] = dict()
            ret['stage4_seg']['data'] = network_seg
            ret['stage4_seg']['shape'] = network_seg.get_shape().as_list()
    
            # # Decoder_2_seg
            network_seg = self.decoder_bottleneck(x=network_seg, output_depth=16,
                                                  scope="seg_bottleneck_5_0", upsampling=True,
                                                  pooling_indices=pooling_indices_1)
    
            network_seg = self.decoder_bottleneck(x=network_seg, output_depth=16,
                                                  scope="seg_bottleneck_5_1")
    
            ret['stage5_seg'] = dict()
            ret['stage5_seg']['data'] = network_seg
            ret['stage5_seg']['shape'] = network_seg.get_shape().as_list()
    
            # segmentation
            # # arg[1] = 2:  in semantic segmentation branch
            # # arg[1] = 3: in embedding branch
            network_seg = tf.contrib.slim.conv2d_transpose(network_seg, 2,
                                                           [2, 2], stride=2, scope="seg_fullconv", padding="SAME")
            print("output = %s" % network_seg.get_shape().as_list())
            ret['fullconv_seg'] = dict()
            ret['fullconv_seg']['data'] = network_seg#输出的二值分割图像
            ret['fullconv_seg']['shape'] = network_seg.get_shape().as_list()
    
            return ret
```

Enet.py 

```
def initial_block(self, x, scope):
# convolution branch
        print("===============initial block")
        print("======x", x.get_shape().as_list())
        W_conv = self.get_variable_weight_decay(scope + "/W", shape=[3, 3, 3, 13],
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                loss_category="encoder_wd_loss")#13个卷积核,3*3,4张图片
        b_conv = self.get_variable_weight_decay(scope + "/b", shape=[13],
                                                initializer=tf.constant_initializer(0),
                                                loss_category="encoder_wd_loss")
        conv_branch = tf.nn.conv2d(x, W_conv, strides=[1, 2, 2, 1], padding="SAME") + b_conv
        print("======conv_branch", conv_branch.get_shape().as_list())#卷积后
        # maxpooling branch
        pool_branch = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        print("======pool_branch",pool_branch.get_shape().as_list())#池化后
        # cancatenate these two branches
        concat = tf.concat([conv_branch, pool_branch], axis=3)  # (3: the depth axis)
        print("======concat", concat.get_shape().as_list())#直接将卷积和池化拼在了一起
        # apply BN and PReLU
        output = tf.contrib.slim.batch_norm(concat)
        print("======output", output.get_shape().as_list())
        output = self.PReLU(output, scope=scope)
        print("===================================")

        return output


输出:

======x [4, 256, 512, 3]
======conv_branch [4, 128, 256, 13]
======pool_branch [4, 128, 256, 3]
======concat [4, 128, 256, 16]
======output [4, 128, 256, 16]

```

