# Enet

运行命令
======================================

训练:

python train_jingjian.py 

继续训练:

python train_jingjian.py --weights_path model/culane_lanenet/culane_lanenet_enet_2018-10-23-18-27-58.ckpt-105600

测试
python test_jingjian.py  --dataset_dir ./data/training_data_example --weights_path model/culane_lanenet/culane_lanenet_enet_2018-10-23-18-27-58.ckpt-105600

下一步方向：
===========================================

更快

更小

更准

网络结构的优化:

卷积运算方式,bottleneck设计,添加或删除不同的层

shuffle net

研究资料:
----------------------
[CNN网络优化学习总结——从MobileNet到ShuffleNet](https://blog.csdn.net/sun_28/article/details/78170878)

高效模型设计： 

CNNs在CV任务中取得了极大的成功，在嵌入式设备上运行高质量深度神经网络需求越来越大，这也促进了对高效模型的探究。

例如，与单纯的堆叠卷积层，GoogleNet增加了网络的宽度，复杂度降低很多；

SqueezeNet在保持精度的同时大大减少参数和计算量；

ResNet利用高效的bottleneck结构实现惊人的效果。Xception中提出深度可分卷积概括了Inception序列。

MobileNet利用深度可分卷积构建的轻量级模型获得了先进的成果；

ShuffleNet的工作是推广群卷积(group convolution)和深度可分卷积(depthwise separable convolution)。



Enet论文阅读
======================================


[Semantic Segmentation--ENet:A Deep Neural Network Architecture for Real-Time Semantic..论文解读](https://blog.csdn.net/u011974639/article/details/78956380)

![](https://pic2.zhimg.com/80/v2-b7f8d494d2777b64a999faff07effe4e_hd.jpg)

![](https://pic4.zhimg.com/80/v2-870fd241940d9a8d2cb1d82537f37778_hd.jpg)


因为此代码是基于lanenet网络改进的，所以需要先了解lanenet网络的一些基本参数：


![](https://pic3.zhimg.com/80/v2-0a7016d069c25a1aa65384349d9d24ee_hd.png)

![](http://owv7la1di.bkt.clouddn.com/blog/180102/Hb10AFHcJI.png)





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


![](https://images2015.cnblogs.com/blog/1062917/201611/1062917-20161117195428888-895158719.png)

strides=[1, 1, 1, 1]参数解释:

conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')

这是一个常见的卷积操作，其中strides=【1,1,1,1】表示滑动步长为1，padding=‘SAME’表示填0操作

当我们要设置步长为2时，strides=【1,2,2,1】，很多同学可能不理解了，这四个参数分别代表了什么，查了官方函数说明一样不明不白，今天我来解释一下。

strides在官方定义中是一个一维具有四个元素的张量，其规定前后必须为1，这点大家就别纠结了，所以我们可以改的是中间两个数，中间两个数分别代表了水平滑动和垂直滑动步长值，于是就很好理解了。

在卷积核移动逐渐扫描整体图时候，因为步长的设置问题，可能导致剩下未扫描的空间不足以提供给卷积核的，大小扫描 比如有图大小为5*5,卷积核为2*2,步长为2,卷积核扫描了两次后，剩下一个元素，不够卷积核扫描了，这个时候就在后面补零，补完后满足卷积核的扫描，这种方式就是same。如果说把刚才不足以扫描的元素位置抛弃掉，就是valid方式。


![](https://images2015.cnblogs.com/blog/1062917/201611/1062917-20161117211920029-1784506227.png)


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

