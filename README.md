# Enet

运行命令
======================================

训练:

python train_jingjian.py 

继续训练:

python train_jingjian.py --weights_path model/culane_lanenet/culane_lanenet_enet_2018-10-23-18-27-58.ckpt-105600

测试
python test_jingjian.py  --dataset_dir ./data/training_data_example --weights_path model/culane_lanenet/culane_lanenet_enet_2018-10-23-18-27-58.ckpt-105600


Enet论文阅读
======================================

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




