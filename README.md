# Enet

Enet论文阅读
======================================

![](https://img-blog.csdn.net/20170328161948506?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvemlqaW54dXh1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![](https://img-blog.csdn.net/20170328162004865?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvemlqaW54dXh1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


因为此代码是基于lanenet网络改进的，所以需要先了解lanenet网络的一些基本参数：


![](https://pic3.zhimg.com/80/v2-0a7016d069c25a1aa65384349d9d24ee_hd.png)

lossfunction-lanenet：

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


