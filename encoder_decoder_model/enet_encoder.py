

# @Time    : 2018.09.06 14:48
# @Author  : Aaron Ran
# @IDE: PyCharm Community Edition
"""
实现一个基于Enet-encoder的特征编码类
"""
from collections import OrderedDict

import tensorflow as tf

from encoder_decoder_model.ENet import Enet_model


class Enet_encoder(Enet_model):
    """
    实现了一个基于ENet的特征编码类
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(Enet_encoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)
    
    def encode(self, input_tensor, early_drop_prob, later_drop_prob, scope):
        """
        根据ENet框架对输入的tensor进行编码
        :param input_tensor:
        :param early_drop_prob:
        :param later_drop_prob:
        :param scope:
        :return: 输出ENet编码特征
        """
        ret = OrderedDict()

        with tf.variable_scope(scope):
            # encoder
            # # initial block
            # network = self()
            network = self.initial_block(x=input_tensor, scope="initial")
            ret['initial'] = dict()
            ret['initial']['data'] = network
            ret['initial']['shape'] = network.get_shape().as_list()
            print('initial_output = %s' % network.get_shape().as_list())

            # # Encoder_1
            # # # save the input shape to use in max_unpool in the decoder
            inputs_shape_1 = network.get_shape().as_list()
            print("inputs_stage_1 = %s" % inputs_shape_1)
            network, pooling_indices_1 = self.encoder_bottleneck_regular(x=network, output_depth=64,
                                                                         drop_prob=early_drop_prob,
                                                                         scope="bottleneck_1_0", downsampling=True)
            print("output_1st_downsample = %s" % (network.get_shape().as_list()))
            print("pooling_indeces_1 = %s" % pooling_indices_1.get_shape().as_list())

            network = self.encoder_bottleneck_regular(x=network, output_depth=64,
                                                      drop_prob=early_drop_prob,
                                                      scope="bottleneck_1_1")

            network = self.encoder_bottleneck_regular(x=network, output_depth=64,
                                                      drop_prob=early_drop_prob,
                                                      scope="bottleneck_1_2")

            network = self.encoder_bottleneck_regular(x=network, output_depth=64,
                                                      drop_prob=early_drop_prob,
                                                      scope="bottleneck_1_3")

            network = self.encoder_bottleneck_regular(x=network, output_depth=64,
                                                      drop_prob=early_drop_prob,
                                                      scope="bottleneck_1_4")

            ret['stage_1'] = dict()
            ret['stage_1']['data'] = network
            ret['stage_1']['shape'] = network.get_shape().as_list()

            # # Encoder_2
            # # # save the input shape to use in max_pool in the decoder
            inputs_shape_2 = network.get_shape().as_list()
            print("inputs_stage_2 = %s" % inputs_shape_2)
            network, pooling_indices_2 = self.encoder_bottleneck_regular(x=network, output_depth=128,
                                                                         drop_prob=later_drop_prob,
                                                                         scope="bottleneck_2_0", downsampling=True)

            print("output_2nd_downsample = %s" % network.get_shape().as_list())
            print("pooling_indices_2 = %s" % pooling_indices_2.get_shape().as_list())
            network = self.encoder_bottleneck_regular(x=network, output_depth=128,
                                                      drop_prob=later_drop_prob,
                                                      scope="bottleneck_2_1")

            network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                                                      drop_prob=later_drop_prob,
                                                      scope="bottleneck_2_2", dilation_rate=2)

            network = self.encoder_bottleneck_asymmetric(x=network, output_depth=128,
                                                         drop_prob=later_drop_prob,
                                                         scope="bottleneck_2_3")

            network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                                                      drop_prob=later_drop_prob,
                                                      scope="bottleneck_2_4", dilation_rate=4)

            network = self.encoder_bottleneck_regular(x=network, output_depth=128,
                                                      drop_prob=later_drop_prob,
                                                      scope="bottleneck_2_5")

            network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                                                      drop_prob=later_drop_prob,
                                                      scope="bottleneck_2_6", dilation_rate=8)

            network = self.encoder_bottleneck_asymmetric(x=network, output_depth=128,
                                                         drop_prob=later_drop_prob,
                                                         scope="bottleneck_2_7")

            network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                                                      drop_prob=later_drop_prob,
                                                      scope="bottleneck_2_8", dilation_rate=16)

            ret['stage_2'] = dict()
            ret['stage_2']['data'] = network
            ret['stage_2']['shape'] = network.get_shape().as_list()

        return ret, network, inputs_shape_1, pooling_indices_1, inputs_shape_2, pooling_indices_2
    # TODO(luoyao) luoyao@baidu.com 检查batch normalization分布和迁移是否合理


if __name__ == '__main__':
    # dropout probability before bottleneck2.0 in ENet
    early_drop_prob_ph = tf.placeholder(tf.float32, name="early_drop_prob_ph")
    # dropout probability after bottleneck2.0 in ENet
    later_drop_prob_ph = tf.placeholder(tf.float32, name="later_drop_prob_ph")
    # input_tensor
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 720, 1280, 3], name='input')
    encoder = Enet_encoder(phase=tf.constant('train', dtype=tf.string))
    ret, _, _, _, _, _ = encoder.encode(input_tensor=input_tensor, early_drop_prob=early_drop_prob_ph,
                         later_drop_prob=later_drop_prob_ph, scope='encode')

    # print(ret)
    for layer_name, layer_info in ret.items():
        print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))