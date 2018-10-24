#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 上午11:35
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_instance_segmentation.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet中的实例图像分割模型
"""
import tensorflow as tf

from encoder_decoder_model import vgg_encoder
from encoder_decoder_model import fcn_decoder
from encoder_decoder_model import dense_encoder
from encoder_decoder_model import cnn_basenet
from encoder_decoder_model import enet_encoder
from encoder_decoder_model import enet_decoder
from lanenet_model import lanenet_discriminative_loss


class LaneNetInstanceSeg(cnn_basenet.CNNBaseModel):
    """
    实现语义分割模型
    """
    def __init__(self, early_drop_prob, later_drop_prob, phase, net_flag='enet'):
        """

        """
        super(LaneNetInstanceSeg, self).__init__()
        self._net_flag = net_flag
        self._phase = phase
        if self._net_flag == 'vgg':
            self._encoder = vgg_encoder.VGG16Encoder(phase=phase)
        elif self._net_flag == 'dense':
            self._encoder = dense_encoder.DenseEncoder(l=20, growthrate=8,
                                                       with_bc=True,
                                                       phase=self._phase,
                                                       n=5)
        elif self._net_flag == 'enet':
            self._encoder = enet_encoder.Enet_encoder(phase=phase)

        if self._net_flag == 'vgg' or self._net_flag == 'dense':
            self._decoder = fcn_decoder.FCNDecoder()
        elif self._net_flag == 'enet':
            self._decoder = enet_decoder.Enet_decoder()

        self.early_drop_prob_ph = early_drop_prob
        self.later_drop_prob_ph = later_drop_prob
        return

    def __str__(self):
        """

        :return:
        """
        info = 'Semantic Segmentation use {:s} as basenet to encode'.format(self._net_flag)
        return info

    def build_model(self, input_tensor, name):
        """
        前向传播过程
        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # first encode
            encode_ret, network,\
            inputs_shape_1, \
            pooling_indices_1, \
            inputs_shape_2, \
            pooling_indices_2 = self._encoder.encode(input_tensor=input_tensor,
                                                     early_drop_prob=self.early_drop_prob_ph,
                                                     later_drop_prob=self.later_drop_prob_ph,
                                                     scope='encode')

            # second decode
            if self._net_flag.lower() == 'vgg':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['pool5',
                                                                     'pool4',
                                                                     'pool3'])
                return decode_ret
            elif self._net_flag.lower() == 'dense':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['Dense_Block_5',
                                                                     'Dense_Block_4',
                                                                     'Dense_Block_3'])
                return decode_ret

            elif self._net_flag.lower() == 'enet':
                decode_ret = self._decoder.decode_emb(input_tensor=network,
                                                      later_drop_prob=self.later_drop_prob_ph,
                                                      pooling_indices_1=pooling_indices_1,
                                                      pooling_indices_2=pooling_indices_2, scope="decode")
                return decode_ret

    def compute_loss(self, input_tensor, label, name):
        """
        计算损失函数
        :param input_tensor:
        :param label: 1D label image with different n lane with pix value from [1] to [n],
                      background pix value is [0]
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # 前向传播获取logits
            inference_ret = self.build_model(input_tensor=input_tensor, name='inference')
            # # 计算损失
            # decode_deconv = inference_ret['']
            # # 像素嵌入
            # pix_embedding = self.conv2d(inputdata=decode_deconv, out_channel=3, kernel_size=1,
            #                             use_bias=False, name='pix_embedding_conv')
            pix_embedding = self.relu(inputdata=inference_ret['fullconv_emb']['data'], name='pix_embedding_relu')
            print("pix_embedding_shape = %s" % pix_embedding.get_shape().as_list())
            # 计算discriminative loss
            image_shape = (pix_embedding.get_shape().as_list()[1], pix_embedding.get_shape().as_list()[2])
            disc_loss, l_var, l_dist, l_reg = \
                lanenet_discriminative_loss.discriminative_loss(
                    prediction=pix_embedding, correct_label=label, feature_dim=3, image_shape=image_shape,
                    delta_v=0.5, delta_d=3., param_var=1.0, param_dist=1.0, param_reg=0.001)

            ret = {
                'disc_loss': disc_loss,
                'loss_var': l_var,
                'loss_dist': l_dist,
                'loss_reg': l_reg,
                'embedding': pix_embedding
            }

            return ret


if __name__ == '__main__':
    early_drop_prob = tf.placeholder(dtype=tf.float32)
    later_drop_prob = tf.placeholder(dtype=tf.float32)

    model = LaneNetInstanceSeg(early_drop_prob=early_drop_prob, later_drop_prob=later_drop_prob, phase='train')
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 720, 1280, 3], name='input')
    label = tf.placeholder(dtype=tf.float32, shape=[1, 720, 1280, 1], name='label')
    loss = model.compute_loss(input_tensor=input_tensor, label=label, name='loss')
    print(loss['disc_loss'].get_shape().as_list())
