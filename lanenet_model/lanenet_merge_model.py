#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午5:28
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_merge_model.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet模型
"""
import tensorflow as tf

from lanenet_model.lanenet_binary_segmentation import LaneNetBinarySeg
from lanenet_model.lanenet_instance_segmentation import LaneNetInstanceSeg
from encoder_decoder_model import cnn_basenet




class LaneNet(cnn_basenet.CNNBaseModel):
    """
    实现语义分割模型
    """
    def __init__(self, early_drop_prob, later_drop_prob, phase, net_flag='enet'):
        """

        """
        super(LaneNet, self).__init__()#首先找到test的父类（比如是类A），然后把类test的对象self转换为类A的对象，然后“被转换”的类A对象调用自己的__init__函数
        self._net_flag = net_flag
        self._phase = phase
        self.early_drop_prob_ph = early_drop_prob
        self.later_drop_prob_ph = later_drop_prob

        return

    def __str__(self):
        """

        :return:
        """
        info = 'Semantic Segmentation use {:s} as basenet to encode'.format(self._net_flag)
        return info

    # def _build_model(self, input_tensor, name):
    #     """
    #     前向传播过程
    #     :param input_tensor:
    #     :param name:
    #     :return:
    #     """
    #     with tf.variable_scope(name):
    #         # binary segmentation branch
    #         seg = LaneNetBinarySeg(input_tensor, name)
    #         # embedding branch
    #         emb = LaneNetInstanceSeg(input_tensor, name)



    def compute_loss(self, input_tensor, binary_label, instance_label, name):
        """
        计算LaneNet模型损失函数
        :param input_tensor:
        :param binary_label:
        :param instance_label:
        :param name:
        :return:
        """

        with tf.variable_scope(name):

            # segmentation branch and embedding branch
            seg = LaneNetBinarySeg(early_drop_prob=self.early_drop_prob_ph, later_drop_prob=self.later_drop_prob_ph,
                                   phase=self._phase, net_flag=self._net_flag)
            # emb = LaneNetInstanceSeg(early_drop_prob=self.early_drop_prob_ph, later_drop_prob=self.later_drop_prob_ph,
            #                          phase=self._phase, net_flag=self._net_flag)

            # segmentation loss and embedding loss
            seg_loss = seg.compute_loss(input_tensor=input_tensor, label=binary_label, name=name+'_seg_loss')
            # emb_loss = emb.compute_loss(input_tensor=input_tensor, label=instance_label, name=name+'_emb_loss')

            # 合并损失
            # total_loss = 0.7 * seg_loss['entropy_loss'] + 0.3 * emb_loss['disc_loss']

            total_loss = seg_loss['entropy_loss']
            print("total_loss======", total_loss)
            print("binary_seg_loss======", seg_loss)
            # print("discriminative_loss======",emb_loss)
            ret = {
                'total_loss': total_loss,
                'binary_seg_logits': seg_loss['seg_logits'],
                # 'instance_seg_logits': emb_loss['embedding'],
                'binary_seg_loss': seg_loss['entropy_loss'],
                # 'discriminative_loss': emb_loss['disc_loss']
            }

            return ret

    def inference(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # 前向传播获取logits

            inference_seg = LaneNetBinarySeg(early_drop_prob=self.early_drop_prob_ph, later_drop_prob=self.later_drop_prob_ph,
                                             phase=self._phase, net_flag=self._net_flag)

            inference_seg_logits = inference_seg.build_model(input_tensor=input_tensor, name='inference_seg')

            # inference_emb = LaneNetInstanceSeg(early_drop_prob=self.early_drop_prob_ph, later_drop_prob=self.later_drop_prob_ph,
            #                                    phase=self._phase, net_flag=self._net_flag)
            # inference_emb_logits = inference_emb.build_model(input_tensor=input_tensor, name='inference_emb')

            # 计算二值分割损失函数
            decode_logits_seg = inference_seg_logits['fullconv_seg']['data']
            binary_seg_ret = tf.nn.softmax(logits=decode_logits_seg)
            binary_seg_ret = tf.argmax(binary_seg_ret, axis=-1)
            # 计算像素嵌入
            # decode_logits_emb = inference_emb_logits['fullconv_emb']['data']

            # pix_embedding = self.relu(inputdata=decode_logits_emb, name='pix_embedding_relu')

            return binary_seg_ret

            # return binary_seg_ret, pix_embedding


if __name__ == '__main__':
    model = LaneNet(tf.constant('train', dtype=tf.string))
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
    binary_label = tf.placeholder(dtype=tf.int64, shape=[1, 256, 512, 1], name='label')
    instance_label = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 1], name='label')
    net =LaneNet(early_drop_prob=0.1, later_drop_prob=0.1,
                                      net_flag='enet', phase='net_phase')
    ret = model.compute_loss(input_tensor=input_tensor, binary_label=binary_label,
                             instance_label=instance_label, name='loss')
    print(ret['total_loss'])
    print(ret)
