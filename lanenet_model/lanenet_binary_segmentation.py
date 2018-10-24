#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 上午11:33
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_binary_segmentation.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet中的二分类图像分割模型
"""
import tensorflow as tf
import os.path as ops
import glog as log
import cv2
import numpy as np
import math

from encoder_decoder_model import enet_encoder
from encoder_decoder_model import enet_decoder
from encoder_decoder_model import vgg_encoder
from encoder_decoder_model import fcn_decoder
from encoder_decoder_model import dense_encoder
from encoder_decoder_model import cnn_basenet
from data_provider import lanenet_data_processor


class LaneNetBinarySeg(cnn_basenet.CNNBaseModel):
    """
    实现语义分割模型
    """
    def __init__(self, early_drop_prob, later_drop_prob, phase, net_flag='enet'):
        """

        """
        super(LaneNetBinarySeg, self).__init__()
        self._net_flag = net_flag
        self._phase = phase

        # choose suitable encode model
        if self._net_flag == 'vgg':
            self._encoder = vgg_encoder.VGG16Encoder(phase=phase)
        elif self._net_flag == 'dense':
            self._encoder = dense_encoder.DenseEncoder(l=20, growthrate=8,
                                                       with_bc=True,
                                                       phase=self._phase,
                                                       n=5)
        elif self._net_flag == "enet":
            self._encoder = enet_encoder.Enet_encoder(phase=phase)

        # choose decode model according to encode model
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
        print("build modelllllllll inggggggggggggggggggggggggg")
        with tf.variable_scope(name):
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
                decode_ret = self._decoder.decode_seg(input_tensor=network,
                                                      later_drop_prob=self.later_drop_prob_ph,
                                                      pooling_indices_1=pooling_indices_1,
                                                      pooling_indices_2=pooling_indices_2, scope="decode")

                print("build modelllllllll endddddddddd")
                return decode_ret

    def compute_loss(self, input_tensor, label, name):
        """
        计算损失函数
        :param input_tensor:
        :param label:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # 前向传播获取logits
            inference_ret = self.build_model(input_tensor=input_tensor, name='inference')
            # 计算损失
            decode_logits = inference_ret['fullconv_seg']['data']
            print("+++++++++++++++++++++++++++decode logits",decode_logits)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=decode_logits, labels=tf.squeeze(label, axis=[3]),
                name='entropy_loss')
            print("+++++++++++++++++++++",loss)
            loss = tf.reduce_mean(input_tensor=loss)
            print("+++++++++++++++++++",loss)
            ret = dict()
            ret['entropy_loss'] = loss
            ret['seg_logits'] = inference_ret['fullconv_seg']['data']

            return ret


if __name__ == '__main__':
    VGG_MEAN = [103.939, 116.779, 123.68]
    dataset_dir = '../../data/training_data_example/'
    train_dataset_file = ops.join(dataset_dir, 'train.txt')
    assert ops.exists(train_dataset_file)
    train_dataset = lanenet_data_processor.DataSet(train_dataset_file)

    input_tensor = tf.placeholder(dtype=tf.float32,
                                  shape=[1, 720,
                                         1280, 3],
                                  name='input_tensor')
    binary_label = tf.placeholder(dtype=tf.int64,
                                  shape=[1, 720, 1280, 1],
                                  name='binary_input_label')
    early_drop_prob = tf.placeholder(dtype=tf.float32)
    later_drop_prob = tf.placeholder(dtype=tf.float32)

    model = LaneNetBinarySeg(early_drop_prob=early_drop_prob, later_drop_prob=later_drop_prob,
                             phase='train')

    loss = model.compute_loss(input_tensor=input_tensor, label=binary_label, name='loss')

    binary_seg_loss = loss['entropy_loss']

    optimizer = tf.train.AdamOptimizer(learning_rate=
                                       1e-5).minimize(loss=binary_seg_loss)

    sess_config = tf.ConfigProto(device_count={'GPU': 1})
    print('++++++++++++++++++++')
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    train_epochs = 200

    with sess.as_default():
        log.info('Training from scratch')
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(train_epochs):
            # training part

            gt_imgs, binary_gt_labels, instance_gt_labels = train_dataset.next_batch(1)
            gt_imgs = [cv2.resize(tmp,
                                  dsize=(1280, 720),
                                  dst=tmp,
                                  interpolation=cv2.INTER_LINEAR)
                       for tmp in gt_imgs]
            gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]
            binary_gt_labels = [cv2.resize(tmp,
                                           dsize=(1280, 720),
                                           dst=tmp,
                                           interpolation=cv2.INTER_NEAREST)
                                for tmp in binary_gt_labels]
            binary_gt_labels = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels]

            phase_train = 'train'

            _,  binary_loss = \
                sess.run([optimizer, binary_seg_loss],
                         feed_dict={input_tensor: gt_imgs,
                                    binary_label: binary_gt_labels,
                                    early_drop_prob: 0.01,
                                    later_drop_prob: 0.1})

            if math.isnan(binary_loss) :
                log.error('binary cost is: {:.5f}'.format(binary_loss))
                break

        sess.close()



