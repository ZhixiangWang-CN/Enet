
# -*- coding: utf-8 -*-
# @Time    : 2018.09.06 14:48
# @Author  : Aaron Ran
# @IDE: PyCharm Community Edition
"""
实现一个基于Enet-encoder的特征编码类
"""
from collections import OrderedDict

import tensorflow as tf
from encoder_decoder_model.ENet import Enet_model


class Enet_decoder(Enet_model):
    """
    实现了一个基于ENet的特征解码类
    """
    def __init__(self):
        """


        """
        super(Enet_decoder, self).__init__()
    
    def decode_seg(self, input_tensor, later_drop_prob,
                   pooling_indices_1, pooling_indices_2, scope):
        ret = OrderedDict()
        
        with tf.variable_scope(scope):
            
            # Encoder_3_seg
            print("####### Encoder_3_seg")
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
            print("#######  # # Decoder_1_seg")
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
            print("#######  # # Decoder_2_seg")
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
            print("################ total output = %s" % network_seg.get_shape().as_list())
            ret['fullconv_seg'] = dict()
            ret['fullconv_seg']['data'] = network_seg#输出的二值分割图像
            ret['fullconv_seg']['shape'] = network_seg.get_shape().as_list()
    
            return ret

    def decode_emb(self, input_tensor, later_drop_prob,
                   pooling_indices_1, pooling_indices_2, scope):
        
        ret = OrderedDict()
        
        with tf.variable_scope(scope):
            
            # Encoder_3_emb
            network_emb = self.encoder_bottleneck_regular(x=input_tensor, output_depth=128,
                                                          drop_prob=later_drop_prob,
                                                          scope="emb_bottleneck_3_1")
    
            network_emb = self.encoder_bottleneck_dilated(x=network_emb, output_depth=128,
                                                          drop_prob=later_drop_prob,
                                                          scope="emb_bottleneck_3_2", dilation_rate=2)
    
            network_emb = self.encoder_bottleneck_asymmetric(x=network_emb, output_depth=128,
                                                             drop_prob=later_drop_prob,
                                                             scope="emb_bottleneck_3_3")
    
            network_emb = self.encoder_bottleneck_dilated(x=network_emb, output_depth=128,
                                                          drop_prob=later_drop_prob,
                                                          scope="emb_bottleneck_3_4", dilation_rate=4)
    
            network_emb = self.encoder_bottleneck_regular(x=network_emb, output_depth=128,
                                                          drop_prob=later_drop_prob,
                                                          scope="emb_bottleneck_3_5")
    
            network_emb = self.encoder_bottleneck_dilated(x=network_emb, output_depth=128,
                                                          drop_prob=later_drop_prob,
                                                          scope="emb_bottleneck_3_6", dilation_rate=8)
    
            network_emb = self.encoder_bottleneck_asymmetric(x=network_emb, output_depth=128,
                                                             drop_prob=later_drop_prob,
                                                             scope="emb_bottleneck_3_7")
    
            network_emb = self.encoder_bottleneck_dilated(x=network_emb, output_depth=128,
                                                          drop_prob=later_drop_prob,
                                                          scope="emb_bottleneck_3_8", dilation_rate=16)
    
            ret['stage3_emb'] = dict()
            ret['stage3_emb']['data'] = network_emb
            ret['stage3_emb']['shape'] = network_emb.get_shape().as_list()
    
            # decoder
            # # Decoder_1_emb
            network_emb = self.decoder_bottleneck(x=network_emb, output_depth=64,
                                                  scope="emb_bottleneck_4_0", upsampling=True,
                                                  pooling_indices=pooling_indices_2)
    
            network_emb = self.decoder_bottleneck(x=network_emb, output_depth=64,
                                                  scope="emb_bottleneck_4_1")
    
            network_emb = self.decoder_bottleneck(x=network_emb, output_depth=64,
                                                  scope="emb_bottleneck_4_2")
            ret['stage4_emb'] = dict()
            ret['stage4_emb']['data'] = network_emb
            ret['stage4_emb']['shape'] = network_emb.get_shape().as_list()
    
            # # Decoder_2_emb
            network_emb = self.decoder_bottleneck(x=network_emb, output_depth=16,
                                                  scope="emb_bottleneck_5_0", upsampling=True,
                                                  pooling_indices=pooling_indices_1)
    
            network_emb = self.decoder_bottleneck(x=network_emb, output_depth=16,
                                                  scope="emb_bottleneck_5_1")
    
            ret['stage5_emb'] = dict()
            ret['stage5_emb']['data'] = network_emb
            ret['stage5_emb']['shape'] = network_emb.get_shape().as_list()
    
            # embedding
            # # arg[1] = 1:  in semantic segmentation branch
            # # arg[1] = 3: in embedding branch
            network_emb = tf.contrib.slim.conv2d_transpose(network_emb, 3,
                                                           [2, 2], stride=2, scope="emb_fullconv", padding="SAME")
    
            ret['fullconv_emb'] = dict()
            ret['fullconv_emb']['data'] = network_emb
            ret['fullconv_emb']['shape'] = network_emb.get_shape().as_list()
    
            return ret


if __name__ == '__main__':

    input_tensor = tf.placeholder(tf.float32, shape=[1, 90, 160, 128], name="input_tensor")
    later_drop_prob_ph = tf.placeholder(tf.float32, name="later_drop_prob_ph")
    inputs_shape_1 = tf.placeholder(tf.float32, shape=[1, 360, 640, 16], name="inputs_shape_1")
    inputs_shape_2 = tf.placeholder(tf.float32, shape=[1, 180, 320, 64], name="inputs_shape_2")
    pooling_indices_1 = tf.placeholder(tf.float32, shape=[1, 180, 320, 16], name="pooling_indices_1")
    pooling_indices_2 = tf.placeholder(tf.float32, shape=[1, 90, 160, 64], name="pooling_indices_2")

    decoder = Enet_decoder()
    seg = decoder.decode_seg(input_tensor=input_tensor, later_drop_prob=later_drop_prob_ph,
                             pooling_indices_1=pooling_indices_1, pooling_indices_2=pooling_indices_2,
                             scope="decode_seg")
    for layer_name, layer_info in seg.items():
        print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))

    emb = decoder.decode_emb(input_tensor=input_tensor, later_drop_prob=later_drop_prob_ph,
                             pooling_indices_1=pooling_indices_1, pooling_indices_2=pooling_indices_2,
                             scope="decode_emb")
    for layer_name, layer_info in emb.items():
        print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))
