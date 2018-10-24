#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
测试LaneNet模型
"""
import os
import os.path as ops
import argparse
import time
import math

import tensorflow as tf
import glob
import glog as log
import numpy as np
import matplotlib.pyplot as plt
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass
from config import global_config
from lanenet_model import lanenet_merge_model
from data_provider import lanenet_data_processor
from lanenet_model import lanenet_merge_model
from lanenet_model import lanenet_cluster
from lanenet_model import lanenet_postprocess
from config import global_config

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]

def init_args():



    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, help='The training dataset dir path')
    parser.add_argument('--net', type=str, help='Which base net work to use', default='enet')
    parser.add_argument('--weights_path', type=str, help='The pretrained weights path')

    return parser.parse_args()


def mergepic(image,binary):
    """
    将预测的位置加到原图上
    :param image:
    :param binary:
    :return:
    """

    spimg = image.shape

    spbimg = binary.shape

    w = spbimg[1]

    h = spbimg[0]

    a = []
    for i in range(h):
        for j in range(w):
            if binary[i][j] != 0:
                a.append([i, j])

    for p in a:
        image[p[0]][p[1]] = (255, 255, 0)


    return image


def train_net(dataset_dir, weights_path=None, net_flag='enet'):
    """

    :param dataset_dir:
    :param net_flag: choose which base network to use
    :param weights_path:
    :return:
    """

    val_dataset_file = ops.join(dataset_dir, 'test.txt')
    namenum = 0
    assert ops.exists(val_dataset_file)


    val_dataset = lanenet_data_processor.DataSet(val_dataset_file)
    # CFG.TRAIN.BATCH_SIZE
    input_tensor = tf.placeholder(dtype=tf.float32,
                                  shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                         CFG.TRAIN.IMG_WIDTH, 3],
                                  name='input_tensor')
    binary_label = tf.placeholder(dtype=tf.int64,
                                  shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                         CFG.TRAIN.IMG_WIDTH, 1],
                                  name='binary_input_label')
    instance_label = tf.placeholder(dtype=tf.float32,
                                    shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                           CFG.TRAIN.IMG_WIDTH],
                                    name='instance_input_label')

    early_drop_prob = tf.placeholder(dtype=tf.float32)
    later_drop_prob = tf.placeholder(dtype=tf.float32)

    phase = tf.placeholder(dtype=tf.string, shape=None, name='net_phase')

    net = lanenet_merge_model.LaneNet(early_drop_prob=early_drop_prob, later_drop_prob=later_drop_prob,
                                      net_flag=net_flag, phase=phase)

    # calculate the loss
    compute_ret = net.compute_loss(input_tensor=input_tensor, binary_label=binary_label,
                                   instance_label=instance_label, name='lanenet_loss')
    total_loss = compute_ret['total_loss']
    binary_seg_loss = compute_ret['binary_seg_loss']


    # calculate the accuracy
    out_logits = compute_ret['binary_seg_logits']
    out_logits = tf.nn.softmax(logits=out_logits)
    out_logits_out = tf.argmax(out_logits, axis=-1)
    out = tf.argmax(out_logits, axis=-1)
    out = tf.expand_dims(out, axis=-1)
    print("out_shape = %s" % out.get_shape().as_list())

    idx = tf.where(tf.equal(binary_label, 1))
    pix_cls_ret = tf.gather_nd(out, idx)
    accuracy = tf.count_nonzero(pix_cls_ret)
    accuracy = tf.divide(accuracy, tf.cast(tf.shape(pix_cls_ret)[0], tf.int64))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(CFG.TRAIN.LEARNING_RATE, global_step,
                                               5000, 0.96, staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=
                                           learning_rate).minimize(loss=total_loss,
                                                                   var_list=tf.trainable_variables(),
                                                                   global_step=global_step)

    # Set tf saver
    saver = tf.train.Saver()
    # model_save_dir = 'model/culane_lanenet'
    # if not ops.exists(model_save_dir):
    #     os.makedirs(model_save_dir)
    #
    # train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # model_name = 'culane_lanenet_{:s}_{:s}.ckpt'.format(net_flag, str(train_start_time))
    # model_save_path = ops.join(model_save_dir, model_name)

    # # Set tf summary
    # tboard_save_path = 'tboard/culane_lanenet/{:s}'.format(net_flag)
    # if not ops.exists(tboard_save_path):
    #     os.makedirs(tboard_save_path)

    train_cost_scalar = tf.summary.scalar(name='train_cost', tensor=total_loss)
    val_cost_scalar = tf.summary.scalar(name='val_cost', tensor=total_loss)
    train_accuracy_scalar = tf.summary.scalar(name='train_accuracy', tensor=accuracy)
    val_accuracy_scalar = tf.summary.scalar(name='val_accuracy', tensor=accuracy)
    train_binary_seg_loss_scalar = tf.summary.scalar(name='train_binary_seg_loss', tensor=binary_seg_loss)
    val_binary_seg_loss_scalar = tf.summary.scalar(name='val_binary_seg_loss', tensor=binary_seg_loss)

    learning_rate_scalar = tf.summary.scalar(name='learning_rate', tensor=learning_rate)

    print("---------------------------------------------------------------------------------")
    train_merge_summary_op = tf.summary.merge([train_accuracy_scalar, train_cost_scalar,
                                               learning_rate_scalar
                                               ])
    val_merge_summary_op = tf.summary.merge([val_accuracy_scalar, val_cost_scalar
                                             ])

    sess_config = tf.ConfigProto(device_count={'GPU': 1})

    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # summary_writer = tf.summary.FileWriter(tboard_save_path)
    # summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    log.info('Global configuration is as follows:')
    log.info(CFG)

    result_dir = 'test_jingjian_output/'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with sess.as_default():

        # tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
        #                      name='{:s}/lanenet_model.pb'.format(model_save_dir))


        log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
        saver.restore(sess=sess, save_path=weights_path)



        train_cost_time_mean = []
        val_cost_time_mean = []
        for epoch in range(train_epochs):
            # training part
            t_start = time.time()
            namenum+=3


            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)
            # summary_writer.add_summary(summary=train_summary, global_step=epoch)

            # validation part
            gt_imgs_val, binary_gt_labels_val, instance_gt_labels_val ,gt_name\
                = val_dataset.next_batch(CFG.TEST.BATCH_SIZE)
            gt_imgs_val = [cv2.resize(tmp,
                                      dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                      dst=tmp,
                                      interpolation=cv2.INTER_LINEAR)
                           for tmp in gt_imgs_val]
            gt_imgs_val = [tmp - VGG_MEAN for tmp in gt_imgs_val]
            binary_gt_labels_val = [cv2.resize(tmp,
                                               dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                               dst=tmp)
                                    for tmp in binary_gt_labels_val]
            binary_gt_labels_val = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels_val]
            instance_gt_labels_val = [cv2.resize(tmp,
                                                 dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                                 dst=tmp,
                                                 interpolation=cv2.INTER_NEAREST)
                                      for tmp in instance_gt_labels_val]
            phase_val = 'test'

            t_start_val = time.time()
            print("test part")

            c_val, binary_seg_img = \
                sess.run([total_loss,out_logits_out ],
                         feed_dict={input_tensor: gt_imgs_val,
                                    binary_label: binary_gt_labels_val,
                                    instance_label: instance_gt_labels_val,
                                    early_drop_prob: 0,
                                    later_drop_prob: 0,
                                    phase: phase_val})

            time_end = (time.time()- t_start)/CFG.TEST.BATCH_SIZE
            for pp in range(CFG.TEST.BATCH_SIZE):



                (filepath, tempfilename) = os.path.split(gt_name[pp])
                name = result_dir+tempfilename
                print("saving ",name)
                mergepic(gt_imgs_val[pp],binary_seg_img[pp])
                gt_imgs_val[pp]=cv2.resize(gt_imgs_val[pp],(CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(name, gt_imgs_val[pp]+ VGG_MEAN)
                print("num = ",namenum)


            cost_time_val = (time.time() - t_start_val)/CFG.TEST.BATCH_SIZE

            print("cost_time_val",cost_time_val)
            print("end_time_val", time_end)





    sess.close()

    return



if __name__ == '__main__':
    # init args
    args = init_args()

    # train lanenet
    train_net(args.dataset_dir, args.weights_path, net_flag=args.net)