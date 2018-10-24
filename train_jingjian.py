#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-18 下午7:31
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : train_lanenet.py
# @IDE: PyCharm Community Edition
"""
训练lanenet模型
"""
import argparse
import math
import os
import os.path as ops
import time

import cv2
import glog as log
import numpy as np
import tensorflow as tf
import pdb
try:
    from cv2 import cv2
except ImportError:
    pass
import sys
sys.path.append("..")
from config import global_config
from lanenet_model import lanenet_merge_model
from data_provider import lanenet_data_processor

from tensorflow.python import debug as tf_debug

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir',type=str, help='The training dataset dir path',default='data/training_data_example/')
    parser.add_argument('--net','--enet',type=str, help='Which base net work to use', default='enet')
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
    train_dataset_file = ops.join(dataset_dir, 'train.txt')
    val_dataset_file = ops.join(dataset_dir, 'val.txt')

    assert ops.exists(train_dataset_file)

    train_dataset = lanenet_data_processor.DataSet(train_dataset_file)
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
    # disc_loss = compute_ret['discriminative_loss']
    # pix_embedding = compute_ret['instance_seg_logits']

    # calculate the accuracy
    out_logits = compute_ret['binary_seg_logits']
    out_logitss = tf.nn.softmax(logits=out_logits)
    print("++++++++++++out_logits",out_logits)
    out_logits_out = tf.argmax(out_logitss, axis=-1)
    print("++++++++++++out_logits_out", out_logits_out)
    out = tf.argmax(out_logitss, axis=-1)
    out = tf.expand_dims(out, axis=-1)#在最后增加一个维度
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
    model_save_dir = 'model/culane_lanenet'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)

    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'culane_lanenet_{:s}_{:s}.ckpt'.format(net_flag, str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)


    # Set tf summary
    tboard_save_path = 'tboard/culane_lanenet/{:s}_{:s}'.format(net_flag,str(train_start_time))
    outputimages_path = tboard_save_path+"/outputimages/"
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    if not os.path.exists(outputimages_path):
        os.makedirs(outputimages_path)
    train_cost_scalar = tf.summary.scalar(name='train_cost', tensor=total_loss)
    val_cost_scalar = tf.summary.scalar(name='val_cost', tensor=total_loss)
    train_accuracy_scalar = tf.summary.scalar(name='train_accuracy', tensor=accuracy)
    val_accuracy_scalar = tf.summary.scalar(name='val_accuracy', tensor=accuracy)
    train_binary_seg_loss_scalar = tf.summary.scalar(name='train_binary_seg_loss', tensor=binary_seg_loss)
    val_binary_seg_loss_scalar = tf.summary.scalar(name='val_binary_seg_loss', tensor=binary_seg_loss)
    # train_instance_seg_loss_scalar = tf.summary.scalar(name='train_instance_seg_loss', tensor=disc_loss)
    # val_instance_seg_loss_scalar = tf.summary.scalar(name='val_instance_seg_loss', tensor=disc_loss)
    learning_rate_scalar = tf.summary.scalar(name='learning_rate', tensor=learning_rate)

    print("---------------------------------------------------------------------------------")
    train_merge_summary_op = tf.summary.merge([train_accuracy_scalar, train_cost_scalar,
                                               learning_rate_scalar]
                                              )
    val_merge_summary_op = tf.summary.merge([val_accuracy_scalar, val_cost_scalar
                                           ])

    # Set sess configuration
    sess_config = tf.ConfigProto(device_count={'GPU': 1})

    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    log.info('Global configuration is as follows:')
    log.info(CFG)

    with sess.as_default():


        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                            name='{:s}/lanenet_model.pb'.format(model_save_dir))

        if weights_path is None:
            log.info('Training from scratch')
            init = tf.global_variables_initializer()
            print("Training from scratch")
            sess.run(init)
        else:
            log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        ## 加载tfdebugger,用于调试
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        # # 加载预训练参数
        # if net_flag == 'vgg':
        #     pretrained_weights = np.load(
        #         './data/vgg16.npy',
        #         encoding='latin1').item()
        #
        #     for vv in tf.trainable_variables():
        #         weights_key = vv.name.split('/')[-3]
        #         try:
        #             weights = pretrained_weights[weights_key][0]
        #             _op = tf.assign(vv, weights)
        #             sess.run(_op)
        #         except Exception as e:
        #             continue
        # saver.restore(sess=sess, save_path=weights_path)
        train_cost_time_mean = []
        val_cost_time_mean = []
        for epoch in range(train_epochs):
            # training part
            t_start = time.time()

            gt_imgs, binary_gt_labels, instance_gt_labels ,gt_name= train_dataset.next_batch(CFG.TRAIN.BATCH_SIZE)



            gt_imgs = [cv2.resize(tmp,
                                  dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                  dst=tmp,
                                  interpolation=cv2.INTER_LINEAR)
                       for tmp in gt_imgs]
            gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]
            binary_gt_labels = [cv2.resize(tmp,
                                           dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                           dst=tmp,
                                           interpolation=cv2.INTER_NEAREST)
                                for tmp in binary_gt_labels]
            binary_gt_labels = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels]
            instance_gt_labels = [cv2.resize(tmp,
                                             dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                             dst=tmp,
                                             interpolation=cv2.INTER_NEAREST)
                                  for tmp in instance_gt_labels]



            phase_train = 'train'
            print("training")

            _, c, train_accuracy, train_summary, binary_loss,  binary_seg_img = \
                sess.run([optimizer, total_loss,
                          accuracy,
                          train_merge_summary_op,
                          binary_seg_loss,
                          # disc_loss,
                          # pix_embedding,
                          # out_logits,
                          # out_logitss,
                          out_logits_out],
                         feed_dict={input_tensor: gt_imgs,
                                    binary_label: binary_gt_labels,

                                    early_drop_prob: 0.01,
                                    later_drop_prob: 0.1,
                                    phase: phase_train})

            # print('out_logits',aa[0][0][0])
            # print('out_logitss',bb[0][0][0])
            # print('out_logits_out',binary_seg_img[0][0][0])
            # print('')

            #pdb.set_trace()
            # print("c=",c)
            # instance_loss =1.0
            # binary_loss=1.0
            # if math.isnan(c) or math.isnan(binary_loss) or math.isnan(instance_loss):
            #     log.error('cost is: {:.5f}'.format(c))
            #     log.error('binary cost is: {:.5f}'.format(binary_loss))
            #     log.error('instance cost is: {:.5f}'.format(instance_loss))
            #     cv2.imwrite('./outputimages/nan_image.png', gt_imgs[0] + VGG_MEAN)
            #     cv2.imwrite('./outputimages/nan_instance_label.png', instance_gt_labels[0])
            #     cv2.imwrite('./outputimages/nan_binary_label.png', binary_gt_labels[0] * 255)
            #     cv2.imwrite('./outputimages/nan_embedding.png', embedding[0])
            #     return
            if epoch % 800 == 0:
                # (filepath, tempfilename) = os.path.split(gt_name[0])
                # name = "./outputimages/" + tempfilename
                name = outputimages_path + str(epoch)+'.png'
                print("saving ", name)
                # if len(a)<300:
                #     print("有问题的图片",name)
                #     name2="./test2/"+gt_name+"error"+".png"
                #     cv2.imwrite(name2,binary_seg_img[pp]*255)
                result = mergepic(gt_imgs[0], binary_seg_img[0])
                cv2.imwrite(name, result + VGG_MEAN)



            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)
            summary_writer.add_summary(summary=train_summary, global_step=epoch)

            # validation part
            gt_imgs_val, binary_gt_labels_val, instance_gt_labels_val ,gt_name_val= val_dataset.next_batch(CFG.TRAIN.VAL_BATCH_SIZE)
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

            c_val, val_summary, val_accuracy, val_binary_seg_loss = \
                sess.run([total_loss, val_merge_summary_op, accuracy, binary_seg_loss ],
                         feed_dict={input_tensor: gt_imgs_val,
                                    binary_label: binary_gt_labels_val,

                                    early_drop_prob: 0,
                                    later_drop_prob: 0,
                                    phase: phase_val})

            # if epoch % 100 == 0:
            #     cv2.imwrite('./outputimages/test_image.png', gt_imgs_val[0] + VGG_MEAN)

            summary_writer.add_summary(val_summary, global_step=epoch)

            cost_time_val = time.time() - t_start_val
            val_cost_time_mean.append(cost_time_val)

            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                log.info('Epoch: {:d} total_loss= {:6f} binary_seg_loss= {:6f}  accuracy= {:6f}'
                         ' mean_cost_time= {:5f}s '.
                         format(epoch + 1, c, binary_loss,  train_accuracy,
                                np.mean(train_cost_time_mean)))
                train_cost_time_mean.clear()
            '''
            if epoch % CFG.TRAIN.TEST_DISPLAY_STEP == 0:
                log.info('Epoch_Val: {:d} total_loss= {:6f} binary_seg_loss= {:6f} '
                         'instance_seg_loss= {:6f} accuracy= {:6f} '
                         'mean_cost_time= {:5f}s '.
                         format(epoch + 1, c_val, val_binary_seg_loss, val_instance_seg_loss, val_accuracy,
                                np.mean(val_cost_time_mean)))
                val_cost_time_mean.clear()
            '''
            if epoch % 200 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # train lanenet
    train_net(args.dataset_dir, args.weights_path, net_flag=args.net)
