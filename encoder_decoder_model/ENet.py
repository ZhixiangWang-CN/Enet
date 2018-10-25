import tensorflow as tf
import os

from encoder_decoder_model.cnn_basenet import CNNBaseModel


class Enet_model(CNNBaseModel):

    def __int__(self, img_height=720, img_width=1280, batch_size=4):
        
        super(Enet_model, self).__init__()

        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

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

    def encoder_bottleneck_regular(self, x, output_depth,
                                   drop_prob, scope, proj_ratio=4, downsampling=False):
        print("===============encoder_bottleneck_regular")
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]
        print("======input_shape",input_shape)
        internal_depth = int(input_depth / proj_ratio)

        # convolution branch
        conv_branch = x

        # # 1x1 projection
        if downsampling:

            W_conv = self.get_variable_weight_decay(scope + "/W_proj",
                                                    shape=[2, 2, input_depth, internal_depth],
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    loss_category="encoder_wd_loss")
            conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 2, 2, 1],
                                       padding="VALID")  # NOTE! there is no bias term in any projection
            print("======conv_branch (downsampling) ", conv_branch.get_shape().as_list())  # 卷积后
        else:
            W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                                                    shape=[1, 1, input_depth, internal_depth],
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    loss_category="encoder_wd_loss")
            conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1], padding="VALID")
            print("======conv_branch", conv_branch.get_shape().as_list())  # 卷积后

        # # # BN and PReLU
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = self.PReLU(conv_branch, scope=scope + "/proj")
        #print("======BN and PReLU", conv_branch.get_shape().as_list())  # 卷积后
        # # convolution
        W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                                                shape=[3, 3, internal_depth, internal_depth],
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                loss_category="encoder_wd_loss")
        b_conv = self.get_variable_weight_decay(scope + "/b_conv",
                                                shape=[internal_depth],
                                                initializer=tf.constant_initializer(0),
                                                loss_category="encoder_wd_loss")
        conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 1, 1, 1], padding="SAME") + b_conv
        print("======convolution", conv_branch.get_shape().as_list())  # 卷积后
        # # # BN and PReLU
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = self.PReLU(conv_branch, scope=scope + "/conv")
        #print("======BN and PReLU", conv_branch.get_shape().as_list())  # 卷积后
        # # 1x1 expansion
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                                               shape=[1, 1, internal_depth, output_depth],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               loss_category="encoder_wd_loss")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1], padding="VALID")
        print("======1x1 expansion", conv_branch.get_shape().as_list())  # 卷积后
        # # # BN
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)  # NOTE! No PReLU!

        # # regularizer
        selfconv_branch = self.spatial_dropout(conv_branch, drop_prob)

        # main branch
        main_branch = x

        if downsampling:

            main_branch, pooling_indices = self.max_pool_with_argmax(main_branch, stride=2)
            # pad with zeros so that the feature block depth matches
            depth_to_pad = output_depth - input_depth
            paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, depth_to_pad]])
            main_branch = tf.pad(main_branch, paddings=paddings, mode="CONSTANT")
            print("======main_branch(downsampling)", main_branch.get_shape().as_list())  # 卷积后
        # add the branchs
        merged = conv_branch + main_branch
        print("======add the branchs", merged.get_shape().as_list())  # 卷积后
        # apply PRelu
        output = self.PReLU(merged, scope=scope + "/output")
        print("======output", output.get_shape().as_list())  # 卷积后
        if downsampling:
            return output, pooling_indices
        else:
            return output

    def encoder_bottleneck_dilated(self, x, output_depth, drop_prob, scope, dilation_rate, proj_ratio=4):
        print("encoder_bottleneck_dilated")
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]
        print("======input_shape",input_shape)
        internal_depth = int(input_depth / proj_ratio)

        # convolution branch
        conv_branch = x

        # # 1x1 projection
        W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                                                shape=[1, 1, input_depth, internal_depth],
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                loss_category="encoder_wd_loss")
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1], padding="VALID")
        print("======1x1 projection", conv_branch.get_shape().as_list())  # 卷积后
        # # # batch normalization and PReLU
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = self.PReLU(conv_branch, scope=scope + "/proj")

        # # dilated conv
        W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                                                shape=[3, 3, internal_depth, internal_depth],
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                loss_category="encoder_wd_loss")
        b_conv = self.get_variable_weight_decay(scope + "b_conv",
                                                shape=[internal_depth],
                                                initializer=tf.constant_initializer(0),
                                                loss_category="encoder_wd_loss")
        conv_branch = tf.nn.atrous_conv2d(conv_branch, W_conv, rate=dilation_rate, padding="SAME") + b_conv
        print("======dilated conv", conv_branch.get_shape().as_list())  # 卷积后
        # # # batch normalization and PReLU
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = self.PReLU(conv_branch, scope=scope + "/conv")

        # # 1x1 expansion
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                                               shape=[1, 1, internal_depth, output_depth],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               loss_category="encoder_wd_loss")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1], padding="VALID")
        print("======1x1 projection", conv_branch.get_shape().as_list())  # 卷积后
        # # # batch normalization
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)

        # # regularizer
        conv_branch = self.spatial_dropout(conv_branch, drop_prob=drop_prob)

        # main branch
        main_branch = x
        print("======main branch", main_branch.get_shape().as_list())  # 卷积后
        # merge the branches
        merged = conv_branch + main_branch
        print("======merge the branches", merged.get_shape().as_list())  # 卷积后
        # apply PReLU
        output = self.PReLU(merged, scope=scope + "/output")
        print("======output", output.get_shape().as_list())  # 卷积后
        return output

    def encoder_bottleneck_asymmetric(self, x, output_depth, drop_prob, scope, proj_ratio=4):
        print("encoder_bottleneck_asymmetric")
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]
        print("======input_shape", input_shape)
        internal_depth = int(input_depth / proj_ratio)

        # convolution branch
        conv_branch = x
        # # 1x1 projection
        W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                                                shape=[1, 1, input_depth, internal_depth],
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                loss_category="encoder_wd_loss")
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1], padding="VALID")
        print("======1x1 projection", conv_branch.get_shape().as_list())  # 卷积后
        # # # batch norm and PReLU
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = self.PReLU(conv_branch, scope=scope + "/proj")

        # # asymmetric conv
        # # # asymmetric conv1
        W_conv1 = self.get_variable_weight_decay(scope + "/W_conv1",
                                                 shape=[5, 1, internal_depth, internal_depth],
                                                 initializer=tf.contrib.layers.xavier_initializer(),
                                                 loss_category="encoder_wd_loss")
        conv_branch = tf.nn.conv2d(conv_branch, W_conv1, strides=[1, 1, 1, 1], padding="SAME")
        print("======asymmetric conv", conv_branch.get_shape().as_list())  # 卷积后
        # # # asymmetric conv2
        W_conv2 = self.get_variable_weight_decay(scope + "/W_conv2",
                                                 shape=[1, 5, internal_depth, internal_depth],
                                                 initializer=tf.contrib.layers.xavier_initializer(),
                                                 loss_category="encoder_wd_loss")

        b_conv2 = self.get_variable_weight_decay(scope + "/b_conv2",
                                                 shape=[internal_depth],
                                                 initializer=tf.constant_initializer(0),
                                                 loss_category="encoder_wd_loss")
        conv_branch = tf.nn.conv2d(conv_branch, W_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2
        print("======asymmetric conv2", conv_branch.get_shape().as_list())  # 卷积后
        # # # batch norm and PReLU
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = self.PReLU(conv_branch, scope=scope + "/conv")

        # # 1x1 expansion
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                                               shape=[1, 1, internal_depth, output_depth],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               loss_category="encoder_wd_loss")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1], padding="VALID")
        print("======1x1 projection", conv_branch.get_shape().as_list())  # 卷积后
        # # # batch norm
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)

        # # regularizer
        conv_branch = self.spatial_dropout(conv_branch, drop_prob=drop_prob)

        # main branch
        main_branch = x

        # merge the branches
        merged = conv_branch + main_branch
        print("======merge the branches", merged.get_shape().as_list())  # 卷积后
        # apply PReLU
        output = self.PReLU(merged, scope=scope + "/output")
        print("======output", output.get_shape().as_list())  # 卷积后
        return output

    def decoder_bottleneck(self, x, output_depth, scope, proj_ratio=4,
                           upsampling=False, pooling_indices=None):
        print("decoder_bottleneck")
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]
        print("======input_shape",input_shape)
        internal_depth = int(input_depth / proj_ratio)

        # main branch
        main_branch = x

        if upsampling:
            # # 1x1 projection (to decrease depth to the same value as before downsampling)
            W_upsample = self.get_variable_weight_decay(scope + "/W_upsample",
                                                        shape=[1, 1, input_depth, output_depth],
                                                        initializer=tf.contrib.layers.xavier_initializer(),
                                                        loss_category="decoder_wd_loss")
            main_branch = tf.nn.conv2d(main_branch, W_upsample, strides=[1, 1, 1, 1], padding="VALID")
            print("======# 1x1 projection (upsampling)", main_branch.get_shape().as_list())  # 卷积后
            # # # batch norm
            main_branch = tf.contrib.slim.batch_norm(main_branch)

            # # max unpooling
            main_branch = self.max_unpool(main_branch, pooling_indices, stride=2)
            print("======# max unpooling", main_branch.get_shape().as_list())  # 卷积后
        main_branch = tf.cast(main_branch, tf.float32)

        # conv branch
        conv_branch = x
        print("======# conv branch ",conv_branch.get_shape().as_list())
        # # 1x1 projection
        W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                                                shape=[1, 1, input_depth, internal_depth],
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                loss_category="decoder_wd_loss")
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1], padding="VALID")
        print("======# # 1x1 projection ", conv_branch.get_shape().as_list())
        # # # batch norm and ReLU
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = tf.nn.relu(conv_branch)

        # # conv
        if upsampling:
            # deconvolution
            W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                                                    shape=[3, 3, internal_depth, internal_depth],
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    loss_category="decoder_wd_loss")
            b_conv = self.get_variable_weight_decay(scope + "/b_conv",
                                                    shape=[internal_depth],
                                                    initializer=tf.constant_initializer(0),
                                                    loss_category="decoder_wd_loss")
            main_branch_shape = main_branch.get_shape().as_list()
            output_shape = tf.convert_to_tensor([main_branch_shape[0],
                                                 main_branch_shape[1],
                                                 main_branch_shape[2],
                                                 internal_depth])
            conv_branch = tf.nn.conv2d_transpose(conv_branch, W_conv, output_shape=output_shape,
                                                 strides=[1, 2, 2, 1], padding="SAME") + b_conv
            print("======# # conv upsampling ", conv_branch.get_shape().as_list())
        else:
            W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                                                    shape=[3, 3, internal_depth, internal_depth],
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    loss_category="decoder_wd_loss")
            b_conv = self.get_variable_weight_decay(scope + "b_conv",
                                                    shape=[internal_depth],
                                                    initializer=tf.constant_initializer(0),
                                                    loss_category="decoder_wd_loss")
            conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 1, 1, 1], padding="SAME") + b_conv
            print("======# # conv  ", conv_branch.get_shape().as_list())
        # # # batch norm and ReLU
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = tf.nn.relu(conv_branch)

        # # 1x1 expansion
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                                               shape=[1, 1, internal_depth, output_depth],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               loss_category="decoder_wd_loss")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1], padding="VALID")
        print("====== # # 1x1 expansion  ", conv_branch.get_shape().as_list())
        # # # batch norm
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)

        # NOTE! No regularizer

        # add the batches
        merged = conv_branch + main_branch
        print("====== # #add the batches  ", merged.get_shape().as_list())
        # apply ReLU
        output = tf.nn.relu(merged)
        print("====== # # output  ", output.get_shape().as_list())
        return output

    def get_variable_weight_decay(self, name, shape, initializer, loss_category, dtype=tf.float32):
        variable = tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer)

        return variable


if __name__ == '__main__':
    # input = tf.placeholder(dtype=tf.float32, shape=[1, 640, 320, 16])
    input_tensor = tf.placeholder(dtype=tf.float32,
                                  shape=[4, 256,
                                         512, 3], name='input_tensor')
    # input = tf.placeholder(dtype=tf.float32, shape=[16, 256, 256])
    # drop_prob = tf.placeholder(dtype=tf.float32)
    net = Enet_model()

    # main_branch, conv_branch, output, pooling_indices = net.encoder_bottleneck_regular(x=input, output_depth=64,
    #                                                                                    drop_prob=drop_prob, scope="test",
    #                                                                                    proj_ratio=4, downsampling=True)
    main_branch, conv_branch, output,pooling_indices   = net.initial_block(x=input,scope="test")
    print("main_branch = %s" % main_branch.get_shape().as_list())
    print("conv_branch = %s" % conv_branch.get_shape().as_list())
    print("output = %s" % output.get_shape().as_list())
    print("pooling_indices = %s" % pooling_indices.get_shape().as_list())
