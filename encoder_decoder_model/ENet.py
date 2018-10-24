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
        W_conv = self.get_variable_weight_decay(scope + "/W", shape=[3, 3, 3, 13],
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                loss_category="encoder_wd_loss")
        b_conv = self.get_variable_weight_decay(scope + "/b", shape=[13],
                                                initializer=tf.constant_initializer(0),
                                                loss_category="encoder_wd_loss")
        conv_branch = tf.nn.conv2d(x, W_conv, strides=[1, 2, 2, 1], padding="SAME") + b_conv

        # maxpooling branch
        pool_branch = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        # cancatenate these two branches
        concat = tf.concat([conv_branch, pool_branch], axis=3)  # (3: the depth axis)

        # apply BN and PReLU
        output = tf.contrib.slim.batch_norm(concat)
        output = self.PReLU(output, scope=scope)

        return output

    def encoder_bottleneck_regular(self, x, output_depth,
                                   drop_prob, scope, proj_ratio=4, downsampling=False):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

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
        else:
            W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                                                    shape=[1, 1, input_depth, internal_depth],
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    loss_category="encoder_wd_loss")
            conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1], padding="VALID")

        # # # BN and PReLU
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = self.PReLU(conv_branch, scope=scope + "/proj")

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

        # # # BN and PReLU
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = self.PReLU(conv_branch, scope=scope + "/conv")

        # # 1x1 expansion
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                                               shape=[1, 1, internal_depth, output_depth],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               loss_category="encoder_wd_loss")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1], padding="VALID")

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

        # add the branchs
        merged = conv_branch + main_branch

        # apply PRelu
        output = self.PReLU(merged, scope=scope + "/output")

        if downsampling:
            return output, pooling_indices
        else:
            return output

    def encoder_bottleneck_dilated(self, x, output_depth, drop_prob, scope, dilation_rate, proj_ratio=4):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(input_depth / proj_ratio)

        # convolution branch
        conv_branch = x

        # # 1x1 projection
        W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                                                shape=[1, 1, input_depth, internal_depth],
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                loss_category="encoder_wd_loss")
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1], padding="VALID")

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

        # # # batch normalization and PReLU
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = self.PReLU(conv_branch, scope=scope + "/conv")

        # # 1x1 expansion
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                                               shape=[1, 1, internal_depth, output_depth],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               loss_category="encoder_wd_loss")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1], padding="VALID")

        # # # batch normalization
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)

        # # regularizer
        conv_branch = self.spatial_dropout(conv_branch, drop_prob=drop_prob)

        # main branch
        main_branch = x

        # merge the branches
        merged = conv_branch + main_branch

        # apply PReLU
        output = self.PReLU(merged, scope=scope + "/output")

        return output

    def encoder_bottleneck_asymmetric(self, x, output_depth, drop_prob, scope, proj_ratio=4):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(input_depth / proj_ratio)

        # convolution branch
        conv_branch = x
        # # 1x1 projection
        W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                                                shape=[1, 1, input_depth, internal_depth],
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                loss_category="encoder_wd_loss")
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1], padding="VALID")

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

        # # # batch norm and PReLU
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = self.PReLU(conv_branch, scope=scope + "/conv")

        # # 1x1 expansion
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                                               shape=[1, 1, internal_depth, output_depth],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               loss_category="encoder_wd_loss")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1], padding="VALID")

        # # # batch norm
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)

        # # regularizer
        conv_branch = self.spatial_dropout(conv_branch, drop_prob=drop_prob)

        # main branch
        main_branch = x

        # merge the branches
        merged = conv_branch + main_branch

        # apply PReLU
        output = self.PReLU(merged, scope=scope + "/output")

        return output

    def decoder_bottleneck(self, x, output_depth, scope, proj_ratio=4,
                           upsampling=False, pooling_indices=None):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

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
            # # # batch norm
            main_branch = tf.contrib.slim.batch_norm(main_branch)

            # # max unpooling
            main_branch = self.max_unpool(main_branch, pooling_indices, stride=2)

        main_branch = tf.cast(main_branch, tf.float32)

        # conv branch
        conv_branch = x

        # # 1x1 projection
        W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                                                shape=[1, 1, input_depth, internal_depth],
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                loss_category="decoder_wd_loss")
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1], padding="VALID")
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
        # # # batch norm and ReLU
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = tf.nn.relu(conv_branch)

        # # 1x1 expansion
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                                               shape=[1, 1, internal_depth, output_depth],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               loss_category="decoder_wd_loss")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1], padding="VALID")
        # # # batch norm
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)

        # NOTE! No regularizer

        # add the batches
        merged = conv_branch + main_branch

        # apply ReLU
        output = tf.nn.relu(merged)

        return output

    def get_variable_weight_decay(self, name, shape, initializer, loss_category, dtype=tf.float32):
        variable = tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer)

        return variable


if __name__ == '__main__':
    input = tf.placeholder(dtype=tf.float32, shape=[1, 640, 320, 16])
    drop_prob = tf.placeholder(dtype=tf.float32)
    net = Enet_model()

    main_branch, conv_branch, output, pooling_indices = net.encoder_bottleneck_regular(x=input, output_depth=64,
                                                                                       drop_prob=drop_prob, scope="test",
                                                                                       proj_ratio=4, downsampling=True)
    print("main_branch = %s" % main_branch.get_shape().as_list())
    print("conv_branch = %s" % conv_branch.get_shape().as_list())
    print("output = %s" % output.get_shape().as_list())
    print("pooling_indices = %s" % pooling_indices.get_shape().as_list())
