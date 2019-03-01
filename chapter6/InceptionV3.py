#!/usr/bin/env python
#-*-coding:utf-8-*-
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="2"


slim = tf.contrib.slim
trunc_norm = lambda stddev: tf.truncated_normal_initializer(0.0,stddev)

def inception_v3_arg_scope(weight_decay=0.00004,
                           stddev=0.1,
                           batch_norm_var_collection="moving_vars"):
    """
    :param weight_decay: L2正则的weight_decay
    :param stddev: 标准差默认值
    :param batch_norm_var_collection:
    :return:
    """
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collection': tf.GraphKeys.UPDATE_OPS,
        'variables_collections':{
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }

    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        weights_regularizer = slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.conv2d],
            weights_initializer = tf.truncated_normal_initializer(stddev=stddev),
            activation_fn = tf.nn.relu,
            normalizer_fn = slim.batch_norm,
            normalizer_params = batch_norm_params) as sc:
            return sc

def inception_v3_base(inputs,scope=None):

    end_points = {}
    with tf.variable_scope(scope,'InceptionV3',[inputs]):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                            stride=1,padding='VALID'):
            #slim.conv2d的五个参数 输入的tensor，输出的通道数，卷积核尺寸，步长，padding模式
            net = slim.conv2d(inputs,32,[3,3],stride=2,scope='Conv2d_1a_3x3')
            net = slim.conv2d(net,32,[3,3],scope='Conv2d_2a_3x3')
            net = slim.conv2d(net,64,[3,3],padding='SAME',
                              scope='Conv2d_2b_3x3')
            net = slim.max_pool2d(net,[3,3],stride=2,scope='MaxPool_3a_3x3')
            net = slim.conv2d(net,80,[1,1],scope='Conv2d_3b_1x1')
            net = slim.conv2d(net,192,[3,3],scope='Conv2d_4a_3x3')
            net = slim.max_pool2d(net,[3,3],stride=2,scope='MaxPool_5a_3x3')

            #用slim.arg_scope设置所有Inception模块组的参数，包括卷积层，最大池化层，平均池化层，padding的模式
            with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='SAME'):
                with tf.variable_scope('Mixed_5b'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,48,[1,1],scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1,64,[5,5],scope='Conv2d_0b_5x5')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0b_3x3')
                        branch_2 = slim.conv2d(brach_2,96,[3,3],scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3,32,[1,1],scope='Conv2d_0b_1x1')

                    #对输出的通道进行拼接 在第四个维度
                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

                with tf.variable_scope('Mixed_5c'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,48,[1,1],scope='Conv2d_0b_1x1')
                        branch_1 = slim.conv2d(branch_1,64,[5,5],scope='Conv_1_0c_5x5')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0b_3x3')
                        branch_2 = slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3,64,[1,1],scope='Conv2d_0b_1x1')

                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

                with tf.variable_scope('Mixed_5d'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,48,[1,1],scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1,64,[5,5],scope='Conv2d_0b_5x5')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0b_3x3')
                        branch_2 = slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3,64,[1,1],scope='Conv2d_0b_1x1')

                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

                with tf.variable_scope('Mixed_6a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,384,[3,3],stride=2,
                                               padding='VALID',scope='Conv2d_1a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1,96,[3,3],scope='Conv2d_0b_3x3')
                        branch_1 = slim.conv2d(branch_1,96,[3,3],stride=2,
                                               padding='VALID',scope='Conv2d_1a_1x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.max_pool2d(net,[3,3],stride=2,padding='VALID',
                                                   scope='MaxPool_1a_3x3')

                    net = tf.concat([branch_0,branch_1,branch_2,3])

                with tf.variable_scope('Mixed_6b'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,128,[1,1],scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1,128,[1,7],
                                               scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1,192,[7,1],
                                               scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,128,[1,1],scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2,128,[7,1],scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2,128,[1,7],scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2,128,[7,1],scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2,128,[1,7],scope='Conv2d_0e_1x7')

                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3,192,[1,1],scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
                with tf.variable_scope('Mixed_6c'):
                    with tf.variable_scope("Branch_0"):
                        branch_0 = slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')
                    with tf.variable_scope(Branch_1):
                        branch_1 = slim.conv2d(net,160,[1,1],scope='Con2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1,160,[1,7],scope='Con2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1,192,[7,1],scope='Conv2d_0c_7x1')

                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,160,[1,1],scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2,160,[7,1],scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2,160,[1,7],scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2,160,[7,1],scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2,160,[1,7],scope='Conv2d_0e_1x7')

                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope='Avg_0a_3x3')
                        branch_3 = slim.conv2d(branch_3,192,[1,1],scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

                with tf.variable_scope('Mixed_6d'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,160,[1,1],scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1,160,[1,7],scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1,192,[7,1],scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,160,[1,1],scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2,160,[7,1],scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2,160,[1,7],scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2,160,[7,1],scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2,192,[1,7],scope='Conv2d_0e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope='Avg_0a_3x3')
                        branch_3 = slim.conv2d(branch_3,[1,1],scope='Conv2d_0b_1x1')

                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)



                with tf.variable_scope('Mixed_6e'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1,192,[1,7],scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1,192,[7,1],scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2,192,[7,1],scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2,192,[1,7],scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2,192,[7,1],scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2,192,[1,7],scope='Conv2d_0e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope='Avg_0a_3x3')
                        branch_3 = slim.conv2d(branch_3,[1,1],scope='Conv2d_0b_1x1')

                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
                end_points['Mixed_6e'] = net


                #中间的卷积过程中stride和padding发生变化，需要添加参数
                with tf.variable_scope('Mixed_7a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')
                        branch_0 = slim.conv2d(branch_0,320,[3,3],stride=2,
                                               padding='VALID',scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1,192,[1,7],scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1,192,[7,1],scope='Conv2d_0c_7x1')
                        branch_1 = slim.conv2d(branch_1,192,[3,3],stride=2,padding='VALID',
                                               scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.max_pool2d(net,[3,3],stride=2,padding='VALID',
                                                   scope='MaxPool_1a_3x3')
                    net = tf.concat([branch_0,branch_1,branch_2])

                with tf.variable_scope('Mixed_7b'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,320,[1,1],scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,384,[1,1],scope='Conv2d_0a_1x1')
                        branch_1 = tf.concat([
                            slim.conv2d(branch_1,384,[1,3],scope='Conv2d_0b_1x3'),
                            slim.conv2d(branch_1,384,[3,1],scope='Conv_0b_3x1')],3)
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,448,[1,1],scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2,384,[3,3],scope='Conv2d_0b_3x3')
                        tf.concat([
                            slim.conv2d(branch_2,384,[1,3],scope='Conv2d_0c_1x3'),
                            slim.conv2d(branch_2,384,[3,1],scope='Conv2d_0d_3x1)')
                        ],3)
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3,192,[1,1],scope='Conv_0b_1x1')
                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

                with tf.variable_scope('Mixed_7c'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,320,[1,1],scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,384,[1,1],scope='Conv2d_0a_1x1')
                        branch_1 = tf.concat([
                            slim.conv2d(branch_1,384,[1,3],scope='Conv2d_0b_1x3'),
                            slim.conv2d(branch_1,384,[3,1],scope='Conv_0b_3x1')],3)
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,448,[1,1],scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2,384,[3,3],scope='Conv2d_0b_3x3')
                        tf.concat([
                            slim.conv2d(branch_2,384,[1,3],scope='Conv2d_0c_1x3'),
                            slim.conv2d(branch_2,384,[3,1],scope='Conv2d_0d_3x1)')
                        ],3)
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3,192,[1,1],scope='Conv_0b_1x1')
                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
    return net,end_points

def inception_v3(inputs,num_classes=1000,is_training=True,
                 dropout_keep_prob=0.8,prediction_fn=slim.softmax,
                 spatial_squeeze=True,reuse=None,scope='InceptionV3'):
    """
    :param inputs:
    :param numclasses:分类的数量
    :param is_training: 是否是训练过程
    :param dropout_keep_prob: Dropout所需保留节点的比例
    :param prediction_fn: 最后用来进行分类的函数 默认使用slim.softmax
    :param spatial_squeeze: 参数标志是否对输出进行squeeze操作(去除维数为1的维度)
    :param reuse: 是否会对网络和Variable进行重复使用
    :param scope: 为包含了函数默认参数的环境
    :return:
    """


    with tf.variable_scope(scope,'InceptionV3',[inputs,num_classes],
                           reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm,slim.dropout],
                            is_training=is_training):
            net,end_points = inception_v3_base(inputs,scope)

        #Auxiliary Logits是作为辅助分类的节点，对分类结果预测有帮助a
        #用slim.arg_scope将卷积、最大池化、平均池化的默认步长设为1，默认padding模式设为SAME
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='SAME'):
            aux_logits = end_points['Mixed_6e']
            with tf.variable_scope('AuxLogits'):
                aux_logits = slim.avg_pool2d(
                    aux_logits,[5,5],stride=3,padding='VALID',
                    scope='AvgPool_1a_5x5')
                aux_logits = slim.conv2d(aux_logits,128,[1,1],scope='Conv2d_1b_1x1')

                aux_logits = slim.conv2d(
                    aux_logits,768,[5,5],weights_initializer=trunc_norm(0.01),
                    padding = 'VALID',scope='Conv2d_2a_5x5')
                aux_logits = slim.conv2d(aux_logits,num_classes,[1,1],activation_fn=None,
                                         normalizer_fn=None,weights_initializer=trunc_norm(0.001),
                                         scope='Conv2d_2b_1x1')
                if spatial_squeeze:
                    aux_logits = tf.squeeze(aux_logits,[1,2],name='SpatialSqueeze')
                end_points['AuxLogits'] = aux_logits















