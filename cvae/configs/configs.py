# Copyright (C) 2018 Zhixian MA <zx@mazhixian.me>
# MIT liscence

"""Configurations for the DNNAE network."""

import tensorflow as tf

class config_mnist(object):
    rs = 28
    inputs = tf.placeholder(dtype=tf.float32, shape=(None,rs**2), name='x_in')
    outputs = tf.placeholder(dtype=tf.float32, shape=(None,rs**2), name='x_out')
    numclass = 10
    conditions = tf.placeholder(dtype=tf.float32, shape=(None,numclass), name='labels')
    cflag = True
    z_length = 16
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    layers = [512, 256]
    actfun = [tf.nn.relu, tf.nn.relu]
    batchflag = [True, True]
    loss_mse = True


class config_train(object):
    valrate = 0.2
    batchsize = 64
    epochs = 50000
    lr_init = 0.0001
    decay_rate = 0.95
    keep_prob = 0.5
    print_step = 1000
