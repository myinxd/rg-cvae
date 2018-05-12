# Copyright (C) 2018 zhixian MA <zx@mazhixian.me>
# MIT liscence

'''
Conditional variational autoencoder, DNN case.

Characters
==========
1. arbitrary loss function
2. configurable with configurations
3. extendable to general vae
'''

import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm

from cvae.utils import utils


class cvae():
    """
    The conditional variational autoendocer class.

    inputs
    ======
    configs: object class
        configurations for the dnn
        configs.inputs: placeholder of the network's input,
                       whose shape is (None, rows, cols) or (rows*cols).
        configs.conditions: placeholder of the conditions w.r.t. the samples.
        configs.output: placeholder of the network's output of the
                       same shape as the inputs.
        configs.layers: a list of the hidden layers
        configs.actfun: a list of activation functions for the layers
        configs.batchflag: a list of batchly normalization flags for the layers
        configs.cflag: a flag of whether it is cvae or vae
        configs.z_length: length of the latent z
        configs.keep_rate: keep rate for training the network
        configs.init_lr: initialized learning rate
        configs.numclass: number of classes to be classified

    methods
    =======
    cvae_build: build the network
    vae_build: build a general vae network
    cvae_train: train the network
    cvae_print: print the network structure
    cvae_test: test the network
    get_loss: get the loss functions
    get_opt: get the training objective
    cvae_train: train the cvae network
    vae_train: train the vae net
    cvae_train_mnist: train the cvae on the MNIST dataset
    """

    def __init__(self, configs):
        """Initializer"""
        self.inputs = configs.inputs
        self.outputs = configs.outputs
        self.cflag = configs.cflag
        self.keep_prob = configs.keep_prob
        self.numclass = configs.numclass
        self.z_length = configs.z_length
        try:
            self.mseflag = configs.loss_mse
        except:
            self.mseflag = True
        # get input shape
        self.input_shape = self.inputs.get_shape().as_list()
        if len(self.input_shape) == 4:
            self.outlayer = self.input_shape[1]*self.input_shape[2]
            self.net = tf.reshape(
                self.inputs,
                [-1,
                 self.input_shape[1]*self.input_shape[2]])
        elif len(self.input_shape) == 2:
            self.outlayer = self.input_shape[1]
            self.net = self.inputs
        else:
            print("Something wrong on the input shape.")
        # AE flag
        if self.cflag:
            # conditional VAE
            self.layers = configs.layers
            self.conditions = configs.conditions
            self.net = tf.concat(
                [self.net, self.conditions], axis=1)
            self.actfun = configs.actfun
            self.batchflag = configs.batchflag
            self.z = tf.placeholder(
                dtype=tf.float32, shape=[None, self.z_length], name='z')
            self.mu = tf.placeholder(
                dtype=tf.float32, shape=[None, self.z_length], name='en_avg')
            self.sigma = tf.placeholder(
                dtype=tf.float32, shape=[None, self.z_length], name='en_std')
        else:
            # VAE
            self.layers = configs.layers
            self.actfun = configs.actfun
            self.batchflag = configs.batchflag
            self.z = tf.placeholder(
                dtype=tf.float32, shape=[None, self.z_length], name='z')
            self.mu = tf.placeholder(
                dtype=tf.float32, shape=[None, self.z_length], name='en_avg')
            self.sigma = tf.placeholder(
                dtype=tf.float32, shape=[None, self.z_length], name='en_std')
        # batch normalization
        self.is_training = tf.placeholder(tf.bool, name='is_training')


    def vae_build(self):
        """Build the general vae network"""
        self.netprinter = []
        self.netprinter.append(["Input layer",
                                self.net.get_shape().as_list()])
        # The encoder part
        with tf.name_scope("cave"):
            with tf.name_scope("fc_en"):
                for i, layer in enumerate(self.layers):
                    self.net = fully_connected(
                        inputs=self.net,
                        num_outputs = layer,
                        activation_fn = self.actfun[i])
                    self.netprinter.append(
                        ["FC layer " + str(i), self.net.get_shape().as_list()])
                    # batch normalization
                    if self.batchflag[i]:
                        self.net = batch_norm(
                            inputs=self.net,
                            center=True,
                            scale=True,
                            is_training=self.is_training)
                        self.netprinter.append(
                            ["BN layer " + str(i), self.net.get_shape().as_list()])
                    else:
                        # dropout
                        self.net = tf.nn.dropout(
                            x=self.net,
                            keep_prob=self.keep_prob,
                            name="drop_"+str(i))
                        self.netprinter.append(
                            ["Dropout layer " + str(i), self.net.get_shape().as_list()])
            # outputs
            with tf.name_scope("en_output"):
                self.mu = fully_connected(
                    inputs=self.net,
                    num_outputs=self.z_length,
                    activation_fn=None)
                self.sigma=fully_connected(
                    inputs=self.net,
                    num_outputs=self.z_length,
                    activation_fn=None)
                # softplus
                self.sigma=1e-6 + tf.nn.softplus(self.sigma)
                self.netprinter.append(["En_mu", self.mu.get_shape().as_list()])
                self.netprinter.append(["En_sigma", self.sigma.get_shape().as_list()])

            # Reparameterization to obtain z
            with tf.name_scope("reparameterization"):
                self.epsilon = tf.random_normal(tf.shape(self.mu))
                self.z = self.mu + tf.multiply(self.epsilon, self.sigma)
                self.netprinter.append(["z", self.z.get_shape().as_list()])

            # The decoder subnet
            self.net = self.z
            # the decoder
            with tf.name_scope("fc_de"):
                for i in range(len(self.layers)-1,-1,-1):
                    self.net = fully_connected(
                        inputs=self.net,
                        num_outputs = self.layers[i],
                        activation_fn = self.actfun[i])
                    self.netprinter.append(
                        ["FC layer " + str(i), self.net.get_shape().as_list()])
                    # batch normalization
                    if self.batchflag[i]:
                        self.net = batch_norm(
                            inputs=self.net,
                            center=True,
                            scale=True,
                            is_training=self.is_training)
                        self.netprinter.append(
                            ["BN layer " + str(i), self.net.get_shape().as_list()])
                    else:
                        # dropout
                        self.net = tf.nn.dropout(
                            x=self.net,
                            keep_prob=self.keep_prob,
                            name="drop_"+str(i)
                        )
                        self.netprinter.append(
                            ["Dropout layer " + str(i), self.net.get_shape().as_list()])
                # Output
                if self.mseflag:
                    self.net = fully_connected(
                        inputs=self.net,
                        num_outputs = self.outlayer,
                        activation_fn = tf.nn.sigmoid)
                else:
                    self.net_mse = fully_connected(
                        inputs=self.net,
                        num_outputs = self.outlayer,
                        activation_fn = tf.nn.sigmoid)
                    self.net = fully_connected(
                        inputs=self.net,
                        num_outputs = self.outlayer,
                        activation_fn = tf.nn.sigmoid)
                    if len(self.input_shape)==2:
                        self.outputs_mse = self.net_mse
                    else:
                        self.outputs_mse = tf.reshape(
                            self.net_mse,
                            [-1,
                            self.input_shape[1],
                            self.input_shape[2],
                            self.input_shape[3]])
                self.netprinter.append(
                    ["Output layer", self.net.get_shape().as_list()])
                if len(self.input_shape)==2:
                    self.outputs_de = self.net
                else:
                    self.outputs_de = tf.reshape(
                        self.net,
                        [-1,
                        self.input_shape[1],
                        self.input_shape[2],
                        self.input_shape[3]])


    def cvae_build(self):
        """Build the general cvae network"""
        self.netprinter = []
        self.netprinter.append(["Input layer",
                                self.net.get_shape().as_list()])
        # The encoder part
        with tf.name_scope("cave"):
            with tf.name_scope("fc_en"):
                for i, layer in enumerate(self.layers):
                    self.net = fully_connected(
                        inputs=self.net,
                        num_outputs = layer,
                        activation_fn = self.actfun[i])
                    self.netprinter.append(
                        ["FC layer " + str(i), self.net.get_shape().as_list()])
                    # batch normalization
                    if self.batchflag[i]:
                        self.net = batch_norm(
                            inputs=self.net,
                            center=True,
                            scale=True,
                            is_training=self.is_training)
                        self.netprinter.append(
                            ["BN layer " + str(i), self.net.get_shape().as_list()])
                    else:
                        # dropout
                        self.net = tf.nn.dropout(
                            x=self.net,
                            keep_prob=self.keep_prob,
                            name="drop_"+str(i))
                        self.netprinter.append(
                            ["Dropout layer " + str(i), self.net.get_shape().as_list()])
            # outputs
            with tf.name_scope("en_output"):
                self.mu = fully_connected(
                    inputs=self.net,
                    num_outputs=self.z_length,
                    activation_fn=None)
                self.sigma=fully_connected(
                    inputs=self.net,
                    num_outputs=self.z_length,
                    activation_fn=None)
                # softplus
                self.sigma=1e-6 + tf.nn.softplus(self.sigma)
                self.netprinter.append(["En_mu", self.mu.get_shape().as_list()])
                self.netprinter.append(["En_sigma", self.sigma.get_shape().as_list()])

            # Reparameterization to obtain z
            with tf.name_scope("reparameterization"):
                self.epsilon = tf.random_normal(tf.shape(self.mu))
                self.z = self.mu + tf.multiply(self.epsilon, self.sigma)
                self.netprinter.append(["z", self.z.get_shape().as_list()])

            # The decoder subnet
            self.net = tf.concat(
                [self.z, self.conditions], axis=1)
            # the decoder
            with tf.name_scope("fc_de"):
                for i in range(len(self.layers)-1,-1,-1):
                    self.net = fully_connected(
                        inputs=self.net,
                        num_outputs = self.layers[i],
                        activation_fn = self.actfun[i])
                    self.netprinter.append(
                        ["FC layer " + str(i), self.net.get_shape().as_list()])
                    # batch normalization
                    if self.batchflag[i]:
                        self.net = batch_norm(
                            inputs=self.net,
                            center=True,
                            scale=True,
                            is_training=self.is_training)
                        self.netprinter.append(
                            ["BN layer " + str(i), self.net.get_shape().as_list()])
                    else:
                        # dropout
                        self.net = tf.nn.dropout(
                            x=self.net,
                            keep_prob=self.keep_prob,
                            name="drop_"+str(i)
                        )
                        self.netprinter.append(
                            ["Dropout layer " + str(i), self.net.get_shape().as_list()])
                # Output
                if self.mseflag:
                    self.net = fully_connected(
                        inputs=self.net,
                        num_outputs = self.outlayer,
                        activation_fn = tf.nn.sigmoid)
                else:
                    self.net_mse = fully_connected(
                        inputs=self.net,
                        num_outputs = self.outlayer,
                        activation_fn = tf.nn.sigmoid)
                    self.net = fully_connected(
                        inputs=self.net,
                        num_outputs = self.outlayer,
                        activation_fn = tf.nn.sigmoid)
                    if len(self.input_shape)==2:
                        self.outputs_mse = self.net_mse
                    else:
                        self.outputs_mse = tf.reshape(
                            self.net_mse,
                            [-1,
                            self.input_shape[1],
                            self.input_shape[2],
                            self.input_shape[3]])
                self.netprinter.append(
                    ["Output layer", self.net.get_shape().as_list()])
                if len(self.input_shape)==2:
                    self.outputs_de = self.net
                else:
                    self.outputs_de = tf.reshape(
                        self.net,
                        [-1,
                        self.input_shape[1],
                        self.input_shape[2],
                        self.input_shape[3]])


    def cvae_print(self):
        """Print the network"""
        print("Layer ID    Layer type    Layer shape")
        for i, l in enumerate(self.netprinter):
            print(i, l[0], l[1])


    def get_loss(self):
        """Get loss function"""
        with tf.name_scope("loss"):
            if not self.mseflag:
                self.loss_ce = - tf.reduce_sum(
                    self.outputs * tf.log(1e-8+self.outputs_de) +
                    (1.0 - self.outputs) * tf.log(1e-8 + 1.0 - self.outputs_de), 1)
                self.loss_mse = tf.reduce_sum(
                    tf.squared_difference(self.outputs_mse, self.outputs), 1)
                self.loss_recon = self.loss_ce
            else:
                self.loss_ce = - tf.reduce_sum(
                    self.outputs * tf.log(1e-8+self.outputs_de) +
                    (1.0 - self.outputs) * tf.log(1e-8 + 1.0 - self.outputs_de), 1)
                self.loss_mse = tf.reduce_sum(
                    tf.squared_difference(self.outputs_de, self.outputs), 1)
                self.loss_recon = self.loss_mse
            # latend loss
            self.loss_latent = 0.5 * tf.reduce_sum(
                tf.square(self.mu) + tf.square(self.sigma) -
                tf.log(1e-8 + tf.square(self.sigma)) - 1, 1)
            # combine
            self.loss = tf.reduce_mean(self.loss_recon + self.loss_latent)


    def get_opt(self):
        """Training optimizer"""
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope("train_ops"):
                self.train_ops = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)



    def get_learning_rate(self):
        """Get the exponentially decreased learning rate."""
        self.init_lr = tf.placeholder(tf.float32, name="init_lr")
        self.global_step = tf.placeholder(tf.float32, name="global_step")
        self.decay_step = tf.placeholder(tf.float32, name="decay_step")
        self.decay_rate = tf.placeholder(tf.float32, name="decay_rate")
        with tf.name_scope('learning_rate'):
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=self.init_lr ,
                global_step=self.global_step,
                decay_steps=self.decay_step,
                decay_rate=self.decay_rate,
                staircase=False,
                name=None)



    def loss_save(self,savepath):
        """Save the loss and accuracy staffs"""
        import pickle
        with open(savepath, 'wb') as fp:
            pickle.dump(self.train_dict, fp)


    def cvae_test(self, data, labels):
        """Test the network"""
        test_loss = self.sess.run(
            self.loss_recon,
            feed_dict={
                self.inputs: data,
                self.conditions: labels,
                self.outputs: data,
                self.is_training: False,
                self.keep_prob: 1.0})

        return test_loss.mean()


    def cvae_train(self, data, train_configs, labels):
        """Train the cvae network"""

        self.get_learning_rate()
        self.get_loss()
        self.get_opt()

        # Init sess
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        # get validation
        data_trn,data_val = utils.gen_validation(
            data, valrate=train_configs.valrate, label=labels)

        #numbatch_trn = len(data_trn["data"]) // train_configs.batchsize
        #numbatch_val = len(data_val["data"]) // train_configs.batchsize
        numbatch_trn = 5
        numbatch_val = 5

        x_epoch = np.arange(0,train_configs.epochs,1)
        y_loss_recon_trn = np.zeros(x_epoch.shape)
        y_loss_recon_val = np.zeros(x_epoch.shape)
        y_loss_latent_trn = np.zeros(x_epoch.shape)
        y_loss_latent_val = np.zeros(x_epoch.shape)
        y_loss_val = np.zeros(x_epoch.shape)
        y_loss_trn = np.zeros(x_epoch.shape)

        # Init all variables
        timestamp = utils.get_timestamp()
        print("[%s]: Epochs    Trn_loss_recon    Val_loss_recon    Trn_loss_latent    Val_loss_latent"
              % (timestamp))
        for i in range(train_configs.epochs):
            lr_dict = {self.init_lr: train_configs.lr_init,
                       self.global_step: i,
                       self.decay_step: train_configs.batchsize,
                       self.decay_rate: train_configs.decay_rate}

            loss_trn_recon_all = 0.0
            loss_val_recon_all = 0.0
            loss_trn_latent_all = 0.0
            loss_val_latent_all = 0.0
            loss_val_all = 0.0
            loss_trn_all = 0.0

            indices_trn = utils.gen_BatchIterator_label(
                data_trn['data'],
                data_trn['label'],
                batch_size=train_configs.batchsize,
                shuffle=True)

            for i_trn in range(numbatch_trn):
                idx_trn = indices_trn[i_trn*train_configs.batchsize:
                                      (i_trn+1)*train_configs.batchsize]
                train_dict = {
                    self.inputs: data_trn['data'][idx_trn],
                    self.conditions: data_trn['label'][idx_trn],
                    self.outputs: data_trn['data'][idx_trn],
                    self.is_training: True,
                    self.keep_prob: train_configs.keep_prob}
                train_dict.update(lr_dict)
                # train
                _, loss_trn_recon, loss_trn_latent, loss_trn = self.sess.run(
                    [self.train_ops, self.loss_recon,
                     self.loss_latent, self.loss],
                    feed_dict=train_dict)
                loss_trn_recon_all += loss_trn_recon.mean()
                loss_trn_latent_all += loss_trn_latent.mean()
                loss_trn_all += loss_trn.mean()
            # print(loss_trn_recon_all, loss_trn_latent_all, loss_trn_all)

            y_loss_recon_trn[i] = loss_trn_recon_all / numbatch_trn
            y_loss_latent_trn[i] = loss_trn_latent_all / numbatch_trn
            y_loss_trn[i] = loss_trn_all / numbatch_trn
            # validation
            indices_val = utils.gen_BatchIterator_label(
                data_val['data'],
                data_val['label'],
                batch_size=train_configs.batchsize,
                shuffle=True)

            for i_val in range(numbatch_val):
                idx_val = indices_val[i_val*train_configs.batchsize:
                                      (i_val+1)*train_configs.batchsize]
                val_dict = {
                    self.inputs: data_val['data'][idx_val],
                    self.conditions: data_val['label'][idx_val],
                    self.outputs: data_val['data'][idx_val],
                    self.is_training: False,
                    self.keep_prob: 1.0}
                loss_val_recon, loss_val_latent, loss_val = self.sess.run(
                    [self.loss_recon, self.loss_latent, self.loss],
                    feed_dict=val_dict)
                loss_val_recon_all += loss_val_recon.mean()
                loss_val_latent_all += loss_val_latent.mean()
                loss_val_all += loss_val.mean()

            y_loss_recon_val[i] = loss_val_recon_all / numbatch_val
            y_loss_latent_val[i] = loss_val_latent_all / numbatch_val
            y_loss_val[i] = loss_val_all / numbatch_val
            # print results
            if i % train_configs.print_step == 0:
                timestamp = utils.get_timestamp()
                print('[%s]: %6d    %14.5f    %14.5f    %15.5f    %15.5f' % (
                    timestamp, i,
                    y_loss_recon_trn[i], y_loss_recon_val[i],
                    y_loss_latent_trn[i], y_loss_latent_val[i]
                    ))

        self.train_dict = {
            "epochs": x_epoch,
            "trn_loss_recon": y_loss_recon_trn,
            "trn_loss_latent": y_loss_latent_trn,
            "trn_loss": y_loss_trn,
            "val_loss_recon": y_loss_recon_val,
            "val_loss_latent": y_loss_latent_val,
            "val_loss": y_loss_val}


    def vae_train(self, data, train_configs, labels=None):
        """Train the vae network"""

        self.get_learning_rate()
        self.get_loss()
        self.get_opt()

        # Init sess
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        # get validation
        data_trn,data_val = utils.gen_validation(
            data, valrate=train_configs.valrate, label=labels)

        #numbatch_trn = len(data_trn["data"]) // train_configs.batchsize
        #numbatch_val = len(data_val["data"]) // train_configs.batchsize
        numbatch_trn = 5
        numbatch_val = 5

        x_epoch = np.arange(0,train_configs.epochs,1)
        y_loss_recon_trn = np.zeros(x_epoch.shape)
        y_loss_recon_val = np.zeros(x_epoch.shape)
        y_loss_latent_trn = np.zeros(x_epoch.shape)
        y_loss_latent_val = np.zeros(x_epoch.shape)
        y_loss_val = np.zeros(x_epoch.shape)
        y_loss_trn = np.zeros(x_epoch.shape)

        # Init all variables
        timestamp = utils.get_timestamp()
        print("[%s]: Epochs    Trn_loss_recon    Val_loss_recon    Trn_loss_latent    Val_loss_latent"
              % (timestamp))
        for i in range(train_configs.epochs):
            lr_dict = {self.init_lr: train_configs.lr_init,
                       self.global_step: i,
                       self.decay_step: train_configs.batchsize,
                       self.decay_rate: train_configs.decay_rate}

            loss_trn_recon_all = 0.0
            loss_val_recon_all = 0.0
            loss_trn_latent_all = 0.0
            loss_val_latent_all = 0.0
            loss_val_all = 0.0
            loss_trn_all = 0.0

            indices_trn = utils.gen_BatchIterator_label(
                data_trn['data'],
                data_trn['label'],
                batch_size=train_configs.batchsize,
                shuffle=True)

            for i_trn in range(numbatch_trn):
                idx_trn = indices_trn[i_trn*train_configs.batchsize:
                                      (i_trn+1)*train_configs.batchsize]
                train_dict = {
                    self.inputs: data_trn['data'][idx_trn],
                    self.outputs: data_trn['data'][idx_trn],
                    self.is_training: True,
                    self.keep_prob: train_configs.keep_prob}
                train_dict.update(lr_dict)
                # train
                _, loss_trn_recon, loss_trn_latent, loss_trn = self.sess.run(
                    [self.train_ops, self.loss_recon,
                     self.loss_latent, self.loss],
                    feed_dict=train_dict)
                loss_trn_recon_all += loss_trn_recon.mean()
                loss_trn_latent_all += loss_trn_latent.mean()
                loss_trn_all += loss_trn.mean()

            y_loss_recon_trn[i] = loss_trn_recon_all / numbatch_trn
            y_loss_latent_trn[i] = loss_trn_latent_all / numbatch_trn
            y_loss_trn[i] = loss_trn_all / numbatch_trn
            # validation
            indices_val = utils.gen_BatchIterator_label(
                data_val['data'],
                data_val['label'],
                batch_size=train_configs.batchsize,
                shuffle=True)

            for i_val in range(numbatch_val):
                idx_val = indices_val[i_val*train_configs.batchsize:
                                      (i_val+1)*train_configs.batchsize]
                val_dict = {
                    self.inputs: data_val['data'][idx_val],
                    self.outputs: data_val['data'][idx_val],
                    self.is_training: False,
                    self.keep_prob: 1.0}
                loss_val_recon, loss_val_latent, loss_val = self.sess.run(
                    [self.loss_recon, self.loss_latent, self.loss],
                    feed_dict=val_dict)
                loss_val_recon_all += loss_val_recon.mean()
                loss_val_latent_all += loss_val_latent.mean()
                loss_val_all += loss_val.mean()

            y_loss_recon_val[i] = loss_val_recon_all / numbatch_val
            y_loss_latent_val[i] = loss_val_latent_all / numbatch_val
            y_loss_val[i] = loss_val_all / numbatch_val
            # print results
            if i % train_configs.print_step == 0:
                timestamp = utils.get_timestamp()
                print('[%s]: %6d    %14.5f    %14.5f    %15.5f    %15.5f' %
                      (timestamp, i, y_loss_recon_trn[i], y_loss_recon_val[i],
                       y_loss_latent_trn[i], y_loss_latent_val[i]))

        self.train_dict = {
            "epochs": x_epoch,
            "trn_loss_recon": y_loss_recon_trn,
            "trn_loss_latent": y_loss_latent_trn,
            "trn_loss": y_loss_trn,
            "val_loss_recon": y_loss_recon_val,
            "val_loss_latent": y_loss_latent_val,
            "val_loss": y_loss_val}


    def cvae_train_mnist(self, mnist, train_configs):
        """Train the network on MNIST data"""

        self.get_learning_rate()
        self.get_loss()
        self.get_opt()

        # Init sess
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        #numbatch_trn = mnist.train.images.shape[0] // train_configs.batchsize
        #numbatch_val = mnist.validation.images.shape[0] // train_configs.batchsize
        numbatch_trn = 10
        numbatch_val = 10


        x_epoch = np.arange(0,train_configs.epochs,1)
        y_loss_recon_trn = np.zeros(x_epoch.shape)
        y_loss_recon_val = np.zeros(x_epoch.shape)
        y_loss_latent_trn = np.zeros(x_epoch.shape)
        y_loss_latent_val = np.zeros(x_epoch.shape)
        y_loss_val = np.zeros(x_epoch.shape)
        y_loss_trn = np.zeros(x_epoch.shape)

        # Init all variables
        timestamp = utils.get_timestamp()
        print("[%s]: Epochs    Trn_loss_recon    Val_loss_recon    Trn_loss_latent    Val_loss_latent"
              % (timestamp))
        for i in range(train_configs.epochs):
            lr_dict = {self.init_lr: train_configs.lr_init,
                       self.global_step: i,
                       self.decay_step: train_configs.batchsize,
                       self.decay_rate: train_configs.decay_rate}

            loss_trn_recon_all = 0.0
            loss_val_recon_all = 0.0
            loss_trn_latent_all = 0.0
            loss_val_latent_all = 0.0
            loss_val_all = 0.0
            loss_trn_all = 0.0

            for i_trn in range(numbatch_trn):
                data_trn, label_trn = mnist.train.next_batch(
                      batch_size=train_configs.batchsize)
                train_dict = {
                    self.inputs: data_trn,
                    self.conditions: label_trn,
                    self.outputs: data_trn,
                    self.is_training: True,
                    self.keep_prob: train_configs.keep_prob}
                train_dict.update(lr_dict)
                # train
                _, loss_trn_recon, loss_trn_latent, loss_trn = self.sess.run(
                    [self.train_ops, self.loss_recon,
                     self.loss_latent, self.loss],
                    feed_dict=train_dict)
                loss_trn_recon_all += loss_trn_recon.mean()
                loss_trn_latent_all += loss_trn_latent.mean()
                loss_trn_all += loss_trn.mean()

            y_loss_recon_trn[i] = loss_trn_recon_all / numbatch_trn
            y_loss_latent_trn[i] = loss_trn_latent_all / numbatch_trn
            y_loss_trn[i] = loss_trn_all / numbatch_trn

            # validation
            for i_trn in range(numbatch_val):
                data_val, label_val = mnist.validation.next_batch(
                      batch_size=train_configs.batchsize)
                val_dict = {
                    self.inputs: data_val,
                    self.conditions: label_val,
                    self.outputs: data_val,
                    self.is_training: False,
                    self.keep_prob: 1.0}
                loss_val_recon, loss_val_latent, loss_val = self.sess.run(
                    [self.loss_recon, self.loss_latent, self.loss],
                    feed_dict=val_dict)
                loss_val_recon_all += loss_val_recon.mean()
                loss_val_latent_all += loss_val_latent.mean()
                loss_val_all += loss_val.mean()

            y_loss_recon_val[i] = loss_val_recon_all / numbatch_val
            y_loss_latent_val[i] = loss_val_latent_all / numbatch_val
            y_loss_val[i] = loss_val_all / numbatch_val
            # print results
            if i % train_configs.print_step == 0:
                timestamp = utils.get_timestamp()
                print('[%s]: %6d    %14.5f    %14.5f    %15.5f    %15.5f' % (
                    timestamp, i,
                    y_loss_recon_trn[i], y_loss_recon_val[i],
                    y_loss_latent_trn[i], y_loss_latent_val[i]
                    ))

        self.train_dict = {
            "epochs": x_epoch,
            "trn_loss_recon": y_loss_recon_trn,
            "trn_loss_latent": y_loss_latent_trn,
            "trn_loss": y_loss_trn,
            "val_loss_recon": y_loss_recon_val,
            "val_loss_latent": y_loss_latent_val,
            "val_loss": y_loss_val}
