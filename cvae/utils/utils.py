# Copyright (C) 2018 Zhixian MA <zx@mazhixian.me>
# MIT liscence

"""Utilities for the dnn"""

import os
import pickle
import numpy as np

import tensorflow as tf


def get_timestamp():
    """Get time at present"""
    import time
    timestamp = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
    return timestamp


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        with tf.name_scope('mean'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def gen_validation(data, valrate = 0.2, label=None):
    """Separate the dataset into training and validation subsets.
    inputs
    ======
    data: np.ndarray
        The input data, 4D matrix
    label: np.ndarray or list, opt
        The labels w.r.t. input data, optional
    outputs
    =======
    data_train: {"data": , "label": }
    data_val: {"data":, "label":}
    """
    if label is None:
        label_train = None
        label_val = None
        idx = np.random.permutation(len(data))
        num_val = int(len(data)*valrate)
        data_val = {"data": data[idx[0:num_val]],
                    "label": label_val}
        # train
        data_train = {"data": data[idx[num_val:]],
                      "label": label_train}
    else:
        # Training and validation
        idx = np.random.permutation(len(data))
        num_val = int(len(data)* valrate)
        data_val = {"data": data[idx[0:num_val]],
                    "label": label[idx[0:num_val]]}
        # train
        data_train = {"data": data[idx[num_val:]],
                      "label": label[idx[num_val:]]}

    return data_train,data_val


def gen_BatchIterator(data, batch_size=100, shuffle=True):
    """
    Return the next 'batch_size' examples
    from the X_in dataset
    Reference
    =========
    [1] tensorflow.examples.tutorial.mnist.input_data.next_batch
    Input
    =====
    data: 4d np.ndarray
        The samples to be batched
    batch_size: int
        Size of a single batch.
    shuffle: bool
        Whether shuffle the indices.
    Output
    ======
    Yield a batch generator
    """
    if shuffle:
        indices = np.arange(len(data))
        np.random.shuffle(indices)
    for start_idx in range(0, len(data) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx: start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield data[excerpt]


def gen_BatchIterator_label(data, label, batch_size=100, shuffle=True):
        """
        Return the next 'batch_size' examples
        from the X_in dataset
        Reference
        =========
        [1] tensorflow.examples.tutorial.mnist.input_data.next_batch
        Input
        =====
        data: 4d np.ndarray
            The samples to be batched
        label: np.ndarray
            The labels to be batched w.r.t. data
        batch_size: int
            Size of a single batch.
        shuffle: bool
            Whether shuffle the indices.
        Output
        ======
        Yield a batch generator
        """
        if shuffle:
            indices = np.arange(len(data))
            np.random.shuffle(indices)
        else:
            indices = np.arange(len(data))
        return indices


def vec2onehot(label,numclass):
    label_onehot = np.zeros((len(label),numclass))
    for i,l in enumerate(label):
        label_onehot[i, int(l)] = 1

    return label_onehot


def save_net(sess, namepath, netpath, savedict):
    """Save the net"""
    import pickle
    import sys
    sys.setrecursionlimit(1000000)

    with open(namepath, 'wb') as fp:
        pickle.dump(savedict, fp)

    # save the net
    saver = tf.train.Saver()
    saver.save(sess, netpath)


def cvae_test(net, data, labels):
    """Test the network: mse"""
    test_loss = net.sess.run(
        net.loss_mse,
        feed_dict={
            net.inputs: data,
            net.conditions: labels,
            net.outputs: data,
            net.is_training: False,
            net.keep_prob: 1.0})

    return test_loss


def loss_eval(net, data, labels):
    """Evaluate the network performance on test samples"""
    loss = np.zeros([labels.shape[0]])
    for i in range(len(loss)):
        img_est = net.sess.run(
            net.outputs_de,
            feed_dict={
                net.inputs: data[i].reshape([1, data[i].shape[0]]),
                net.conditions: labels[i].reshape([1, labels[i].shape[0]]),
                net.is_training: False,
                net.keep_prob: 1.0})
        img_est = (img_est - img_est.min()) / (img_est.max() - img_est.min())
        loss[i] = np.mean((data[i].reshape([1, data[i].shape[0]]) - img_est)**2)
    # evaluation
    loss_mean = np.mean(loss)
    loss_std = np.std(loss)

    return loss,loss_mean,loss_std


def loss_eval_vae(net, data, labels=None):
    """Evaluate the network performance on test samples"""
    loss = np.zeros([labels.shape[0]])
    for i in range(len(loss)):
        img_est = net.sess.run(
            net.outputs_de,
            feed_dict={
                net.inputs: data[i].reshape([1, data[i].shape[0]]),
                net.is_training: False,
                net.keep_prob: 1.0})
        img_est = (img_est - img_est.min()) / (img_est.max() - img_est.min())
        loss[i] = np.mean((data[i].reshape([1, data[i].shape[0]]) - img_est)**2)
    # evaluation
    loss_mean = np.mean(loss)
    loss_std = np.std(loss)

    return loss,loss_mean,loss_std


def loss_eval_old(net, data, labels):
    """Evaluate the network performance on test samples"""
    loss = np.zeros([labels.shape[0]])
    for i in range(len(loss)):
        loss[i] = cvae_test(
            net,
            data[i].reshape([1, data[i].shape[0]]),
            labels[i].reshape([1, labels[i].shape[0]]))
    # evaluation
    loss_mean = np.mean(loss)
    loss_std = np.std(loss)

    return loss,loss_mean,loss_std    


def get_feature(net, data, labels=None):
    """Get the extracted features"""
    features = net.sess.run(
        net.code,
        feed_dict = {net.inputs: data,
                     net.is_training: False,
                     net.keep_prob: 1.0})

    return features


def loss_save(net,savepath):
    """Save the loss and accuracy staffs"""
    import pickle
    with open(savepath, 'wb') as fp:
        pickle.dump(net.train_dict, fp)


def load_net(namepath):
    """
    Load the cae network
    reference
    =========
    [1] https://www.cnblogs.com/azheng333/archive/2017/06/09/6972619.html
    input
    =====
    namepath: str
        Path to save the trained network
    output
    ======
    sess: tf.Session()
        The restored session
    names: dict
        The dict saved variables names
    """
    try:
        fp = open(namepath,'rb')
    except:
        return None

    names = pickle.load(fp)

    # load the net
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, names['netpath'])

    return sess, names

def load_raw_image(names, folder, rs=80):
    def gen_norm(img):
        return (img-img.min())/(img.max() - img.min())
    from astropy.io import fits
    img_all = np.zeros((len(names),rs,rs))
    for i, name in enumerate(names):
        filepath = os.path.join(folder, name+".fits")
        with fits.open(filepath) as h:
            img = h[0].data
            img_shape = img.shape
            r_c = img_shape[0] // 2
            c_c = img_shape[1] // 2
            r_h = rs // 2
            c_h = rs // 2
            img = img[r_c-r_h:r_c+r_h, c_c-c_h:c_c+c_h]
            img_all[i, :, :] = gen_norm(img)
                
    return img_all