#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf
from cfgs.config import cfg
import pdb

from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.models import (
    Conv2D, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected,
    LinearWrap)


def resnet_shortcut(l, n_out, stride, rate, nl=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format == 'NCHW' else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, stride=stride, dilation_rate=rate, nl=nl)
    else:
        return l


def apply_preactivation(l, preact):
    if preact == 'bnrelu':
        shortcut = l    # preserve identity mapping
        l = BNReLU('preact', l)
    else:
        shortcut = l
    return l, shortcut


def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name: BatchNorm('bn', x, gamma_init=tf.zeros_initializer())
    else:
        return lambda x, name: BatchNorm('bn', x)


def preresnet_basicblock(l, ch_out, stride, rate, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 3, stride=stride, dilation_rate=rate, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, dilation_rate=rate)
    return l + resnet_shortcut(shortcut, ch_out, stride, rate)


def preresnet_bottleneck(l, ch_out, stride, rate, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, dilation_rate=rate, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=stride, dilation_rate=rate, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, dilation_rate=rate)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, rate)


def preresnet_group(l, name, block_func, features, count, stride, rate):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu', rate)
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l


def resnet_basicblock(l, ch_out, stride, rate):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 3, stride=stride, dilation_rate=rate, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, dilation_rate=rate, nl=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out, stride, rate, nl=get_bn(zero_init=False))


def resnet_bottleneck(l, ch_out, stride, rate, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, stride=stride if stride_first else 1, dilation_rate=rate, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=1 if stride_first else stride, dilation_rate=rate, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, dilation_rate=rate, nl=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, rate, nl=get_bn(zero_init=False))


def se_resnet_bottleneck(l, ch_out, stride, rate):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, dilation_rate=rate, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=stride, dilation_rate=rate, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, dilation_rate=rate, nl=get_bn(zero_init=True))

    squeeze = GlobalAvgPooling('gap', l)
    squeeze = FullyConnected('fc1', squeeze, ch_out // 4, nl=tf.nn.relu)
    squeeze = FullyConnected('fc2', squeeze, ch_out * 4, nl=tf.nn.sigmoid)
    data_format = get_arg_scope()['Conv2D']['data_format']
    ch_ax = 1 if data_format == 'NCHW' else 3
    shape = [-1, 1, 1, 1]
    shape[ch_ax] = ch_out * 4
    l = l * tf.reshape(squeeze, shape)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, rate, nl=get_bn(zero_init=False))


def resnet_group(l, name, block_func, features, count, stride, rate):
    with tf.variable_scope(name):
        if rate != 1:
            multi_grid_rate = [m * rate for m in cfg.multi_grid]
        else:
            multi_grid_rate = [1] * count
        # import pdb
        # pdb.set_trace()
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1, multi_grid_rate[i])
                # end of each block need an activation
                l = tf.nn.relu(l)
    return l


def resnet_backbone(image, num_blocks, group_func, block_func):
    with argscope(Conv2D, nl=tf.identity, use_bias=False,
                  W_init=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        logits = (LinearWrap(image)
                  .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                  .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                  .apply(group_func, 'group0', block_func, 64, num_blocks[0], 1, 1)
                  .apply(group_func, 'group1', block_func, 128, num_blocks[1], 2, 1)
                  .apply(group_func, 'group2', block_func, 256, num_blocks[2], 2, 1)
                  .apply(group_func, 'group3', block_func, 512, num_blocks[3], 1, 2)())
    return logits 
