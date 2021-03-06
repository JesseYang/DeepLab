import pdb
import cv2
import sys
import argparse
import numpy as np
import os
import shutil
import multiprocessing
import json
from abc import abstractmethod

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import init_ops
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.slim.python.slim.nets import resnet_utils

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.common import get_tf_version_number
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.models import (
    Conv2D, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected,
    LinearWrap)
from resnet_model import (
    preresnet_group, preresnet_basicblock, preresnet_bottleneck,
    resnet_group, resnet_basicblock, resnet_bottleneck, se_resnet_bottleneck,
    resnet_backbone)

from cfgs.config import cfg
from reader import Data

class DeeplabModel(ModelDesc):
    def __init__(self, data_format='NHWC', depth=101, mode='resnet'):
        super(DeeplabModel, self).__init__()
        self.data_format = data_format
        self.mode = mode
        basicblock = preresnet_basicblock if mode == 'preact' else resnet_basicblock
        bottleneck = {
            'resnet': resnet_bottleneck,
            'preact': preresnet_bottleneck,
            'se': se_resnet_bottleneck}[mode]
        self.num_blocks, self.block_func = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[depth]

    @staticmethod
    def image_preprocess(image, bgr=True):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            image = image * (1.0 / 255)

            mean = [0.485, 0.456, 0.407]    # rgb
            std = [0.229, 0.224, 0.225]
            if bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32)
            image_std = tf.constant(std, dtype=tf.float32)
            image = (image - image_mean) / image_std
            return image

    @staticmethod
    def image_preprocess_slim(image, bgr=True):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)

            mean = [123.68, 116.78, 103.94] # rgb
            if bgr:
                mean = mean[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32)
            image = image - image_mean
            return image

    def _get_inputs(self):
        return [InputDesc(tf.uint8, [None, None, None, 3], 'input'),
                InputDesc(tf.uint8, [None, None, None, 1], 'label')]
   
    def _get_logits(self, image):
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            return resnet_backbone(image, self.num_blocks, preresnet_group if self.mode == 'preact' else resnet_group, self.block_func)       

    def _get_logits_by_slim_model(self, inputs):
        ctx = get_current_tower_context()
        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=0.9997)):
            logits, end_points = resnet_v2.resnet_v2_101(inputs,
                                                         num_classes=None,
                                                         is_training=ctx.is_training,
                                                         global_pool=False,
                                                         output_stride=16)
        net = end_points['resnet_v2_101/block4']
        return net
 
    def _build_graph(self, inputs):
        image, label = inputs
        self.batch_size = tf.shape(image)[0]
        org_label = label
        # when show image summary, first convert to RGB format
        image = tf.reverse(image, axis=[-1])
        label_shown = tf.where(tf.equal(label, cfg.ignore_label), tf.zeros_like(label), label)
        label_shown = tf.cast(label_shown * 10, tf.uint8) 
        tf.summary.image('input-image', image, max_outputs=3)
        tf.summary.image('input-label', label_shown, max_outputs=3)

        # image = DeeplabModel.image_preprocess(image, bgr=True)
        image = DeeplabModel.image_preprocess_slim(image, bgr=False)

        if self.data_format == "NCHW":
            image = tf.transpose(image, [0, 3, 1, 2])

        # the backbone part
        # logits = self._get_logits(image)
        logits = self._get_logits_by_slim_model(image)
        logits_size = tf.shape(logits)[1:3]


        ctx = get_current_tower_context()
        with tf.variable_scope("aspp"):
            inputs = logits
            depth = 256
            output_stride = 16
            atrous_rates = [6, 12, 18]
        
            with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=0.9997)):
                with arg_scope([layers.batch_norm], is_training=ctx.is_training):
                    inputs_size = tf.shape(inputs)[1:3]
                    # (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16.
                    # the rates are doubled when output stride = 8.
                    conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
                    conv_3x3_1 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
                    conv_3x3_2 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
                    conv_3x3_3 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[2], scope='conv_3x3_3')
        
                    # (b) the image-level features
                    with tf.variable_scope("image_level_features"):
                        # global average pooling
                        image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
                        # 1×1 convolution with 256 filters( and batch normalization)
                        image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
                        # bilinearly upsample features
                        image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')
        
                    net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
                    net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')
    
    
        with tf.variable_scope("upsampling_logits"):
            net = layers_lib.conv2d(net, cfg.num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
            logits = tf.image.resize_bilinear(net, tf.shape(label)[1:3], name='upsample')


        '''
        # Compute the ASPP.
        with argscope(Conv2D, filters=256, kernel_size=3, activation=BNReLU):
            ASPP_1 = Conv2D('aspp_conv1', logits, kernel_size=1)
            ASPP_2 = Conv2D('aspp_conv2', logits, dilation_rate=cfg.atrous_rates[0])
            ASPP_3 = Conv2D('aspp_conv3', logits, dilation_rate=cfg.atrous_rates[1])
            ASPP_4 = Conv2D('aspp_conv4', logits, dilation_rate=cfg.atrous_rates[2])
            # ImagePooling = GlobalAvgPooling('image_pooling', logits)
            ImagePooling = tf.reduce_mean(logits, [1, 2], name='global_average_pooling', keepdims=True)
            image_level_features = Conv2D('image_level_conv', ImagePooling, kernel_size=1)
        image_level_features = tf.image.resize_bilinear(image_level_features, logits_size, name='upsample')
        logits = tf.concat([ASPP_1, ASPP_2, ASPP_3, ASPP_4, image_level_features], -1, name='concat')
        logits = Conv2D('conv_after_concat', logits, 256, 1, activation=BNReLU)
        # logits = BatchNorm(logits)
        logits = Conv2D('final_conv', logits, cfg.num_classes, 1)

        # Compute softmax cross entropy loss for logits
        logits = tf.image.resize_bilinear(logits, tf.shape(label)[1:3], align_corners=True)
        '''

        label_flatten = tf.reshape(label, shape=[-1])
        mask = tf.to_float(tf.not_equal(label_flatten, cfg.ignore_label)) * 1.0
        one_hot_label = tf.one_hot(label_flatten, cfg.num_classes, on_value=1.0, off_value=0.0)

        loss = tf.losses.softmax_cross_entropy(one_hot_label, tf.reshape(logits, shape=[-1, cfg.num_classes]), weights=mask)
        if cfg.weight_decay > 0:
            # wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
            train_var_list = [v for v in tf.trainable_variables()]
            wd_cost = cfg.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in train_var_list])
        else:
            wd_cost = tf.constant(0.0)

        self.cost = tf.add_n([loss, wd_cost], name='cost')
        
        pred = tf.argmax(tf.nn.softmax(logits), 3, name='predicts')
        pred_shown = pred * 10
        pred_shown = tf.cast(tf.expand_dims(pred_shown, -1), tf.uint8)
        pred_shown = tf.where(tf.equal(label, cfg.ignore_label), tf.zeros_like(label), pred_shown)
        tf.summary.image('input-preds', tf.cast(pred_shown, tf.uint8), max_outputs=3)

        # compute the mean_iou
        pred_flatten = tf.reshape(pred, shape=[-1])
        label_flatten = tf.where(tf.equal(label_flatten, cfg.ignore_label), tf.zeros_like(label_flatten), label_flatten)
        label_flatten = tf.cast(label_flatten, tf.int64)
        miou, miou_update_op = tf.metrics.mean_iou(label_flatten, pred_flatten, cfg.num_classes, weights=mask, name="metric_scope")
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="InferenceTower/metric_scope")
        # running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
        miou_reset_op = tf.variables_initializer(var_list=running_vars, name='miou_reset_op')
        miou = tf.identity(miou, name='miou')
        miou_update_op = tf.identity(miou_update_op, name='miou_update_op')
    
        add_moving_summary(loss, wd_cost, self.cost)

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', cfg.base_lr, summary=True)
        optimizer = tf.train.MomentumOptimizer(lr, cfg.momentum)
        
        return optimizer

class CalMIOU(Inferencer):

    def __init__(self):
        self.cal_miou_name = 'InferenceTower/miou:0'
        self.reset_miou_name = 'InferenceTower/miou_reset_op'
        self.update_miou_name = 'InferenceTower/miou_update_op'
        self.names = ["miou_update_op"]

    def _get_from_graph(self):
        sess = tf.get_default_session()
        graph = sess.graph
        reset_miou = graph.get_operation_by_name(self.reset_miou_name)
        update_miou = graph.get_operation_by_name(self.update_miou_name)
        cal_miou = graph.get_tensor_by_name(self.cal_miou_name)
        return sess, reset_miou, update_miou, cal_miou

    def _before_inference(self):
        sess, reset_miou, update_miou, cal_miou = self._get_from_graph()
        sess.run(reset_miou)

    def _get_fetches(self):
        return self.names

    def _after_inference(self):
        # the following code should find the target names:
        #   ['tower0/miou', 'tower0/miou_update_op', ['tower1/miou', 'tower1/miou_update_op', ...
        #    'InferenceTower/miou', 'InferenceTower/miou_update_op']
        '''
        op_list = tf.get_default_session().graph.get_operations()
        op_names = [e.name for e in op_list]
        target_names = [e for e in op_names if "miou" in e]
        '''
        sess, reset_miou, update_miou, cal_miou = self._get_from_graph()
        cal_miou = sess.run(cal_miou)
        ret = {"val_miou": cal_miou}
        return ret

def get_data(train_or_test, batch_size):
    is_train = train_or_test == 'train'

    filename_list = cfg.train_list if is_train else cfg.test_list
    ds = Data(filename_list, shuffle=is_train, flip=is_train, random_crop=is_train, test_set = not is_train)
    sample_num = ds.size()

    if is_train:
        augmentors = [
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
        ]
    else:
        augmentors = []

#    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, batch_size, remainder=not is_train)
    if is_train:
        ds = PrefetchDataZMQ(ds, min(6, multiprocessing.cpu_count()))
    return ds, sample_num

def get_config(args, model):

    ds_train, train_sample_num = get_data('train', args.batch_size_per_gpu)
    ds_test, _ = get_data('test', 1) #args.batch_size_per_gpu)

    training_number_of_steps = 300 * train_sample_num // (args.batch_size_per_gpu * get_nr_gpu())

    steps_per_epoch = train_sample_num // (args.batch_size_per_gpu * get_nr_gpu())

    epoch_num = int(cfg.max_itr_num / steps_per_epoch)

    callbacks = [
      ModelSaver(),
      InferenceRunner(ds_test, CalMIOU()),
      HyperParamSetterWithFunc('learning_rate', lambda e, x: (((cfg.base_lr - cfg.end_lr) * (1 - steps_per_epoch * e / cfg.max_itr_num) ** cfg.learning_power) + cfg.end_lr)),
      HumanHyperParamSetter('learning_rate'),
    ]

    return TrainConfig(
        dataflow=ds_train,
        callbacks=callbacks,
        model=model,
        steps_per_epoch=steps_per_epoch,
        max_epoch=epoch_num,
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default="1")
    parser.add_argument('--batch_size_per_gpu', help='batch size per gpu', type=int, default=8)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--flops', action='store_true', help='print flops and exit')
    parser.add_argument('--logdir', help='train log directory name')
    args = parser.parse_args()


    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = DeeplabModel()
    
    if args.flops:
        input_desc = [
            InputDesc(tf.uint8, [None, None, None, 3], 'input'),
            InputDesc(tf.uint8, [None, None, None, 1], 'label')
        ]
        input = PlaceholderInput()
        input.setup(input_desc)
        with TowerContext('', is_training=True):
            model.build_graph(*input.get_input_tensors())

        tf.profiler.profile(
            tf.get_default_graph(),
            cmd='op',
            options=tf.profiler.ProfileOptionBuilder.float_operation())
    else:
        if args.logdir != None:
            logger.set_logger_dir(os.path.join("train_log", args.logdir))
        else:
            logger.auto_set_dir()
        nr_tower = get_nr_gpu()
        config = get_config(args, model)

        if args.load:
            if args.load.endswith('npz'):
                config.session_init = DictRestore(dict(np.load(args.load)))
            else:
                config.session_init = SaverRestore(args.load)

        trainer = SyncMultiGPUTrainerParameterServer(nr_tower)
        launch_train_with_config(config, trainer)
