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
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.common import get_tf_version_number
from tensorpack.tfutils.summary import *
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
    def __init__(self, data_format='NHWC', depth=50, mode='resnet'):
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

            mean = [0.485, 0.456, 0.406]    # rgb
            std = [0.229, 0.224, 0.225]
            if bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32)
            image_std = tf.constant(std, dtype=tf.float32)
            image = (image - image_mean) / image_std
            return image


    def _get_inputs(self):
        return [InputDesc(tf.uint8, [None, cfg.crop_size[0], cfg.crop_size[1], 3], 'input'), 
                InputDesc(tf.uint8, [None, cfg.crop_size[0], cfg.crop_size[1], 1], 'label')
               ]
   
    def _get_logits(self, image):
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            return resnet_backbone(image, self.num_blocks, preresnet_group if self.mode == 'preact' else resnet_group, self.block_func)       
 
    def _build_graph(self, inputs):
        image, label = inputs
        self.batch_size = tf.shape(image)[0]
        org_label = label
        # when show image summary, first convert to RGB format
        image_rgb = tf.reverse(image, axis=[-1])
        label_without_255 = tf.where(tf.equal(label, cfg.ignore_label), tf.zeros_like(label), label)
        label_without_255 = tf.cast(label_without_255 * 10, tf.uint8) 
        tf.summary.image('input-image', image_rgb, max_outputs=3)
        tf.summary.image('input-label', label_without_255, max_outputs=3)

        image = DeeplabModel.image_preprocess(image, bgr=True)

        if self.data_format == "NCHW":
            image = tf.transpose(image, [0, 3, 1, 2])

        # the backbone part
        logits = self._get_logits(image)
        logits_size = logits.get_shape().as_list()[1:3]

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
        label = tf.reshape(label, shape=[-1])
        not_ignore_mask = tf.to_float(tf.not_equal(label, cfg.ignore_label)) * 1.0
        one_hot_label = tf.one_hot(label, cfg.num_classes, on_value=1.0, off_value=0.0)

        # valid_indices = tf.to_int32(label <= cfg.num_classes - 1)
        # valid_logits = tf.dynamic_partition(tf.reshape(logits, shape=[-1, cfg.num_classes]), valid_indices, num_partitions=2)[1]
        # valid_labels = tf.dynamic_partition(label, valid_indices, num_partitions=2)[1]

        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.to_int32(valid_labels), logits=valid_logits)
        # self.cost = tf.reduce_sum(loss, name='cost')
        if not cfg.freeze_batch_norm:
            train_var_list = [v for v in tf.trainable_variables()]
        else:
            train_var_list = [v for v in tf.trainable_variabels() if 'beta' not in v.name and 'gamma' not in v.name]
        cross_entropy = tf.losses.softmax_cross_entropy(one_hot_label, tf.reshape(logits, shape=[-1, cfg.num_classes]), weights=not_ignore_mask)
        
        self.cost = tf.add(cross_entropy, cfg.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in train_var_list]), name='cost')
        
        # self.cost = tf.identity(cost, name='cost')

        # compute the mean_iou
        # miou = MIOU(logits, label)
        predictions = tf.argmax(tf.nn.softmax(logits), 3, name='predicts')
        out_pred = predictions * 10
        out_pred = tf.cast(tf.expand_dims(out_pred, -1), tf.uint8)
        out_pred = tf.where(tf.equal(org_label, cfg.ignore_label), tf.zeros_like(org_label), out_pred)
        tf.summary.image('input-preds', tf.cast(out_pred, tf.uint8), max_outputs=3)

        predictions = tf.reshape(predictions, shape=[-1], name='flat_predicts')
        # weights = tf.to_float(tf.not_equal(labels, cfg.ignore_label))
        label = tf.where(tf.equal(label, cfg.ignore_label), tf.zeros_like(label), label)
        label = tf.cast(label, tf.int64)

        miou, miou_update_op = tf.metrics.mean_iou(label, predictions, cfg.num_classes, weights=not_ignore_mask)
        miou = tf.identity(miou, name='miou')
        miou_update_op = tf.identity(miou_update_op, name='miou_update_op')
    
        add_moving_summary(self.cost)

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', cfg.base_lr, summary=True)
        optimizer = tf.train.MomentumOptimizer(lr, cfg.momentum)
        
        return optimizer

class CalMIOU(Inferencer):

    def __init__(self):

        self.inf_miou_name = 'InferenceTower/miou:0'
        self.names = ["miou_update_op"]

    def _get_fetches(self):
        return self.names

    def _after_inference(self):
        sess = tf.get_default_session()
        graph = sess.graph
        # the following code should find the target names:
        #   ['tower0/miou', 'tower0/miou_update_op', ['tower1/miou', 'tower1/miou_update_op', ...
        #    'InferenceTower/miou', 'InferenceTower/miou_update_op']
        '''
        op_list = tf.get_default_session().graph.get_operations()
        op_names = [e.name for e in op_list]
        target_names = [e for e in op_names if "miou" in e]
        '''
        val_miou = graph.get_tensor_by_name(self.val_miou_name)
        val_miou = sess.run(val_miou)
        ret = {"val_miou": val_miou}
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
    ds_test, _ = get_data('test', args.batch_size_per_gpu)

    training_number_of_steps = 300 * train_sample_num // (args.batch_size_per_gpu * get_nr_gpu())

    steps_per_epoch = train_sample_num // (args.batch_size_per_gpu * get_nr_gpu())

    epoch_num = int(cfg.max_itr_num / steps_per_epoch)

    callbacks = [
      ModelSaver(),
      PeriodicTrigger(InferenceRunner(ds_test, CalMIOU()), every_k_epochs=3),
      HyperParamSetterWithFunc('learning_rate', lambda e, x: (((cfg.base_lr-cfg.end_lr) * (1 - steps_per_epoch * e / cfg.max_itr_num) ** cfg.learning_power)+cfg.end_lr)),
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
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default="0")
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
            InputDesc(tf.uint8, [None, cfg.crop_size[0], cfg.crop_size[1], 3], 'input'),
            InputDesc(tf.uint8, [None, cfg.crop_size[0], cfg.crop_size[1], 3], 'label')
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
