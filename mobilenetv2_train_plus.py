import pdb
import cv2
import sys
import argparse
import numpy as np
import os
import shutil
import multiprocessing
import json

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import init_ops
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.common import get_tf_version_number
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.models import * 
from tensorpack.callbacks import *
from tensorpack.train import *
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.tfutils.scope_utils import under_name_scope

from cfgs.config import cfg
from reader import Data

@layer_register(log_shape=True)
def DepthConv(x, out_channel, kernel_shape, padding='SAME', stride=1,
              W_init=None, nl=tf.identity, data_format='NHWC'):
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[1] if data_format=='NCHW' else in_shape[3]
    assert out_channel % in_channel == 0
    channel_mult = out_channel // in_channel

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer()
    kernel_shape = [kernel_shape, kernel_shape]
    filter_shape = kernel_shape + [in_channel, channel_mult]

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    conv = tf.nn.depthwise_conv2d(x, W, [1, stride, stride, 1], padding=padding, data_format=data_format)
    return nl(conv, name='output')


def BN(x, name):
    return BatchNorm('bn', x)

def BNReLU6(x, name):
    x = BN(x, 'bn')
    return tf.nn.relu6(x, name=name)

def atrous_spatial_pyramid_pooling(logits):
    # Compute the ASPP.
    logits_size = tf.shape(logits)[1:3]
    with argscope(Conv2D, filters=256, kernel_size=3, activation=BNReLU):
        ASPP_1 = Conv2D('aspp_conv1', logits, kernel_size=1)
        ASPP_2 = Conv2D('aspp_conv2', logits, dilation_rate=cfg.atrous_rates[0])
        ASPP_3 = Conv2D('aspp_conv3', logits, dilation_rate=cfg.atrous_rates[1])
        ASPP_4 = Conv2D('aspp_conv4', logits, dilation_rate=cfg.atrous_rates[2])
        # ImagePooling = GlobalAvgPooling('image_pooling', logits)
        ImagePooling = tf.reduce_mean(logits, [1, 2], name='global_average_pooling', keepdims=True)
        image_level_features = Conv2D('image_level_conv', ImagePooling, kernel_size=1)
    image_level_features = tf.image.resize_bilinear(image_level_features, logits_size, name='upsample')
    output = tf.concat([ASPP_1, ASPP_2, ASPP_3, ASPP_4, image_level_features], -1, name='concat')
    output = Conv2D('conv_after_concat', output, 256, 1, activation=BNReLU)
    return output
 

class DeeplabModel(ModelDesc):
    def __init__(self, data_format='NHWC'):
        super(DeeplabModel, self).__init__()
        self.data_format = data_format

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

    def _get_inputs(self):
        return [InputDesc(tf.uint8, [None, None, None, 3], 'input'),
                InputDesc(tf.uint8, [None, None, None, 1], 'label')]
   
    def _get_logits(self, image):

        def bottleneck_v2(l, t, out_channel, stride=1):
            in_shape = l.get_shape().as_list()

            in_channel = in_shape[1] if self.data_format == "NCHW" else in_shape[3]
            shortcut = l
            l = Conv2D('conv1', l, t*in_channel, 1, nl=BNReLU6)
            mid_l = l
            l = DepthConv('depthconv', l, t*in_channel, 3, stride=stride, nl=BNReLU6)
            l = Conv2D('conv2', l, out_channel, 1, nl=BN)
            if stride == 1 and out_channel == in_channel:
                l = l + shortcut
            return l, mid_l

        with argscope([Conv2D, GlobalAvgPooling, BatchNorm], data_format=self.data_format), \
                argscope([Conv2D], use_bias=False):
            l = Conv2D('covn1', image, 32, 3, stride=2, nl=BNReLU6)
            with tf.variable_scope('bottleneck1'):
                l, _ = bottleneck_v2(l, out_channel=16, t=1, stride=1)

            with tf.variable_scope('bottleneck2'):
                for j in range(2):
                    with tf.variable_scope('block{}'.format(j)):
                        l, _ = bottleneck_v2(l, out_channel=24, t=6, stride=2 if j == 0 else 1)

            with tf.variable_scope('bottleneck3'):
                for j in range(3):
                    with tf.variable_scope('block{}'.format(j)):
                        if j == 0:
                            l, mid_l = bottleneck_v2(l, out_channel=32, t=6, stride=2 if j == 0 else 1)
                        else:              
                            l, _ = bottleneck_v2(l, out_channel=32, t=6, stride=2 if j == 0 else 1)

            with tf.variable_scope('bottleneck4'):
                for j in range(4):
                    with tf.variable_scope('block{}'.format(j)):
                        l, _ = bottleneck_v2(l, out_channel=64, t=6, stride=2 if j == 0 else 1)
            
            with tf.variable_scope('bottleneck5'):
                for j in range(3):
                    with tf.variable_scope('block{}'.format(j)):
                        l, _ = bottleneck_v2(l, out_channel=96, t=6, stride=1)
            
            with tf.variable_scope('bottleneck6'):
                for j in range(3):
                    with tf.variable_scope('block{}'.format(j)):
                        l, _ = bottleneck_v2(l, out_channel=160, t=6, stride=2 if j== 0 else 1)
            with tf.variable_scope('bottleneck7'):
                l, _ = bottleneck_v2(l, out_channel=320, t=6, stride=1)
            logits = Conv2D('conv2', l, 1280, 1, nl=BNReLU6)
            
            return logits, mid_l
 
    def _build_graph(self, inputs):
        image, label = inputs
        self.batch_size = tf.shape(image)[0]
        self.image_size = tf.shape(image)[1:3]
        org_label = label
        # when show image summary, first convert to RGB format
        image_rgb = tf.reverse(image, axis=[-1])
        label_shown = tf.where(tf.equal(label, cfg.ignore_label), tf.zeros_like(label), label)
        label_shown = tf.cast(label_shown * 10, tf.uint8) 
        tf.summary.image('input-image', image_rgb, max_outputs=3)
        tf.summary.image('input-label', label_shown, max_outputs=3)

        image = DeeplabModel.image_preprocess(image, bgr=True)

        if self.data_format == "NCHW":
            image = tf.transpose(image, [0, 3, 1, 2])

        # the backbone part
        logits, low_level_features = self._get_logits(image)
        encoder_output = atrous_spatial_pyramid_pooling(logits)
        
        with tf.variable_scope('decoder'):
            with tf.variable_scope('low_level_features'):
                low_level_features = Conv2D('conv_1x1', low_level_features, 48, 1, strides=1, activation=tf.nn.relu)
                low_level_features_size = tf.shape(low_level_features)[1:3]
            with tf.variable_scope('upsampling_logits'):
                net = tf.image.resize_bilinear(encoder_output, low_level_features_size, name='upsample_1')
                net = tf.concat([net, low_level_features], axis=3, name='concat')
                with argscope(Conv2D, filters=256, kernel_size=3, strides=1, activation=tf.nn.relu):
                    net = (LinearWrap(net)
                         .Conv2D('conv_3x3_1')
                         .Conv2D('conv_3x3_2')
                         .Conv2D('conv_1x1', filters=cfg.num_classes, kernel_size=1, strides=1, activation=None)())

        # Compute softmax cross entropy loss for logits
        logits = tf.image.resize_bilinear(net, self.image_size, align_corners=True)
        label_flatten = tf.reshape(label, shape=[-1])
        mask = tf.to_float(tf.not_equal(label_flatten, cfg.ignore_label)) * 1.0
        one_hot_label = tf.one_hot(label_flatten, cfg.num_classes, on_value=1.0, off_value=0.0)

        loss = tf.losses.softmax_cross_entropy(one_hot_label, tf.reshape(logits, shape=[-1, cfg.num_classes]), weights=mask)
        if cfg.weight_decay > 0:
            wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
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
        # lr = get_scalar_var('learning_rate', cfg.base_lr, summary=True)
        lr = get_scalar_var('learning_rate', 1e-3, summary=True)
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
