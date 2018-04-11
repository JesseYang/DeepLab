# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides flags that are common to scripts.

Common flags from train/eval/vis/export_model.py are collected in this script.
"""
import collections

import tensorflow as tf

from easydict import EasyDict as edict 

flags = tf.app.flags

# flags from train.py

FLAGS = flags.FLAGS

# Settings for multi-GPUs/multi-replicas training.

flags.DEFINE_string('gpu', '0', help='comma separated list of GPU(s) to use.')

flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy.')

flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')

flags.DEFINE_integer('num_replicas', 1, 'Number of worker replicas.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_integer('num_ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then '
                     'the parameters are handled locally by the worker.')

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

flags.DEFINE_integer('task', 0, 'The task ID.')

# Settings for logging.

flags.DEFINE_string('train_logdir', None,
                    'Where the checkpoint and logs are stored.')

flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')

flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')

flags.DEFINE_integer('save_summaries_secs', 600,
                     'How often, in seconds, we compute the summaries.')

flags.DEFINE_boolean('save_summaries_images', False,
                     'Save sample inputs, labels, and semantic predictions as images to summary.')

# Settings for training strategy.

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')

# Use 0.007 when training on PASCAL augmented training set, train_aug. When
# fine-tuning on PASCAL trainval set, use learning rate=0.0001.
flags.DEFINE_float('base_learning_rate', .0001,
                   'The base learning rate for model training.')

flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'The rate to decay the base learning rate.')

flags.DEFINE_integer('learning_rate_decay_step', 2000,
                     'Decay the base learning rate at a fixed step.')

flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')

flags.DEFINE_integer('training_number_of_steps', 30000,
                     'The number of steps used for training')

flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

# When fine_tune_batch_norm=True, use at least batch size larger than 12
# (batch size more than 16 is better). Otherwise, one could use smaller batch
# size and set fine_tune_batch_norm=False.
flags.DEFINE_integer('train_batch_size', 8,
                     'The number of images in each batch during training.')

flags.DEFINE_float('weight_decay', 0.00004,
                   'The value of the weight decay for training.')

flags.DEFINE_multi_integer('train_crop_size', [513, 513],
                           'Image crop size [height, width] during training.')

flags.DEFINE_float('last_layer_gradient_multiplier', 1.0,
                   'The gradient multiplier for last layers, which is used to '
                   'boost the gradient of last layers if the value > 1.')

flags.DEFINE_boolean('upsample_logits', True,
                     'Upsample logits during training.')

# Settings for fine-tuning the network.

flags.DEFINE_string('tf_initial_checkpoint', None,
                    'The initial checkpoint in tensorflow format.')

# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', True,
                     'Initialize the last layer.')

flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                     'Only consider logits as last layers or not.')

flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')

flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')

# Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
# Set to False and use small batch size to save GPU memory.
flags.DEFINE_boolean('fine_tune_batch_norm', True,
                     'Fine tune the batch norm parameters or not.')

flags.DEFINE_float('min_scale_factor', 0.5,
                   'Mininum scale factor for data augmentation.')

flags.DEFINE_float('max_scale_factor', 2.,
                   'Maximum scale factor for data augmentation.')

flags.DEFINE_float('scale_factor_step_size', 0.25,
                   'Scale factor step size for data augmentation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Dataset settings.
flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')
# end

# Flags for input preprocessing.

flags.DEFINE_integer('min_resize_value', None,
                     'Desired size of the smaller image side.')

flags.DEFINE_integer('max_resize_value', None,
                     'Maximum allowed size of the larger image side.')

flags.DEFINE_integer('resize_factor', None,
                     'Resized dimensions are multiple of factor plus one.')

# Model dependent flags.

flags.DEFINE_integer('logits_kernel_size', 1,
                     'The kernel size for the convolutional kernel that '
                     'generates logits.')

# When using 'mobilent_v2', we set atrous_rates = decoder_output_stride = None.
# When using 'xception_65', we set atrous_rates = [6, 12, 18] (output stride 16)
# and decoder_output_stride = 4.
flags.DEFINE_enum('model_variant', 'mobilenet_v2',
                  ['xception_65', 'mobilenet_v2'], 'DeepLab model variant.')

flags.DEFINE_multi_float('image_pyramid', None,
                         'Input scales for multi-scale feature extraction.')

flags.DEFINE_boolean('add_image_level_feature', True,
                     'Add image level feature.')

flags.DEFINE_boolean('aspp_with_batch_norm', True,
                     'Use batch norm parameters for ASPP or not.')

flags.DEFINE_boolean('aspp_with_separable_conv', True,
                     'Use separable convolution for ASPP or not.')

flags.DEFINE_multi_integer('multi_grid', None,
                           'Employ a hierarchy of atrous rates for ResNet.')

flags.DEFINE_float('depth_multiplier', 1.0,
                   'Multiplier for the depth (number of channels) for all '
                   'convolution ops used in MobileNet.')

# For `xception_65`, use decoder_output_stride = 4. For `mobilenet_v2`, use
# decoder_output_stride = None.
flags.DEFINE_integer('decoder_output_stride', None,
                     'The ratio of input to output spatial resolution when '
                     'employing decoder to refine segmentation results.')

flags.DEFINE_boolean('decoder_use_separable_conv', True,
                     'Employ separable convolution for decoder or not.')

flags.DEFINE_enum('merge_method', 'max', ['max', 'avg'],
                  'Scheme to merge multi scale features.')

FLAGS = flags.FLAGS

cfg = edict()

# backbone selecting 
name_scope = {
    'mobilenet_v2': 'MobilenetV2',
    'xception_65': 'xception_65',
}
DECODER_END_POINTS = 'decoder_end_points'
networks_to_feature_maps = {
    #'mobilenet_v2': {
        # The provided checkpoint does not include decoder module.
    #    DECODER_END_POINTS: None,
    #},
    'xception_65': {
        DECODER_END_POINTS: [
            'entry_flow/block2/unit_1/xception_module/'
            'separable_conv2_pointwise',
        ],
    }
}


# Constants

# Perform semantic segmentation predictions.
OUTPUT_TYPE = 'semantic'

# Semantic segmentation item names.
LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'

# Test set name.
TEST_SET = 'test'

cfg.ignore_label = 255
cfg.outputs_to_num_classes = {'semantic':21}
cfg.train_sample_num = 0
cfg.random_expand = True
cfg.random_inter = True
cfg.random_crop = True
cfg.mean_pixel = [127.5, 127.5, 127.5]
cfg.ignore_label = 255
cfg.train_list = ["train.txt"]
cfg.test_list = ["val.txt"]
cfg.crop_size = [513, 513]
cfg.model_variant = 'xception_65'
cfg.min_resize_value = None
cfg.max_resize_value = None
cfg.resize_factor = None
cfg.min_scale_factor = 0.5
cfg.max_scale_factor = 2.0
cfg.scale_factor_step_size = 0.25
cfg.PROB_OF_FLIP = 0.5


class ModelOptions(
    collections.namedtuple('ModelOptions', [
        'outputs_to_num_classes',
        'crop_size',
        'atrous_rates',
        'output_stride',
        'merge_method',
        'add_image_level_feature',
        'aspp_with_batch_norm',
        'aspp_with_separable_conv',
        'multi_grid',
        'decoder_output_stride',
        'decoder_use_separable_conv',
        'logits_kernel_size',
        'model_variant'
    ])):
  """Immutable class to hold model options."""

  __slots__ = ()

  def __new__(cls,
              outputs_to_num_classes,
              crop_size=None,
              atrous_rates=None,
              output_stride=8):
    """Constructor to set default values.

    Args:
      outputs_to_num_classes: A dictionary from output type to the number of
        classes. For example, for the task of semantic segmentation with 21
        semantic classes, we would have outputs_to_num_classes['semantic'] = 21.
      crop_size: A tuple [crop_height, crop_width].
      atrous_rates: A list of atrous convolution rates for ASPP.
      output_stride: The ratio of input to output spatial resolution.

    Returns:
      A new ModelOptions instance.
    """
    return super(ModelOptions, cls).__new__(
        cls, outputs_to_num_classes, crop_size, atrous_rates, output_stride,
        FLAGS.merge_method, FLAGS.add_image_level_feature,
        FLAGS.aspp_with_batch_norm, FLAGS.aspp_with_separable_conv,
        FLAGS.multi_grid, FLAGS.decoder_output_stride,
        FLAGS.decoder_use_separable_conv, FLAGS.logits_kernel_size,
        FLAGS.model_variant)
