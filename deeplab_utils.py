#!/usr/bin/env python
# -*- coding: UTF-8 -*-

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
import time

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorflow.python import debug as tf_debug

#orignal import and parameters

import common
import xception
try:
    from .common import cfg
    from .reader import Data
    # from .evaluation import mIOU
    # from .utils import postprocess
except Exception:
    from common import cfg
    from reader import Data
    # from evaluation import mIOU
    # from utils import postprocess
FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim

_LOGITS_SCOPE_NAME = 'logits'
_MERGED_LOGITS_SCOPE = 'merged_logits'
_IMAGE_POOLING_SCOPE = 'image_pooling'
_ASPP_SCOPE = 'aspp'
_CONCAT_PROJECTION_SCOPE = 'concat_projection'
_DECODER_SCOPE = 'decoder'


def get_extra_layer_scopes():
    """Gets the scopes for extra layers.
    Returns:
       A list of scopes for extra layers.
    """
    return [
        _LOGITS_SCOPE_NAME,
        _IMAGE_POOLING_SCOPE,
        _ASPP_SCOPE,
        _CONCAT_PROJECTION_SCOPE,
        _DECODER_SCOPE,
    ]


class DeeplabModel(ModelDesc):
    """docstring for DeeplabModel"""
    def __init__(self, data_format='NHWC'):
        super(DeeplabModel, self).__init__()
        self.data_format = data_format

    def _get_inputs(self):
        return [InputDesc(tf.uint8, [None, cfg.crop_size[0], cfg.crop_size[1], 3], 'input'), 
                InputDesc(tf.uint8, [None, cfg.crop_size[0], cfg.crop_size[1], 1], 'label')
               ]

    # name should be reconfirmed, anyway the role of this function is a backbone, same as feature_extractor.extract_features
    # @abstractmethod
    # def get_features_endpoints(self, ):
    #   pass

    def extract_features(self,
                         images,
                         output_stride=8,
                         multi_grid=None,
                         depth_multiplier=1.0,
                         final_endpoint=None,
                         model_variant=None,
                         weight_decay=0.0001,
                         reuse=None,
                         is_training=False,
                         fine_tune_batch_norm=False,
                         regularize_depthwise=False,
                         preprocess_images=True,
                         num_classes=None,
                         global_pool=False):
        features, end_points = xception.xception_65(inputs=images,
                                                    num_classes=num_classes,
                                                    is_training=(is_training and fine_tune_batch_norm),
                                                    global_pool=global_pool,
                                                    output_stride=output_stride,
                                                    regularize_depthwise=regularize_depthwise,
                                                    multi_grid=multi_grid,
                                                    reuse=reuse,
                                                    scope=common.name_scope[model_variant])
        return features, end_points

    def predict_labels_multi_scale(self,
                                   images,
                                   model_options,
                                   eval_scales=(1.0,),
                                   add_flipped_images=False):
        """Predicts segmentation labels.
        Args:
            images: A tensor of size [batch, height, width, channels].
            model_options: A ModelOptions instance to configure models.
            eval_scales: The scales to resize images for evaluation.
            add_flipped_images: Add flipped images for evaluation or not.
        Returns:
            A dictionary with keys specifying the output_type (e.g., semantic
            prediction) and values storing Tensors representing predictions (argmax
            over channels). Each prediction has size [batch, height, width].
        """
        outputs_to_predictions = {
            output: []
            for output in model_options.outputs_to_num_classes
        }

        for i, image_scale in enumerate(eval_scales):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True if i else None):
                outputs_to_scales_to_logits = self.multi_scale_logits(
                    images,
                    model_options=model_options,
                    image_pyramid=[image_scale],
                    is_training=False,
                    fine_tune_batch_norm=False)

            if add_flipped_images:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    outputs_to_scales_to_logits_reversed = self.multi_scale_logits(
                        tf.reverse_v2(images, [2]),
                        model_options=model_options,
                        image_pyramid=[image_scale],
                        is_training=False,
                        fine_tune_batch_norm=False)

            for output in sorted(outputs_to_scales_to_logits):
                scales_to_logits = outputs_to_scales_to_logits[output]
                logits = tf.image.resize_bilinear(
                    scales_to_logits[_MERGED_LOGITS_SCOPE],
                    tf.shape(images)[1:3],
                    align_corners=True)
                outputs_to_predictions[output].append(
                    tf.expand_dims(tf.nn.softmax(logits), 4))

                if add_flipped_images:
                    scales_to_logits_reversed = (
                    outputs_to_scales_to_logits_reversed[output])
                logits_reversed = tf.image.resize_bilinear(
                    tf.reverse_v2(scales_to_logits_reversed[_MERGED_LOGITS_SCOPE], [2]),
                    tf.shape(images)[1:3],
                    align_corners=True)
                outputs_to_predictions[output].append(
                    tf.expand_dims(tf.nn.softmax(logits_reversed), 4))

        for output in sorted(outputs_to_predictions):
            predictions = outputs_to_predictions[output]
            # Compute average prediction across different scales and flipped images.
            predictions = tf.reduce_mean(tf.concat(predictions, 4), axis=4)
            outputs_to_predictions[output] = tf.argmax(predictions, 3)

        return outputs_to_predictions


    def predict_labels(self, images, model_options, image_pyramid=None):
        """Predicts segmentation labels.
        Args:
          images: A tensor of size [batch, height, width, channels].
          model_options: A ModelOptions instance to configure models.
          image_pyramid: Input image scales for multi-scale feature extraction.
        Returns:
          A dictionary with keys specifying the output_type (e.g., semantic
            prediction) and values storing Tensors representing predictions (argmax
            over channels). Each prediction has size [batch, height, width].
        """
        outputs_to_scales_to_logits = self.multi_scale_logits(
            images,
            model_options=model_options,
            image_pyramid=image_pyramid,
            is_training=False,
            fine_tune_batch_norm=False)

        predictions = {}
        for output in sorted(outputs_to_scales_to_logits):
            scales_to_logits = outputs_to_scales_to_logits[output]
            logits = tf.image.resize_bilinear(
                scales_to_logits[_MERGED_LOGITS_SCOPE],
                tf.shape(images)[1:3],
                align_corners=True)
            predictions[output] = tf.argmax(logits, 3)

        return predictions


    def scale_dimension(self, dim, scale):
        """Scales the input dimension.
        Args:
          dim: Input dimension (a scalar or a scalar Tensor).
          scale: The amount of scaling applied to the input.
        Returns:
          Scaled dimension.
        """
        if isinstance(dim, tf.Tensor):
            return tf.cast((tf.to_float(dim) - 1.0) * scale + 1.0, dtype=tf.int32)
        else:
            return int((float(dim) - 1.0) * scale + 1.0)


    def multi_scale_logits(self, 
                           images,
                           model_options,
                           image_pyramid,
                           weight_decay=0.0001,
                           is_training=False,
                           fine_tune_batch_norm=False):
        """Gets the logits for multi-scale inputs.
        The returned logits are all downsampled (due to max-pooling layers)
        for both training and evaluation.
        Args:
          images: A tensor of size [batch, height, width, channels].
          model_options: A ModelOptions instance to configure models.
          image_pyramid: Input image scales for multi-scale feature extraction.
          weight_decay: The weight decay for model variables.
          is_training: Is training or not.
          fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
        Returns:
          outputs_to_scales_to_logits: A map of maps from output_type (e.g.,
            semantic prediction) to a dictionary of multi-scale logits names to
            logits. For each output_type, the dictionary has keys which
            correspond to the scales and values which correspond to the logits.
            For example, if `scales` equals [1.0, 1.5], then the keys would
            include 'merged_logits', 'logits_1.00' and 'logits_1.50'.
        Raises:
          ValueError: If model_options doesn't specify crop_size and its
            add_image_level_feature = True, since add_image_level_feature requires
            crop_size information.
        """
        # Setup default values.
        if not image_pyramid:
            image_pyramid = [1.0]

        if model_options.crop_size is None and model_options.add_image_level_feature:
            raise ValueError(
                'Crop size must be specified for using image-level feature.')
        if model_options.model_variant == 'mobilenet_v2':
            if (model_options.atrous_rates is not None or
                model_options.decoder_output_stride is not None):
                # Output a warning and users should make sure if the setting is desired.
                tf.logging.warning('Our provided mobilenet_v2 checkpoint does not include ASPP and decoder modules.')

        crop_height = (
            model_options.crop_size[0]
            if model_options.crop_size else tf.shape(images)[1])
        crop_width = (
            model_options.crop_size[1]
            if model_options.crop_size else tf.shape(images)[2])

        # Compute the height, width for the output logits.
        logits_output_stride = (
            model_options.decoder_output_stride or model_options.output_stride)

        logits_height = self.scale_dimension(
            crop_height,
            max(1.0, max(image_pyramid)) / logits_output_stride)
        logits_width = self.scale_dimension(
            crop_width,
            max(1.0, max(image_pyramid)) / logits_output_stride)

        # Compute the logits for each scale in the image pyramid.
        outputs_to_scales_to_logits = {
            k: {}
            for k in model_options.outputs_to_num_classes
        }

        for count, image_scale in enumerate(image_pyramid):
            if image_scale != 1.0:
                scaled_height = self.scale_dimension(crop_height, image_scale)
                scaled_width = self.scale_dimension(crop_width, image_scale)
                scaled_crop_size = [scaled_height, scaled_width]
                scaled_images = tf.image.resize_bilinear(
                    images, scaled_crop_size, align_corners=True)
                if model_options.crop_size:
                    scaled_images.set_shape([None, scaled_height, scaled_width, 3])
            else:
                scaled_crop_size = model_options.crop_size
                scaled_images = images

            updated_options = model_options._replace(crop_size=scaled_crop_size)
            outputs_to_logits = self._get_logits(
                scaled_images,
                updated_options,
                weight_decay=weight_decay,
                reuse=True if count else None,
                is_training=is_training,
                fine_tune_batch_norm=fine_tune_batch_norm)

            # Resize the logits to have the same dimension before merging.
            for output in sorted(outputs_to_logits):
                outputs_to_logits[output] = tf.image.resize_bilinear(
                    outputs_to_logits[output], [logits_height, logits_width],
                    align_corners=True)

            # Return when only one input scale.
            if len(image_pyramid) == 1:
                for output in sorted(model_options.outputs_to_num_classes):
                    outputs_to_scales_to_logits[output][_MERGED_LOGITS_SCOPE] = outputs_to_logits[output]
                return outputs_to_scales_to_logits

            # Save logits to the output map.
            for output in sorted(model_options.outputs_to_num_classes):
                outputs_to_scales_to_logits[output]['logits_%.2f' % image_scale] = outputs_to_logits[output]

        # Merge the logits from all the multi-scale inputs.
        for output in sorted(model_options.outputs_to_num_classes):
            # Concatenate the multi-scale logits for each output type.
            all_logits = [
                tf.expand_dims(logits, axis=4)
                for logits in outputs_to_scales_to_logits[output].values()
            ]
            all_logits = tf.concat(all_logits, 4)
            merge_fn = (
                tf.reduce_max
                if model_options.merge_method == 'max' else tf.reduce_mean)
            outputs_to_scales_to_logits[output][_MERGED_LOGITS_SCOPE] = merge_fn(
                all_logits, axis=4)

        return outputs_to_scales_to_logits


    def _extract_features(self, 
                          images,
                          model_options,
                          weight_decay=0.0001,
                          reuse=None,
                          is_training=False,
                          fine_tune_batch_norm=False):
        """Extracts features by the particular model_variant.
        Args:
          images: A tensor of size [batch, height, width, channels].
          model_options: A ModelOptions instance to configure models.
          weight_decay: The weight decay for model variables.
          reuse: Reuse the model variables or not.
          is_training: Is training or not.
          fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
        Returns:
          concat_logits: A tensor of size [batch, feature_height, feature_width,
            feature_channels], where feature_height/feature_width are determined by
            the images height/width and output_stride.
          end_points: A dictionary from components of the network to the corresponding
            activation.
        """
        features, end_points = self.extract_features(
            images,
            output_stride=model_options.output_stride,
            multi_grid=model_options.multi_grid,
            model_variant=model_options.model_variant,
            weight_decay=weight_decay,
            reuse=reuse,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm)

        if not model_options.aspp_with_batch_norm:
            return features, end_points
        else:
            batch_norm_params = {
                'is_training': is_training and fine_tune_batch_norm,
                'decay': 0.9997,
                'epsilon': 1e-5,
                'scale': True,
                }  
        
            with slim.arg_scope(
                [slim.conv2d, slim.separable_conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                padding='SAME',
                stride=1,
                reuse=reuse):
                with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                    depth = 256
                    branch_logits = []
  
                    if model_options.add_image_level_feature:
                        pool_height = self.scale_dimension(model_options.crop_size[0],
                                                      1. / model_options.output_stride)
                        pool_width = self.scale_dimension(model_options.crop_size[1],
                                                     1. / model_options.output_stride)
                        image_feature = slim.avg_pool2d(
                            features, [pool_height, pool_width], [pool_height, pool_width],
                            padding='VALID')
                        image_feature = slim.conv2d(
                            image_feature, depth, 1, scope=_IMAGE_POOLING_SCOPE)
                        image_feature = tf.image.resize_bilinear(
                            image_feature, [pool_height, pool_width], align_corners=True)
                        image_feature.set_shape([None, pool_height, pool_width, depth])
                        branch_logits.append(image_feature)
  
                    # Employ a 1x1 convolution.
                    branch_logits.append(slim.conv2d(features, depth, 1,
                                                     scope=_ASPP_SCOPE + str(0)))
  
                    if model_options.atrous_rates:
                        # Employ 3x3 convolutions with different atrous rates.
                        for i, rate in enumerate(model_options.atrous_rates, 1):
                            scope = _ASPP_SCOPE + str(i)
                            if model_options.aspp_with_separable_conv:
                                aspp_features = self._split_separable_conv2d(
                                    features,
                                    filters=depth,
                                    rate=rate,
                                    weight_decay=weight_decay,
                                    scope=scope)
                            else:
                                aspp_features = slim.conv2d(
                                    features, depth, 3, rate=rate, scope=scope)
                            branch_logits.append(aspp_features)
  
                    # Merge branch logits.
                    concat_logits = tf.concat(branch_logits, 3)
                    concat_logits = slim.conv2d(
                        concat_logits, depth, 1, scope=_CONCAT_PROJECTION_SCOPE)
                    concat_logits = slim.dropout(
                        concat_logits,
                        keep_prob=0.9,
                        is_training=is_training,
                        scope=_CONCAT_PROJECTION_SCOPE + '_dropout')
  
                    return concat_logits, end_points


    def _get_logits(self, 
                    images,
                    model_options,
                    weight_decay=0.0001,
                    reuse=None,
                    is_training=False,
                    fine_tune_batch_norm=False):
        """Gets the logits by atrous/image spatial pyramid pooling.
        Args:
            images: A tensor of size [batch, height, width, channels].
            model_options: A ModelOptions instance to configure models.
            weight_decay: The weight decay for model variables.
            reuse: Reuse the model variables or not.
            is_training: Is training or not.
            fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
        Returns:
            outputs_to_logits: A map from output_type to logits.
        """
        features, end_points = self._extract_features(
            images,
            model_options,
            weight_decay=weight_decay,
            reuse=reuse,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm)

        if model_options.decoder_output_stride is not None:
            decoder_height = self.scale_dimension(model_options.crop_size[0],
                                             1.0 / model_options.decoder_output_stride)
            decoder_width = self.scale_dimension(model_options.crop_size[1],
                                            1.0 / model_options.decoder_output_stride)
            features = self.refine_by_decoder(
                features,
                end_points,
                decoder_height=decoder_height,
                decoder_width=decoder_width,
                decoder_use_separable_conv=model_options.decoder_use_separable_conv,
                model_variant=model_options.model_variant,
                weight_decay=weight_decay,
                reuse=reuse,
                is_training=is_training,
                fine_tune_batch_norm=fine_tune_batch_norm)

        outputs_to_logits = {}
        for output in sorted(model_options.outputs_to_num_classes):
            outputs_to_logits[output] = self._get_branch_logits(
                features,
                model_options.outputs_to_num_classes[output],
                model_options.atrous_rates,
                aspp_with_batch_norm=model_options.aspp_with_batch_norm,
                kernel_size=model_options.logits_kernel_size,
                weight_decay=weight_decay,
                reuse=reuse,
                scope_suffix=output)

        return outputs_to_logits


    def refine_by_decoder(self, 
                          features,
                          end_points,
                          decoder_height,
                          decoder_width,
                          decoder_use_separable_conv=False,
                          model_variant=None,
                          weight_decay=0.0001,
                          reuse=None,
                          is_training=False,
                          fine_tune_batch_norm=False):
        """Adds the decoder to obtain sharper segmentation results.
        Args:
            features: A tensor of size [batch, features_height, features_width,
              features_channels].
            end_points: A dictionary from components of the network to the corresponding
              activation.
            decoder_height: The height of decoder feature maps.
            decoder_width: The width of decoder feature maps.
            decoder_use_separable_conv: Employ separable convolution for decoder or not.
            model_variant: Model variant for feature extraction.
            weight_decay: The weight decay for model variables.
            reuse: Reuse the model variables or not.
            is_training: Is training or not.
            fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
        Returns:
            Decoder output with size [batch, decoder_height, decoder_width,
            decoder_channels].
        """
        batch_norm_params = {
            'is_training': is_training and fine_tune_batch_norm,
            'decay': 0.9997,
            'epsilon': 1e-5,
            'scale': True,
            }

        with slim.arg_scope(
            [slim.conv2d, slim.separable_conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            padding='SAME',
            stride=1,
            reuse=reuse):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with tf.variable_scope(_DECODER_SCOPE, _DECODER_SCOPE, [features]):
                    feature_list = common.networks_to_feature_maps[
                        model_variant][common.DECODER_END_POINTS]
                    if feature_list is None:
                        tf.logging.info('Not found any decoder end points.')
                        return features
                    else:
                        decoder_features = features
                        for i, name in enumerate(feature_list):
                            decoder_features_list = [decoder_features]
                            feature_name = '{}/{}'.format(
                                common.name_scope[model_variant], name)
                            decoder_features_list.append(
                                slim.conv2d(
                                    end_points[feature_name],
                                    48,
                                    1,
                                    scope='feature_projection' + str(i)))
                            # Resize to decoder_height/decoder_width.
                            for j, feature in enumerate(decoder_features_list):
                                decoder_features_list[j] = tf.image.resize_bilinear(
                                    feature, [decoder_height, decoder_width], align_corners=True)
                                decoder_features_list[j].set_shape(
                                    [None, decoder_height, decoder_width, None])
                            decoder_depth = 256
                            if decoder_use_separable_conv:
                                decoder_features = self._split_separable_conv2d(
                                    tf.concat(decoder_features_list, 3),
                                    filters=decoder_depth,
                                    rate=1,
                                    weight_decay=weight_decay,
                                    scope='decoder_conv0')
                                decoder_features = self._split_separable_conv2d(
                                    decoder_features,
                                    filters=decoder_depth,
                                    rate=1,
                                    weight_decay=weight_decay,
                                    scope='decoder_conv1')
                            else:
                                num_convs = 2
                                decoder_features = slim.repeat(
                                    tf.concat(decoder_features_list, 3),
                                    num_convs,
                                    slim.conv2d,
                                    decoder_depth,
                                    3,
                                    scope='decoder_conv' + str(i))
                        return decoder_features


    def _get_branch_logits(self, 
                           features,
                           num_classes,
                           atrous_rates=None,
                           aspp_with_batch_norm=False,
                           kernel_size=1,
                           weight_decay=0.0001,
                           reuse=None,
                           scope_suffix=''):
        """Gets the logits from each model's branch.
        The underlying model is branched out in the last layer when atrous
        spatial pyramid pooling is employed, and all branches are sum-merged
        to form the final logits.
        Args:
          features: A float tensor of shape [batch, height, width, channels].
          num_classes: Number of classes to predict.
          atrous_rates: A list of atrous convolution rates for last layer.
          aspp_with_batch_norm: Use batch normalization layers for ASPP.
          kernel_size: Kernel size for convolution.
          weight_decay: Weight decay for the model variables.
          reuse: Reuse model variables or not.
          scope_suffix: Scope suffix for the model variables.
        Returns:
          Merged logits with shape [batch, height, width, num_classes].
        Raises:
          ValueError: Upon invalid input kernel_size value.
        """
        # When using batch normalization with ASPP, ASPP has been applied before
        # in _extract_features, and thus we simply apply 1x1 convolution here.
        if aspp_with_batch_norm or atrous_rates is None:
            if kernel_size != 1:
                raise ValueError('Kernel size must be 1 when atrous_rates is None or '
                                 'using aspp_with_batch_norm. Gets %d.' % kernel_size)
            atrous_rates = [1]

        with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            reuse=reuse):
            with tf.variable_scope(_LOGITS_SCOPE_NAME, _LOGITS_SCOPE_NAME, [features]):
                branch_logits = []
                for i, rate in enumerate(atrous_rates):
                    scope = scope_suffix
                    if i:
                        scope += '_%d' % i

                    branch_logits.append(
                        slim.conv2d(
                            features,
                            num_classes,
                            kernel_size=kernel_size,
                            rate=rate,
                            activation_fn=None,
                            normalizer_fn=None,
                            scope=scope))

                return tf.add_n(branch_logits)


    def _split_separable_conv2d(self, 
                                inputs,
                                filters,
                                rate=1,
                                weight_decay=0.00004,
                                depthwise_weights_initializer_stddev=0.33,
                                pointwise_weights_initializer_stddev=0.06,
                                scope=None):
        """Splits a separable conv2d into depthwise and pointwise conv2d.
        This operation differs from `tf.layers.separable_conv2d` as this operation
        applies activation function between depthwise and pointwise conv2d.
        Args:
          inputs: Input tensor with shape [batch, height, width, channels].
          filters: Number of filters in the 1x1 pointwise convolution.
          rate: Atrous convolution rate for the depthwise convolution.
          weight_decay: The weight decay to use for regularizing the model.
          depthwise_weights_initializer_stddev: The standard deviation of the
            truncated normal weight initializer for depthwise convolution.
          pointwise_weights_initializer_stddev: The standard deviation of the
            truncated normal weight initializer for pointwise convolution.
          scope: Optional scope for the operation.
        Returns:
          Computed features after split separable conv2d.
        """
        outputs = slim.separable_conv2d(
            inputs,
            None,
            3,
            depth_multiplier=1,
            rate=rate,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=depthwise_weights_initializer_stddev),
            weights_regularizer=None,
            scope=scope + '_depthwise')
        return slim.conv2d(
            outputs,
            filters,
            1,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=pointwise_weights_initializer_stddev),
            weights_regularizer=slim.l2_regularizer(weight_decay),
            scope=scope + '_pointwise')

    def add_softmax_cross_entropy_loss_for_each_scale(scales_to_logits,
                                                      labels,
                                                      num_classes,
                                                      ignore_label,
                                                      loss_weight=1.0,
                                                      upsample_logits=True,
                                                      scope=None):
    # """Adds softmax cross entropy loss for logits of each scale.
    # Args:
    #   scales_to_logits: A map from logits names for different scales to logits.
    #     The logits have shape [batch, logits_height, logits_width, num_classes].
    #   labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
    #   num_classes: Integer, number of target classes.
    #   ignore_label: Integer, label to ignore.
    #   loss_weight: Float, loss weight.
    #   upsample_logits: Boolean, upsample logits or not.
    #   scope: String, the scope for the loss.
    # Raises:
    #   ValueError: Label or logits is None.
    # """
        pdb.set_trace()
        if labels is None:
            raise ValueError('No label for softmax cross entropy loss.')
        
        for scale, logits in scales_to_logits.items():
            loss_scope = None
            if scope:
                loss_scope = '%s_%s' % (scope, scale)

            if upsample_logits:
                # Label is not downsampled, and instead we upsample logits.
                logits = tf.image.resize_bilinear(
                    logits, tf.shape(labels)[1:3], align_corners=True)
                scaled_labels = labels
            else:
                # Label is downsampled to the same size as logits.
                scaled_labels = tf.image.resize_nearest_neighbor(
                    labels, tf.shape(logits)[1:3], align_corners=True)

            scaled_labels = tf.reshape(scaled_labels, shape=[-1])
            not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels,
                                                       ignore_label)) * loss_weight
            one_hot_labels = slim.one_hot_encoding(
                scaled_labels, num_classes, on_value=1.0, off_value=0.0)
            tf.losses.softmax_cross_entropy(
                one_hot_labels,
                tf.reshape(logits, shape=[-1, num_classes]),
                weights=not_ignore_mask,
                scope=loss_scope)

    def get_model_learning_rate(self, learning_policy, base_learning_rate, learning_rate_decay_step,
        learning_rate_decay_factor, training_number_of_steps, learning_power,
        slow_start_step, slow_start_learning_rate):
        """Gets model's learning rate.
        Computes the model's learning rate for different learning policy.
        Right now, only "step" and "poly" are supported.
        (1) The learning policy for "step" is computed as follows:
          current_learning_rate = base_learning_rate *
            learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
        See tf.train.exponential_decay for details.
        (2) The learning policy for "poly" is computed as follows:
          current_learning_rate = base_learning_rate *
            (1 - global_step / training_number_of_steps) ^ learning_power
        Args:
          learning_policy: Learning rate policy for training.
          base_learning_rate: The base learning rate for model training.
          learning_rate_decay_step: Decay the base learning rate at a fixed step.
          learning_rate_decay_factor: The rate to decay the base learning rate.
          training_number_of_steps: Number of steps for training.
          learning_power: Power used for 'poly' learning policy.
          slow_start_step: Training model with small learning rate for the first
            few steps.
          slow_start_learning_rate: The learning rate employed during slow start.
        Returns:
          Learning rate for the specified learning policy.
        Raises:
          ValueError: If learning policy is not recognized.
        """
        global_step = tf.train.get_or_create_global_step()
        if learning_policy == 'step':
            learning_rate = tf.train.exponential_decay(
                base_learning_rate,
                global_step,
                learning_rate_decay_step,
                learning_rate_decay_factor,
                staircase=True)
        elif learning_policy == 'poly':
            learning_rate = tf.train.polynomial_decay(
                base_learning_rate,
                global_step,
                training_number_of_steps,
                end_learning_rate=0,
                power=learning_power)
        else:
            raise ValueError('Unknown learning policy.')

        # Employ small learning rate at the first few steps for warm start.
        return tf.where(global_step < slow_start_step, slow_start_learning_rate,
                        learning_rate)

    def _build_graph(self, inputs):
        image, label = inputs
        self.batch_size = tf.shape(image)[0]

        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)

        # when show image summary, first convert to RGB format
        image_rgb = tf.reverse(image, axis=[-1])
        # image_with_bbox = tf.image.draw_bounding_boxes(image_rgb, gt_bbox)
        tf.summary.image('input-image', image_rgb, max_outputs=3)

        image = image * (1.0 / 255)

        if self.data_format == "NCHW":
            image = tf.transpose(image, [0, 3, 1, 2])

        model_options = common.ModelOptions(
            outputs_to_num_classes=cfg.outputs_to_num_classes,
            crop_size=FLAGS.train_crop_size,
            atrous_rates=FLAGS.atrous_rates,
            output_stride=FLAGS.output_stride)
        outputs_to_scales_to_logits = self.multi_scale_logits(
            image,
            model_options=model_options,
            image_pyramid=FLAGS.image_pyramid,
            weight_decay=FLAGS.weight_decay,
            is_training=True,
            fine_tune_batch_norm=FLAGS.fine_tune_batch_norm)

        # pdb.set_trace()
        # for output, num_classes in cfg.outputs_to_num_classes.items():
        #     self.add_softmax_cross_entropy_loss_for_each_scale(
        #         outputs_to_scales_to_logits[output],
        #         label,
        #         num_classes,
        #         cfg.ignore_label,
        #         # loss_weight=1.0,
        #         upsample_logits=FLAGS.upsample_logits,
        #         scope=output)

        for output, num_classes in cfg.outputs_to_num_classes.items():
            scope = output
            upsample_logits=FLAGS.upsample_logits
            loss_weight=1.0
            if label is None:
                raise ValueError('No label for softmax cross entropy loss.')
            
            for scale, logits in outputs_to_scales_to_logits[output].items():
                loss_scope = None
                if scope:
                    loss_scope = '%s_%s' % (scope, scale)

                if upsample_logits:
                    # Label is not downsampled, and instead we upsample logits.
                    logits = tf.image.resize_bilinear(
                        logits, tf.shape(label)[1:3], align_corners=True)
                    scaled_labels = label
                else:
                    # Label is downsampled to the same size as logits.
                    scaled_labels = tf.image.resize_nearest_neighbor(
                        label, tf.shape(logits)[1:3], align_corners=True)

                scaled_labels = tf.reshape(scaled_labels, shape=[-1])
                not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels,
                                                           cfg.ignore_label)) * loss_weight
                one_hot_labels = slim.one_hot_encoding(
                    scaled_labels, num_classes, on_value=1.0, off_value=0.0)
                self.cost = tf.losses.softmax_cross_entropy(
                    one_hot_labels,
                    tf.reshape(logits, shape=[-1, num_classes]),
                    weights=not_ignore_mask,
                    scope=loss_scope)

        # total_loss = []
        # # summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        # for loss in tf.get_collection(tf.GraphKeys.LOSSES):
        #     # summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
        #     add_moving_summary(loss)
        #     total_loss.append(loss)
        #     # loss_op.append(loss.op.name)
        
        # self.cost = tf.add_n(total_loss, name='cost') 

        # return outputs_to_scales_to_logits

    def _get_optimizer(self):
        learning_rate = self.get_model_learning_rate(
            FLAGS.learning_policy, FLAGS.base_learning_rate,
            FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
            FLAGS.training_number_of_steps, FLAGS.learning_power,
            FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
        lr = get_scalar_var('learning_rate', 0.01, summary=True)
        optimizer = tf.train.MomentumOptimizer(lr, FLAGS.momentum)

        return optimizer

class mIOU(Inferencer):
    # def __init__(self, test_path):
    #     # self.names = ["loc_pred", "cls_pred", "ori_shape", "loss"]
    #     self.test_path = test_path
    #     self.image_format = 'jpg'
    #     self.label_format = 'png'
    #     self.gt_dir = "result_gt"
    #     if os.path.isdir(self.gt_dir):
    #         shutil.rmtree(self.gt_dir)

    #     self.pred_dir = "result_pred/"
    #     if os.path.isdir(self.pred_dir):
    #         shutil.rmtree(self.pred_dir)
    #     os.mkdir(self.pred_dir)

    #     with open(self.test_path) as f:
    #         content = f.readlines()

    #     self.image_path_list = []
    #     self.label_path_list = []
    #     for line in content:
    #         self.image_path_list.append(os.path.join(
    #         'VOC2012/JPEGImages' + '/' + line.split() + '.' + self.image_format))
    #         self.label_path_list.append(os.path.join(
    #         'VOC2012/JPEGImages' + '/' + line.split() + '.' + self.label_format))

    #     self.cur_image_idx = 0

    # def _get_fetches(self):
    #     return self.names

    # # def _before_inference(self):
    # #     # if the "result_gt" dir does not exist, generate it from the data_set
    # #     generate_gt_result(self.test_path, self.gt_dir, overwrite=False)
    # #     self.results = { }
    # #     self.loss = []
    # #     self.cur_image_idx = 0

    # def _on_fetches(self, output):
    #     self.loss.append(output[3])
    #     output = output[0:3]
    #     for i in range(output[0].shape[0]):
    #         # for each ele in the batch
    #         image_path = self.image_path_list[self.cur_image_idx]
    #         self.cur_image_idx += 1
    #         image_id = os.path.basename(image_path).split('.')[0] if cfg.gt_format == "voc" else image_path

    #         cur_output = [ele[i] for ele in output]
    #         predictions = [np.expand_dims(ele, axis=0) for ele in cur_output[0:2]]
    #         image_shape = cur_output[2]


    #         pred_results = postprocess(predictions, image_shape=image_shape)
    #         for class_name in pred_results.keys():
    #             if class_name not in self.results.keys():
    #                 self.results[class_name] = []
    #             for box in pred_results[class_name]:
    #                 record = [image_id]
    #                 record.extend(box)
    #                 record = [str(ele) for ele in record]
    #                 self.results[class_name].append(' '.join(record))

    # def _after_inference(self):
    #     # write the result to file
    #     for class_name in self.results.keys():
    #         with open(os.path.join(self.pred_dir, class_name + ".txt"), 'wt') as f:
    #             for record in self.results[class_name]:
    #                 f.write(record + '\n')
    #     # calculate the mAP based on the predicted result and the ground truth
    #     aps = do_python_eval(self.pred_dir)
    #     ap_result = { "mAP": np.mean(aps) }
    #     for idx, cls in enumerate(cfg.classes_name):
    #         ap_result["_" + cls] = aps[idx]
    #     # return { "mAP": np.mean(aps) }
    #     return ap_result
    pass




def get_data(train_or_test, batch_size):

    isTrain = train_or_test == 'train'

    filename_list = cfg.train_list if isTrain else cfg.test_list
    #still needs to confirm:
    ds = Data(filename_list, shuffle=isTrain, flip=isTrain, random_crop=cfg.random_crop and isTrain, random_expand=cfg.random_expand and isTrain)
    sample_num = ds.size()

    # if isTrain:
    #     augmentors = [
    #         imgaug.RandomOrderAug(
    #             [imgaug.Brightness(32, clip=False),
    #              imgaug.Contrast((0.5, 1.5), clip=False),
    #              imgaug.Saturation(0.5),
    #              imgaug.Hue((-18, 18), rgb=True)]),
    #         imgaug.Clip(),
    #         imgaug.ToUint8()
    #     ]
    # else:
    #     augmentors = [
    #         imgaug.ToUint8()
    #     ]
    # ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, batch_size, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(6, multiprocessing.cpu_count()))
    return ds, sample_num

def get_config(args, model):

    batch_size_per_gpu = int(args.train_batch_size / args.num_clones)
    ds_train, train_sample_num = get_data('train', batch_size_per_gpu)
    ds_test, _ = get_data('test', batch_size_per_gpu)

    loss_op = []

    for loss in tf.get_collection(tf.GraphKeys.LOSSES):
        loss_op.append(loss.op.name)

    callbacks = [
      ModelSaver(),
      # PeriodicTrigger(InferenceRunner(ds_test,
      #                                 ScalarStats(loss_op)),
      #                 every_k_epochs=3),
      # ScheduledHyperParamSetter('learning_rate',
      #                           cfg.lr_schedule),
      HyperParamSetterWithFunc('learning_rate',
                               lambda e, x: (0.5 * 1e-3 * (1 + np.cos((e - 100) / 200 * np.pi))) if e >= 100 else 1e-3 ),
      HumanHyperParamSetter('learning_rate'),
    ]
    # if cfg.mIOU == True:
    #     callbacks.append(EnableCallbackIf(PeriodicTrigger(InferenceRunner(ds_test,
    #                                                                      [MIOU(cfg.test_list)]),
    #                                       every_k_epochs=3),
    #                      lambda x : x.epoch_num >= 10))

    return TrainConfig(
        dataflow=ds_train,
        callbacks=callbacks,
        model=model,
        steps_per_epoch=train_sample_num // (batch_size_per_gpu * get_nr_gpu()),
        # max_epoch=300,
    )
        
