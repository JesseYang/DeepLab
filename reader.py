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

"""Contains common utility functions and classes for building dataset.

This script contains utility functions and classes to converts dataset to
TFRecord file format with Example protos.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import collections
import tensorflow as tf


import os, sys, shutil
import time
import pickle
import numpy as np
import random
#from scipy import misc
#import six
#from six.moves import urllib, range
import copy
import logging
import cv2
import json
import uuid
try:
    from .cfgs.config import cfg
except Exception:
    from cfgs.config import cfg

from tensorpack import *
from preprocess_utils import *


SAVE_DIR = 'input_images'

class Data(RNGDataFlow):
    def __init__(self,
                 filename_list, 
                 shuffle, flip, 
                 random_crop, 
                 random_expand, 
                 save_img=False, 
                 image_format='jpg', 
                 label_format='png', 
                 channels=3):
        self.filename_list = filename_list
        self.save_img = save_img
        self._image_format = image_format
        self._label_format = label_format

        if save_img == True:
            if os.path.isdir(SAVE_DIR):
                shutil.rmtree(SAVE_DIR)
            os.mkdir(SAVE_DIR)

        if isinstance(filename_list, list) == False:
            filename_list = [filename_list]

        content = []
        for filename in filename_list:
            with open(filename) as f:
                content.extend(f.readlines())

        self.img_list = [x.split(' ')[0] for x in content] 
        self.label_list = [x.split(' ')[1] for x in content] 
        self.shuffle = shuffle
        self.flip = flip
        self.random_crop = random_crop
        self.random_expand = random_expand

    def size(self):
        return len(self.img_list)

    def generate_sample(self, idx):

        # hflip = False if self.flip == False else (random.random() > 0.5)
        img_path = self.img_list[idx].strip()
        label_path = self.label_list[idx].strip()

        image = cv2.imread(img_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
  
        # s = image.shape
        # h, w, c = image.shape

        if self.save_img:
            cv2.imwrite(os.path.join(SAVE_DIR, 'image_%d.jpg' % idx), image)
            cv2.imwrite(os.path.join(SAVE_DIR, 'label_%d.jpg' % idx), label)

        ori_image = image.copy()
        ori_label = label.copy()

        '''
        if FLAGS.min_resize_value is not None or FLAGS.max_resize_value is not None:
            [image, label] = (resize_to_range(image=image,
                                              label=label, 
                                              min_size=cfg.min_resize_value, 
                                              max_size=cfg.max_resize_value,
                                              factor=cfg.resize_factor,
                                              align_corners=True))
        '''

        if self.random_crop:

            scale = get_random_scale(cfg.min_scale_factor,
                cfg.max_scale_factor, cfg.scale_factor_step_size)
            image, label = randomly_scale_image_and_label(
                image, label, scale)

            image_shape = image.shape
            image_height = image_shape[0]
            image_width = image_shape[1]

            target_height = image_height + max(cfg.img_h - image_height, 0)
            target_width = image_width + max(cfg.img_w - image_width, 0)

            # Pad image with mean pixel value.
            mean_pixel = np.reshape(cfg.mean_pixel, [1, 1, 3])
            image = pad_to_bounding_box(image, 0, 0, target_height, target_width, cfg.mean_pixel)

            if label is not None:
                label = pad_to_bounding_box(
                    label, 0, 0, target_height, target_width, cfg.ignore_label)

            # Randomly crop the image and label.
            if label is not None:
                image, label = random_crop(
                    image, label, cfg.img_h, cfg.img_w)

            if self.flip:
                image, label = flip_dim(image, label, prob=cfg.flip_prob, dim=1)

        aug_img = np.copy(image) if self.save_img else None
        aug_label = np.copy(label) if self.save_img else None

        if self.save_img:
            cv2.imwrite(os.path.join(SAVE_DIR, "%d_image_aug.jpg" % idx), aug_img)
            cv2.imwrite(os.path.join(SAVE_DIR, "%d_label_aug.jpg" % idx), aug_label)

        label = np.expand_dims(label, axis=-1)
        return [image, label]

    def get_data(self):
        idxs = np.arange(len(self.img_list))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            retval = self.generate_sample(k)
            if retval == None:
                continue
            yield retval

    def reset_state(self):
        super(Data, self).reset_state()

if __name__ == '__main__':
    # df = Data('voc_train_sbd_aug.txt', shuffle=False, flip=True, random_crop=True, random_expand=True, save_img=True)
    df = Data('voc_train_sbd_aug.txt', shuffle=False, flip=False, random_crop=False, random_expand=False, save_img=True)
    df.reset_state()

    g = df.get_data()
    for i in range(20):
        next(g)

