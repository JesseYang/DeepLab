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
                 test_set = False, 
                 save_img=False, 
                 image_format='jpg', 
                 label_format='png', 
                 channels=3):
        self.filename_list = filename_list
        self.test_set = test_set
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

    def size(self):
        return len(self.img_list)

    def generate_sample(self, idx):

        # hflip = False if self.flip == False else (random.random() > 0.5)
        img_path = self.img_list[idx].strip()
        label_path = self.label_list[idx].strip()

        image = cv2.imread(img_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                      
        if self.save_img:
            cv2.imwrite(os.path.join(SAVE_DIR, 'image_%d.jpg' % idx), image)
            cv2.imwrite(os.path.join(SAVE_DIR, 'label_%d.jpg' % idx), label)
        if self.test_set:
            image_height = image.shape[0]
            image_width = image.shape[1]
            target_height = image_height + max(cfg.crop_size[0] - image_height, 0)
            target_width = image_width + max(cfg.crop_size[1] - image_width, 0)
            top = int(max(target_height - image_height, 0)/2)
            bottom = max(target_height - image_height - top, 0)
            left = int(max(target_width - image_width, 0)/2)
            right = max(target_width - image_width - left, 0)
            # Pad image to crop_size with mean pixel value.

            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=cfg.mean_pixel)

            if label is not None:
                label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=cfg.ignore_label)

            image, label = random_crop(image, label, target_height, target_width)

            label = np.expand_dims(label, axis=-1)

            # cv2.imwrite(os.path.join(SAVE_DIR, "%d_image_aug.jpg" % idx), image)
            # cv2.imwrite(os.path.join(SAVE_DIR, "%d_label_aug.jpg" % idx), label)

            return [image.astype(np.float32), label.astype(np.float32)]
 
        if self.random_crop:
            scale = get_random_scale(cfg.min_scale_factor, cfg.max_scale_factor)
            image, label = randomly_scale_image_and_label(image, label, scale)


            image_shape = image.shape
            image_height = image_shape[0]
            image_width = image_shape[1]

            target_height = image_height + max(cfg.crop_size[0] - image_height, 0)
            target_width = image_width + max(cfg.crop_size[1] - image_width, 0)

            # Pad image with mean pixel value.
            mean_pixel = np.reshape(cfg.mean_pixel, [1, 1, 3])
            image = pad_to_bounding_box(image, 0, 0, target_height, target_width, cfg.mean_pixel)

            if label is not None:
                label = pad_to_bounding_box(
                    label, 0, 0, target_height, target_width, cfg.ignore_label)

            # Randomly crop the image and label.
            if label is not None:
                image, label = random_crop(
                    image, label, cfg.crop_size[0], cfg.crop_size[1])

        if self.flip:
            image, label = flip_dim(image, label, prob=cfg.flip_prob, dim=1)

        aug_img = np.copy(image) if self.save_img else None
        aug_label = np.copy(label) if self.save_img else None

        if self.save_img:
            cv2.imwrite(os.path.join(SAVE_DIR, "%d_image_aug.jpg" % idx), aug_img)
            cv2.imwrite(os.path.join(SAVE_DIR, "%d_label_aug.jpg" % idx), aug_label)

        label = np.expand_dims(label, axis=-1)

        return [image.astype(np.float32), label.astype(np.float32)]

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
    ds = Data('voc_val.txt', shuffle=False, flip=False, random_crop=False, test_set=True, save_img=True)
    # ds = Data('voc_train_sbd_aug.txt', shuffle=False, flip=False, random_crop=False, save_img=True)

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

    ds = AugmentImageComponent(ds, augmentors)
    ds.reset_state()

    g = ds.get_data()
    for i in range(3):
        next(g)

