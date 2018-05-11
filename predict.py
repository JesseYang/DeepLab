#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import uuid
import shutil
import ntpath
import numpy as np
from scipy import misc
import argparse
import json
import cv2
import re

# %matplotlib inline
import matplotlib.pyplot as plt
import skimage.io as io
import pylab

from PIL import Image
import tensorflow as tf

from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorpack import *

from preprocess_utils_clean import *
try:
    from .cfgs.config import cfg
except Exception:
    from cfgs.config import cfg

try:
    from .train import DeeplabModel
except Exception:
    from train import DeeplabModel


def get_pred_func(args):
    sess_init = SaverRestore(args.model_path)
    model = DeeplabModel() 
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["input"],
                                   output_names=["predicts"])

    predict_func = OfflinePredictor(predict_config)
    return predict_func

# def draw_result(image, boxes):
#    return image_result
def decode_labels(mask, num_classes=21):
    """Decode batch of segmentation masks.
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    mask = mask[0]
    n, h, w = mask.shape
    outputs = np.zeros((h, w, 3), dtype=np.uint8)
    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for j_, j in enumerate(mask[0, :, :]):
        for k_, k in enumerate(j):
            if k < num_classes:
                pixels[k_, j_] = cfg.label_colours[k]
    outputs = np.array(img)
    return outputs

def predict_image(input_path, output_path, predict_func, args):
    image = cv2.imread(input_path)
    # image = cv2.resize(cvt_clr_image, (cfg.crop_size[0], cfg.crop_size[1]))
    if args.multi_scale_input:
        scale = get_random_scale(cfg.min_scale_factor, cfg.max_scale_factor, 0)
        image, _ = randomly_scale_image_and_label(image, scale=scale)
    if args.flip:
        image, _ = flip_dim(image) 
    image = np.expand_dims(ori_image, axis=0)
    prediction = predict_func(image)

    image_result = decode_labels(prediction, cfg.num_classes)
    save_image = cv2.cvtColor(image_result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, save_image)


def generate_pred_images(image_paths, predict_func, output_dir, args):
   
    for image_idx, image_path in enumerate(image_paths):

        if not os.path.exists(image_path):
            continue
        if image_idx % 100 == 0 and image_idx > 0:
            print(str(image_idx))
        # print(image_path)
        img_name = image_path.split('/')[-1].split('.')[0]
        image = cv2.imread(image_path)

        # image = cv2.resize(cvt_color_image, (cfg.img_w, cfg.img_h))
        if args.multi_scale_input:
            scale = get_random_scale(cfg.min_scale_factor, cfg.max_scale_factor, 0)
            image, _ = randomly_scale_image_and_label(image, scale=scale)
        if args.flip:
            image, _ = flip_dim(image) 
        image = np.expand_dims(image, axis=0)
        prediction = predict_func(image)

        image_result = decode_labels(prediction, cfg.num_classes)

        save_path = os.path.join(output_dir, image_path.split('/')[-1])
        save_image = cv2.cvtColor(image_result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--backbone', help='the backbone network', default='mobilenetv2')
    parser.add_argument('--model_path', help='path of the model waiting for validation.')
    parser.add_argument('--data_format', choices=['NCHW', 'NHWC'], default='NHWC')
    parser.add_argument('--input_path', help='path of the input image')
    parser.add_argument('--output_path', help='path of the predictive output image', default='output.png')
    parser.add_argument('--test_list', help='list of the test files', default=None)
    parser.add_argument('--output_dir', help='directory to save image result', default='output')
    # parser.add_argument('--pred_txt_dir', help='directory to save txt result', default='result_pred')
    # parser.add_argument('--gen_image', action='store_true')
    parser.add_argument('--multi_scale_input', help='whether scale input images', default=False)
    parser.add_argument('--flip', help='whether flip input images', default=False)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    predict_func = get_pred_func(args)
    
    new_dir = args.output_dir # if args.gen_image else args.pred_txt_dir

    if os.path.isdir(new_dir):
        shutil.rmtree(new_dir)
    os.mkdir(new_dir)

    if args.input_path != None:
        # predict one image (given the input image path) and save the result image
        predict_image(args.input_path, args.output_path, predict_func, args) 
    elif args.test_list != None:
        test_paths = args.test_list.split(',')
        image_paths = []
        for test_path in test_paths:
            with open(test_path) as f:
                content = f.readlines()
            image_paths.extend([line.split(' ')[0].strip() for line in content])
                
        print("Number of images to predict: " + str(len(image_paths)))
        generate_pred_images(image_paths, predict_func, args.output_dir, args)
