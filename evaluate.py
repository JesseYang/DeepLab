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
from tqdm import tqdm
import tensorflow as tf

from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorpack import *
from tensorpack.utils import logger

from cfgs.config import cfg
from train_slim import DeeplabModel
# from train import DeeplabModel

def get_pred_func(args):
    sess_init = SaverRestore(args.model_path)
    model = DeeplabModel(depth=101)
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["input", "label"],
                                   output_names=["softmax_cross_entropy_loss/value", "miou_update_op"])

    predict_func = OfflinePredictor(predict_config) 
    return predict_func

def evaluate(test_path, predict_func):
    '''
    op_list = predict_func.sess.graph.get_operations()
    op_names = [e.name for e in op_list]
    target_names = [e for e in op_names if "miou" in e]
    '''

    f = open(test_path, 'r')
    lines = f.readlines()
    cost_list = []
    for line in tqdm(lines):
        path_ary = line.split(' ')
        img_path = path_ary[0].strip()
        label_path = path_ary[1].strip()

        image = cv2.imread(img_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = np.expand_dims(label, axis=-1)

        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        cost, _ = predict_func(image, label)
        cost_list.append(cost)

    miou_tensor = predict_func.sess.graph.get_tensor_by_name('miou:0')
    miou = predict_func.sess.run(miou_tensor)
    logger.info("miou is: %.2f" % (miou * 100))
    logger.info("cost is: %.2f" % np.mean(cost_list))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of the model waiting for validation.')
    parser.add_argument('--test_path', help='path of the test file', default="voc_val.txt")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    predict_func = get_pred_func(args)

    evaluate(args.test_path, predict_func)
