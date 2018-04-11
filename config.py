# from .config_voc_0712 import cfg
# from .config_voc_0712_ssdlite import cfg
# from .config_coco_ssdlite import cfg
#from .config_voc_0712_ssdlite_1 import cfg

# step based learning rate schedule
# cfg.lr_schedule = [(0, 1e-3), (16e4, 1e-4), (20e4, 1e-5)]

# epoch based learning rate schedule
#cfg.lr_schedule = [(0, 1e-3)]


from easydict import EasyDict as edict
import numpy as np
#from .config_utils import *

cfg = edict()

cfg.img_size = 300
cfg.img_w = cfg.img_size
cfg.img_h = cfg.img_size
cfg.label_w = cfg.img_size
cfg.label_h = cfg.img_size

cfg.weight_decay = 1e-4

cfg.classes_name =  ["aeroplane", "bicycle", "bird", "boat",
                     "bottle", "bus", "car", "cat",
                     "chair", "cow", "diningtable", "dog",
                     "horse", "motorbike", "person", "pottedplant",
                     "sheep", "sofa", "train","tvmonitor"]

cfg.class_num = len(cfg.classes_name)

cfg.feat_shapes = [(19, 19), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)]

cfg.anchor_sizes = [[0.15, np.sqrt(0.15 * 0.3)],
                    [0.3, np.sqrt(0.3 * 0.45)],
                    [0.45, np.sqrt(0.45 * 0.6)],
                    [0.6, np.sqrt(0.6 * 0.75)],
                    [0.75, np.sqrt(0.75 * 0.9)],
                    [0.9, np.sqrt(0.9 * 1.05)]]


cfg.prior_scaling = [0.1, 0.1, 0.2, 0.2]

cfg.random_crop = True
cfg.random_expand = True
cfg.random_inter = True

cfg.classes_num = { }
for idx, name in enumerate(cfg.classes_name):
    cfg.classes_num[name] = idx

cfg.train_list = ["reader_test.txt"]
cfg.test_list = ["reader_test.txt"]
cfg.train_sample_num = 0
for train_file in cfg.train_list:
    f = open(train_file, 'r')
    cfg.train_sample_num += len(f.readlines())

cfg.max_gt_box_shown = 30
cfg.det_th = 0.01
cfg.iou_th = 0.5
cfg.neg_iou_th = 0.5
cfg.nms = True
cfg.nms_th = 0.45

cfg.hard_sample_mining = True

cfg.alpha = 1.0
cfg.freeze_backbone = False

cfg.mAP = True

cfg.neg_ratio = 3

cfg.gt_from_xml = True
cfg.gt_format = "voc"
cfg.annopath = 'voc/VOCdevkit/VOC2007/Annotations/{:s}.xml'
cfg.imagesetfile = 'voc/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

cfg.lr_schedule = [(0, 1e-3)]
