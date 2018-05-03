from easydict import EasyDict as edict

cfg = edict()

cfg.base_lr = 7e-3
cfg.end_lr = 1e-6

cfg.weight_decay = 1e-4

cfg.crop_size = [513, 513]

cfg.min_scale_factor = 0.5
cfg.max_scale_factor = 2.0
cfg.scale_factor_step_size = 0.25

cfg.multi_grid = [1, 2, 4]
cfg.atrous_rates = [6, 12, 18]

cfg.output_stride = 16
cfg.decode_output_stride = 4

cfg.mean_pixel = [123.68, 116.78, 103.94]   # RGB order
cfg.ignore_label = 255
cfg.flip_prob = 0.5

cfg.num_classes = 21

# cfg.train_list = 'voc_train_sbd_aug.txt'
cfg.train_list = 'voc_train_sbd_aug100.txt'
# cfg.test_list = 'voc_val.txt'
cfg.test_list = 'voc_train_sbd_aug100.txt'

cfg.max_itr_num = 30000

cfg.momentum = 0.9

cfg.learning_power = 0.9

cfg.freeze_batch_norm = False

# colour map
cfg.label_colours = [(0, 0, 0),  # 0=background
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]


cfg.base_architecture = 'resnet_v2_101'
cfg.pre_trained_model = 'resnet_v2_101.ckpt'
