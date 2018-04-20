from easydict import EasyDict as edict

cfg = edict()

cfg.base_lr = 7e-3
cfg.end_lr = 1e-6

cfg.weight_decay = 5e-4

cfg.crop_size = [513, 513]

cfg.min_scale_factor = 0.5
cfg.max_scale_factor = 2.0
cfg.scale_factor_step_size = 0.25

cfg.multi_grid = [1, 2, 4]
cfg.atrous_rates = [6, 12, 18]

cfg.output_stride = 16
cfg.decode_output_stride = 4

cfg.mean_pixel = [127.5, 127.5, 127.5]
cfg.ignore_label = 255
cfg.flip_prob = 0.5

cfg.num_classes = 21

cfg.train_list = 'voc_train_sbd_aug100.txt'
cfg.test_list = 'voc_val.txt'

cfg.max_itr_num = 30000

cfg.momentum = 0.9

cfg.learning_power = 0.9

cfg.freeze_batch_norm = False
