#detector config

import os
from script.config import ROOT

faster_rcnn = {
    'cfg_path': os.path.join(ROOT, 'train/train_configs/faster_rcnn_101_rpn.py'),
    'url': os.path.join(ROOT,'data/weights/faster_rcnn_r101_fpn_2x_20181129-73e7ade7.pth'),
#'data/weights/faster_rcnn_r101_fpn_2x_20181129-73e7ade7.pth'),
}

hrnet = {
    'cfg_path':os.path.join(ROOT, 'train/train_configs/hrnet_v2p_w48.py'),
    'url':os.path.join(ROOT, 'train/work_dirs/faster_rcnn_hrnetv2p_w32_2x/epoch_1.pth')
}

ssd512 = {
    'cfg_path':os.path.join(ROOT, 'train/train_configs/ssd_512.py'),
    'url': os.path.join(ROOT, 'train/work_dirs/ssd512_coco/latest.pth')
}

retinanet = {
    'cfg_path': os.path.join(ROOT, 'mmdetection/configs/retinanet_x101_64x4d_fpn_1x.py'),
    'url':  'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/'+
            'retinanet_x101_64x4d_fpn_1x_20181218-a0a22662.pth',
}

cascade_rcnn = {
    'cfg_path': os.path.join(ROOT,'train/train_configs/cascade_rcnn_r101_fpn_1x.py'),
    'url':''
}

detector = {
    'name': 'faster_rcnn', # ssd300 for mobilenetv2
    'trained_path': os.path.join(ROOT, 'data/weights/ssd300_vgg.pth'), # ssd300 for mobilenetv2
    'cuda': True,
    'name_list': ['faster_rcnn', 'cascade_rcnn', 'retinanet', 'ssd512', 'hrnet'],
    'cfg_list': [faster_rcnn, cascade_rcnn, retinanet, ssd512, hrnet],
    'class_list': 'full_body', #visible_body #full_body
    'threshold': 0.01
}

ssd = {
    'mean': (104, 117, 123),
    'shape': (1333, 800)
}


