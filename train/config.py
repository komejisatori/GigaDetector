from script.config import *
train_cfg = {
    'faster-rcnn': './train_configs/faster_rcnn.py',
    'faster-rcnn_101_rpn': './train_configs/faster_rcnn_101_rpn.py',
    'faster-rcnn_hrnetv2p_w48': './train_configs/hrnet_v2p_w48.py',
    'ssd-512': './train_configs/ssd_512.py'
}

eval = {
    'ovthresh': 0.5,
    'use_07_metric': True,
    'save_path': ROOT
}

