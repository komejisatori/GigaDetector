"""
GIGA十亿像素检测系统
训练函数

"""
import os
from sys import path
path.append(r'./')
from api.det_util import detect
import run.config as run_cfg
from train.train_script_cascade_rcnn import train_cascadercnn
from train.train_script_faster_rcnn import train_fasterrcnn
from train.train_script_retinanet import train_retinanet


def train_total():
    if run_cfg.TRAIN_MODEL_NAME == 'faster_rcnn':#faster_rcnn cascade_rcnn retina_net
        train_fasterrcnn()
    if run_cfg.TRAIN_MODEL_NAME == 'cascade_rcnn':
        train_cascadercnn()
    if run_cfg.TRAIN_MODEL_NAME == 'retina_net':
        train_retinanet()