"""
GIGA十亿像素检测系统
数据集制作工具

"""
import os
from sys import path
path.append(r'./')
from api.det_util import detect
from data.dataloader.crop_images import make_dataset

if __name__ == '__main__':
    make_dataset()