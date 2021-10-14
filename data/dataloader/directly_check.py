"""
this file is used for directly check dataset labels
"""
from sys import path
path.append(r'../')
from data.dataloader.config import dataloader as loader_cfg
from data.dataloader.crop_images import *

if __name__ == '__main__':
    directly_check()