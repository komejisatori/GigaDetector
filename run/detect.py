"""
GIGA十亿像素检测系统
检测函数

"""

import os
from sys import path
path.append(r'./')
from api.det_util import detect

if __name__ == '__main__':
    detect()