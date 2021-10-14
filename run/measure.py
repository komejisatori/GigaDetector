"""
GIGA十亿像素检测系统
测试工具

"""

import os
from sys import path
path.append(r'./')
from api.det_util import evaluate
from api.convert_det_file import read_det_file
from script.get_tn import run_fp_analyse
MEASUREMENT = 'AP' #fp_analyse fppi_curve

if __name__ == '__main__':
    if MEASUREMENT == 'AP':
        evaluate()
    if MEASUREMENT == 'fp_analyse':
        run_fp_analyse()
    if MEASUREMENT == 'fppi_curve':
        read_det_file()
        print('[measure] please run draw_fppi.m')


