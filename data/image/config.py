# image config
from data.video.config import video
from script.config import *
import os

image = {
    'resize_shape': (7998, 4800),
    'frame_path': os.path.join(ROOT,'data/frame'),
    'slide_stride': 0.2, # normalized,
    ## 添加了一些筛选框的算法对应参数
    'h/w_threshold': 1.5,
    'min_offset_threshold': 5,
    'multi_scale': True,
    'ms_resize_scale': (2666*2, 800*2),
    'nms': True
}
