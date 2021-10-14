from ..video.config import video
from ..image.config import image
from script.config import *
import os

dataloader = {
    'name': '1-HIT_Cateen',
    'json_path': os.path.join(ROOT, 'data/label/gt_human_bbox_all.json'),
    'img_shape': image['resize_shape'],
    'test_name': '1-HIT_Canteen',
    'img_path': video['frame_path'],
    'train_data_split': 3000
}

SAVE_PATH_IMG = os.path.join(ROOT, 'data/dataset/jpg')
SAVE_PATH_ANNO = os.path.join(ROOT, 'data/dataset/anno')
SAVE_PATH_IMG_TEST = os.path.join(ROOT, 'data/dataset/jpg_test')
SAVE_PATH_ANNO_TEST = os.path.join(ROOT, 'data/dataset/anno_test')
SAVE_PATH_CHECK = os.path.join(ROOT, 'data/dataset/check')
SAVE_PATH_NAME = os.path.join(ROOT, 'data/dataset/name.txt')
SAVE_PATH_NAME_TEST = os.path.join(ROOT, 'data/dataset/name_test.txt')
SAVE_PATH_XML = os.path.join(ROOT, 'data/dataset/xml')
SAVE_PATH_TRAIN = os.path.join(ROOT, 'data/dataset/train.txt')
SAVE_PATH_TEST = os.path.join(ROOT, 'data/dataset/test.txt')