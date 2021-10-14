"""
crop images in train dataset into specific w*h and adjust bounding boxes
"""
from sys import path
path.append(r'../')
import os
import cv2
import math
import mmcv
import glob
import numpy as np
from data.dataloader.config import dataloader as loader_cfg
from data.dataloader.config import SAVE_PATH_ANNO, SAVE_PATH_IMG, SAVE_PATH_CHECK
from data.image.config import image as img_cfg
from detector.config import ssd as det_cfg
from data.dataloader.loader_util import load_label_file, load_all_names_and_labels, generate_name_list, convert_to_pascal,\
    split_train_test
from data.image.image_entity import Image
from script.config import *

WIDTH = 1333
HEIGHT = 800


def directly_check(json_path=loader_cfg['json_path']):
    TEST_PATH = 'data/check'
    names, labels = load_all_names_and_labels(json_path)
    for name, label in zip(names, labels):
        print('[croper log] dealing with {}'.format(name))
        image = Image(dataset_name=name.split('/')[0], frame_name=name.split('/')[1].split('.')[0])
        if image.image is None:
            print('[croper log] {} not found maybe a test img'.format(name))
            continue
        image.cv2_resize(for_ms=False)
        image.result = label
        path = os.path.join(ROOT, TEST_PATH, name.split('/')[1])
        image.draw(path=path)


def crop_images(json_path=loader_cfg['json_path'], image_path=loader_cfg['img_path']):
    slide = img_cfg['slide_stride'] # store origin strides
    img_cfg['slide_stride'] = 1
    names, labels = load_all_names_and_labels(json_path)
    for name, label in zip(names, labels):
        print('[croper log] dealing with {}'.format(name))
        image = Image(dataset_name=name.split('/')[0], frame_name=name.split('/')[1].split('.')[0])
        if image.image is None:
            print('[croper log] {} not found maybe a test img'.format(name))
            continue
        image.cv2_resize(for_ms=False)
        count = 0
        for img in image.sample():
            x1, y1 = image.cur_pos()
            img_box = (x1, y1, x1+det_cfg['shape'][0], y1+det_cfg['shape'][1])
            l, t, b, r, save_labels = check_labels(img_box, label)
           #  print(len(label))
            save_img = image.expand(l = l, t = t, b= b, r=r)
            save_labels = np.asarray(save_labels)
            if save_labels.shape[0] > 0:
                mmcv.imwrite(save_img, os.path.join(SAVE_PATH_IMG, name.split('/')[1].split('.')[0]+'_{}.jpg'.format(count)))
                np.save(os.path.join(SAVE_PATH_ANNO, name.split('/')[1].split('.')[0]+'_{}'.format(count)), save_labels)
            count += 1
        

    img_cfg['slide_stride'] = slide

def check_labels(img_box, label_boxes):
    box_temp = label_boxes.copy()
    label_list = []
    l = 0
    t = 0
    b = 0
    r = 0
    continue_ = 0
    while continue_ is not -1:
        continue_, l, t, b, r, reval, box_temp = check_labels_once(img_box, box_temp, l, t, b, r)
        if continue_ == -1:
            for label in label_list:
                label[0] -= (img_box[0] - l)
                label[1] -= (img_box[1] - t)
                label[2] -= (img_box[0] - l)
                label[3] -= (img_box[1] - t)
            return l, t, b, r, label_list
        else:
            label_list.append(reval)

def check_labels_once(img_box, label_boxes, l, t, b, r):
    for i in range(len(label_boxes)):
        count = 0
        origin_l = l
        origin_t = t
        origin_b = b
        origin_r = r
        l_, t_, b_, r_, label_ = box_in(img_box, label_boxes[i], origin_l, origin_t, origin_b, origin_r)
        if l_ is not -1:
            l = max(l, l_)
            t = max(t, t_)
            b = max(b, b_)
            r = max(r, r_)
            reval = label_boxes[i].copy()
            label_boxes.pop(i)
            count += 1
            return count, l ,t, b, r, reval, label_boxes
    return -1, l, t, b, r, -1, label_boxes



def box_in(img_box, label_box, origin_l, origin_t, origin_b, origin_r):
    l = 0
    t = 0
    b = 0
    r = 0
    img_x1 = img_box[0] - origin_l
    img_y1 = img_box[1] - origin_t
    img_x2 = img_box[2] + origin_r
    img_y2 = img_box[3] + origin_b

    if label_box[3] > img_y1 and label_box[1] < img_y2 \
        and label_box[2] > img_x1 and label_box[0] < img_x2:
        if label_box[0] < img_box[0]:
            l = math.ceil(img_box[0] - label_box[0])
        if label_box[1] < img_box[1]:
            t = math.ceil(img_box[1] - label_box[1])
        if label_box[2] > img_box[2]:
            r = math.ceil(label_box[2] - img_box[2])
        if label_box[3] > img_box[3]:
            b = math.ceil(label_box[3] - img_box[3])
        label_x1 = label_box[0]
        label_y1 = label_box[1]
        label_x2 = label_box[2]
        label_y2 = label_box[3]
        return l, t, b, r, [label_x1, label_y1, label_x2, label_y2]
    else:
        return -1, -1, -1, -1, -1


def crop_checker():
    img_files = glob.glob(os.path.join(SAVE_PATH_IMG, '*.jpg'))
    for img_f in img_files:
        print('[check log] dealing with {}'.format(img_f.split('/')[-1].split('.')[0]))
        img = cv2.imread(img_f)
        bbox = np.load(os.path.join(SAVE_PATH_ANNO, img_f.split('/')[-1].split('.')[0] + '.npy'))
        if bbox.shape[0] > 0:
            for box in bbox:
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
        cv2.imwrite(os.path.join(SAVE_PATH_CHECK, img_f.split('/')[-1].split('.')[0] + '.jpg'), img)


def make_dataset():
    directly_check()
    crop_images()
    crop_checker()
    generate_name_list()
    convert_to_pascal()
    split_train_test()

if __name__ == '__main__':
    # directly_check()
    # crop_images()
    # crop_checker()
    # generate_name_list()
    convert_to_pascal()
    #split_train_test()




