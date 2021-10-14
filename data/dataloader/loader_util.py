import numpy as np
import os
import json
import cv2
import glob
import random
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
from script.config import *
from data.dataloader.config import dataloader
from data.dataloader.config import SAVE_PATH_CHECK, SAVE_PATH_IMG, SAVE_PATH_ANNO, SAVE_PATH_NAME,SAVE_PATH_XML,\
SAVE_PATH_TRAIN,SAVE_PATH_TEST,SAVE_PATH_IMG_TEST,SAVE_PATH_ANNO_TEST, SAVE_PATH_NAME_TEST

def load_label_file(filepath = dataloader['json_path'], dataset_name = dataloader['test_name'],
                    filename = None, test=False):
    """
    load exist label file
    :param filepath: json annotations file path
    
    :return:
    """
    assert filename is not None
    with open(filepath) as f:
        load_dict = json.load(f)
    label_list = []
    if test:
        img_path = os.path.join(dataloader['img_path']+'/test' ,dataset_name, filename + '.jpg')
    else:
        img_path = os.path.join(dataloader['img_path'], dataset_name, filename + '.jpg')
    if not os.path.exists(img_path):
        print("[dataloader error] test_img not exist")
    label_name = os.path.join(dataset_name, filename + '.jpg')
    image_dict = load_dict[label_name]
    #width = cv2.imread(test_img_path).shape
    #img = cv2.resize(cv2.imread(test_img_path), dataloader['img_shape'])
    for label_dict in image_dict:
        if label_dict['attr']['box_type'] == 'visible body':
            temp_dict = np.asarray(label_dict['rect']) * dataloader['img_shape']
            label_list.append(np.asarray([temp_dict[0,0], temp_dict[0,1], temp_dict[2,0], temp_dict[2,1]]))

    return np.asarray(label_list)


def load_all_names_and_labels_v2(json_path_1, json_path_2, type='visible body'):
    assert json_path_1 is not None
    assert json_path_2 is not None
    DUPLICATE_INDEX = ['1', '2', '3', '4', '5']
    with open(json_path_1) as f:
        load_dict = json.load(f)

    names = []
    labels = []
    for name, label in load_dict.items():
        names.append(name.split('/')[-1])
        label_list = []
        for label_dict in label:
            if label_dict['attr']['box_type'] == type and label_dict['attr']['attr1'] == 'person':
                temp_dict = np.asarray(label_dict['rect']) * dataloader['img_shape']
                label_list.append(np.asarray([temp_dict[0, 0], temp_dict[0, 1], temp_dict[2, 0], temp_dict[2, 1]]))
        labels.append(label_list)

    with open(json_path_2) as f:
        load_dict = json.load(f)

    for name, label in load_dict.items():
        if name.split('_')[1].split('_')[0] in DUPLICATE_INDEX:
            names.append(name.split('/')[-1].split('.')[0] + '_new.jpg')
        else:
            names.append(name.split('/')[-1])
        label_list = []
        for label_dict in label:
            if label_dict['attr']['box_type'] == type and label_dict['attr']['attr1'] == 'person':
                temp_dict = np.asarray(label_dict['rect']) * dataloader['img_shape']
                label_list.append(np.asarray([temp_dict[0, 0], temp_dict[0, 1], temp_dict[2, 0], temp_dict[2, 1]]))
        labels.append(label_list)
    return names, labels


def load_all_names_and_labels(json_path, type='visible body'):
    assert json_path is not None
    with open(json_path) as f:
        load_dict = json.load(f)

    names = []
    labels = []
    for name, label in load_dict.items():
        names.append(name)
        label_list = []
        for label_dict in label:
            if label_dict['attr']['box_type'] == type and label_dict['attr']['attr1'] == 'person':
                temp_dict = np.asarray(label_dict['rect']) * dataloader['img_shape']
                label_list.append(np.asarray([temp_dict[0, 0], temp_dict[0, 1], temp_dict[2, 0], temp_dict[2, 1]]))
        labels.append(label_list)
    return names, labels


def generate_name_list(shuffle=True):
    img_files = glob.glob(os.path.join(SAVE_PATH_IMG, '*.jpg'))
    namelist = []
    with open(SAVE_PATH_NAME, 'w') as f:
        for img_f in img_files:
            namelist.append(img_f.split('/')[-1])
            if shuffle:
                random.shuffle(namelist)
        for img_f in namelist:
            f.writelines(img_f + '\n')

def split_train_test(trainnum=dataloader['train_data_split']):
    with open(SAVE_PATH_NAME, 'r') as f:
        f1 = open(SAVE_PATH_TRAIN, 'w')
        f2 = open(SAVE_PATH_TEST, 'w')
        img_f = f.readline()[:-1]
        count = 0
        while (img_f):
            count += 1
            if count <= trainnum:
                f1.writelines(img_f.split('.')[0] + '\n')
            else:
                f2.writelines(img_f.split('.')[0] + '\n')
            img_f = f.readline()[:-1]
        f1.close()
        f2.close()

def convert_single_pascal_v2(name, bbox, img_size, outpath, CLASS_NAME):
    h = img_size[1]
    w = img_size[0]
    c = 3
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC2007'
    img_name = name
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = img_name
    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = 'GIGA'
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(w)

    node_height = SubElement(node_size, 'height')
    node_height.text = str(h)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(c)

    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'

    # bbox = np.load(os.path.join(SAVE_PATH_ANNO, img_f.split('.')[0] + '.npy'))
    for box in bbox:
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = CLASS_NAME
        node_pose = SubElement(node_object, 'pose')
        node_pose.text = 'Unspecified'

        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(box[0]))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(box[1]))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(box[2]))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(box[3]))
        xml = tostring(node_root, pretty_print=True)
        dom = parseString(xml)
    xml_f = open(os.path.join(outpath, name.split('.')[0] + '.xml'), 'wb')
    xml_f.write(xml)
    xml_f.close()

def convert_single_pascal(name, rename, bbox, outpath, CLASS_NAME='visible_body'):
    IMG_PATH = os.path.join(ROOT, 'data/frame/test/testimg', name)
    img = cv2.imread(IMG_PATH)
    h, w, c = img.shape
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC2007'
    img_name = rename
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = img_name
    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = 'GIGA'
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(w)

    node_height = SubElement(node_size, 'height')
    node_height.text = str(h)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(c)

    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'

    #bbox = np.load(os.path.join(SAVE_PATH_ANNO, img_f.split('.')[0] + '.npy'))
    for box in bbox:
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = CLASS_NAME
        node_pose = SubElement(node_object, 'pose')
        node_pose.text = 'Unspecified'

        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(box[0]))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(box[1]))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(box[2]))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(box[3]))
        xml = tostring(node_root, pretty_print=True)
        dom = parseString(xml)
    xml_f = open(os.path.join(outpath, rename.split('.')[0]+ '.xml'), 'wb')
    xml_f.write(xml)
    xml_f.close()


def convert_to_pascal(CLASS_NAME='visible_body'):
    with open(SAVE_PATH_NAME, 'r') as f:
        img_f = f.readline()[:-1]
        count = 0
        while(img_f):
            count += 1
            print('[xml log] dealing with {}'.format(img_f))
            h, w, c = cv2.imread(os.path.join(SAVE_PATH_IMG,img_f)).shape
            node_root = Element('annotation')
            node_folder = SubElement(node_root, 'folder')
            node_folder.text = 'VOC2007'
            img_name = img_f
            node_filename = SubElement(node_root, 'filename')
            node_filename.text = img_name
            node_source = SubElement(node_root, 'source')
            node_database = SubElement(node_source, 'database')
            node_database.text = 'GIGA'
            node_size = SubElement(node_root, 'size')
            node_width = SubElement(node_size, 'width')
            node_width.text = str(w)

            node_height = SubElement(node_size, 'height')
            node_height.text = str(h)

            node_depth = SubElement(node_size, 'depth')
            node_depth.text = str(c)

            node_segmented = SubElement(node_root, 'segmented')
            node_segmented.text = '0'

            bbox = np.load(os.path.join(SAVE_PATH_ANNO, img_f.split('.')[0] + '.npy'))
            for box in bbox:
                node_object = SubElement(node_root, 'object')
                node_name = SubElement(node_object, 'name')
                node_name.text = CLASS_NAME
                node_pose = SubElement(node_object, 'pose')
                node_pose.text = 'Unspecified'

                node_truncated = SubElement(node_object, 'truncated')
                node_truncated.text = '0'
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'
                node_bndbox = SubElement(node_object, 'bndbox')
                node_xmin = SubElement(node_bndbox, 'xmin')
                node_xmin.text = str(int(box[0]))
                node_ymin = SubElement(node_bndbox, 'ymin')
                node_ymin.text = str(int(box[1]))
                node_xmax = SubElement(node_bndbox, 'xmax')
                node_xmax.text = str(int(box[2]))
                node_ymax = SubElement(node_bndbox, 'ymax')
                node_ymax.text = str(int(box[3]))
                xml = tostring(node_root, pretty_print=True)
                dom = parseString(xml)
            xml_f = open(os.path.join(SAVE_PATH_XML, img_f.split('.')[0] + '.xml'), 'wb')
            xml_f.write(xml)
            xml_f.close()
            img_f = f.readline()[:-1]
        print('[xml log] total num : {}'.format(count))
        #3780 files




def load_single_image():
    """
    return single image and label for map test
    :return:
    """
    #return image, label


if __name__ == '__main__':
    load_all_names_and_labels(json_path=dataloader['json_path'])
