# generate detection result and save in the format of MOTChallenge
import os
from sys import path
path.append(r'./')
import detector.config as det_cfg
import train.train_configs.faster_rcnn_101_rpn as fstrcn_cfg
import train.train_configs.cascade_rcnn_r101_fpn_1x as cscdrcn_cfg
import train.train_configs.retinanet_101_FPN as rtna_cfg
from detector.detector_entity_api import Detector
from data.image.image_api_entity import Image_api
from data.dataloader.loader_util import load_all_names_and_labels_v2
from data.dataloader.coco_converter import result_to_coco_v2, convert_to_coco
from script.cocoeval import eval
import run.config as run_cfg
os.environ["CUDA_VISIBLE_DEVICES"] = run_cfg.GPU_ID
PRETRAIN_MODEL = False # True from coco pretrain model, False for finetune model
PREFIX = run_cfg.ROOT
DETECTOR_TYPE = run_cfg.MODEL_NAME #faster_rcnn cascade_rcnn retina_net
DRAWRESULT = run_cfg.DRAW_RESULT
MODEL_CFG_PRETRAIN = {
    'faster_rcnn' : PREFIX + '/train/train_configs/faster_rcnn_101_rpn_pre.py',
    'cascade_rcnn': PREFIX + '/train/train_configs/cascade_rcnn_r101_fpn_1x_pre.py',
    'retinanet': PREFIX + '/train/train_configs/retinanet_101_FPN_pre.py'
}
MODEL_CFG_FINETUNE = {
    'faster_rcnn': PREFIX + '/train/train_configs/faster_rcnn_101_rpn.py',
    'cascade_rcnn': PREFIX + '/train/train_configs/cascade_rcnn_r101_fpn_1x.py',
    'retinanet': PREFIX + '/train/train_configs/retinanet_101_FPN.py'
}

MODEL_PATH_PRETRAIN = {
    'faster_rcnn': PREFIX + '/data/weights/faster_rcnn_r101_fpn_2x_20181129-73e7ade7.pth',
    'cascade_rcnn':PREFIX + '/data/weights/cascade_rcnn_r101_fpn_20e_20181129-b46dcede.pth',
    'retinanet': PREFIX + '/data/weights/retinanet_r101_fpn_2x_20181129-72c14526.pth'
}
MODEL_PATH_FINETUNE = {
    'faster_rcnn': PREFIX + '/train/work_dirs/faster_rcnn_r101_fpn_1x_newdataset/epoch_10.pth',
    'cascade_rcnn': PREFIX + '/train/work_dirs/cascade_rcnn_r101_fpn_1x_newdataset/epoch_10.pth',
    'retinanet': PREFIX + '/train/work_dirs/retinanet_r101_fpn_1x_newdataset/epoch_10.pth'
}
JSON_PATH = [
    PREFIX + '/data/label/gt_human_bbox_all.json',
    PREFIX + '/data/label/gt_human_bbox_all_part2_v2.json'
]
TEST_LIST_PATH ='/home/wangzerun/giga/api/test_list.txt'
TEST_LIST = []
EVAL_LIST = []
with open(TEST_LIST_PATH, 'r') as f:
    while True:
        lines = f.readline()
        if not lines:
            break
        TEST_LIST.append(PREFIX + '/data/frame/test/' + lines.split('\n')[0])
        EVAL_LIST.append(lines.split('\n')[0])

def evaluate_det_result(prediction_txt_path, gt_json_path_1, gt_json_path_2, img_files_list):
    """
    :param prediction_txt_path: path to detection result txt file (det.txt)
    :param gt_json_path: path to ground-truth json file
    :param img_files_list: detected giga image filename list, e.g. ["IMG_1_1.jpg", "IMG_1_2.jpg", "IMG_1_3.jpg", ...]
    :return: evaluation results
    """
    #import pdb
    #pdb.set_trace()
    names, labels = load_all_names_and_labels_v2(gt_json_path_1, gt_json_path_2, type='visible body')
    result_to_coco_v2(prediction_txt_path, output=os.path.join(PREFIX, 'api/result'))
    eval_dict = {}
    eval_names = []
    eval_labels = []
    for name, label in zip(names, labels):
        if name in img_files_list:
            eval_dict[name] = label
    for name in img_files_list:
        eval_names.append(name)
        eval_labels.append(eval_dict[name])
    # make index in order

    convert_to_coco(eval_names, eval_labels, type="visible_body", outpath=os.path.join(PREFIX, 'api/anno'))
    eval(os.path.join(PREFIX, 'api/anno/anno.json'), os.path.join(PREFIX, 'api/result/output_result.json'))


def det_giga_full(img_root_dir, detector_type, conf_thres):
    """
    :param img_root_dir: the directory path which contains giga image named as {0001.jpg, 0002.jpg, ...}
    :param detector_type: can be 'faster_rcnn' or 'cascade_rcnn' or 'retina_net'
    :param conf_thres: the threshold of detection results' confidence, only keep results > threshold
    """
    # set cfg file
    det_types = ['faster_rcnn', 'cascade_rcnn', 'retina_net']
    det_cfg.detector['name'] = det_cfg.detector['name_list'][det_types.index(detector_type)]
    det_cfg.detector['threshold'] = conf_thres
    if PRETRAIN_MODEL:
        det_cfg.faster_rcnn['cfg_path'] = MODEL_CFG_PRETRAIN['faster_rcnn']
        det_cfg.cascade_rcnn['cfg_path'] = MODEL_CFG_PRETRAIN['cascade_rcnn']
        det_cfg.retinanet['cfg_path'] = MODEL_CFG_PRETRAIN['retinanet']
        det_cfg.detector['class_list'] = 'person'
        det_cfg.faster_rcnn['url'] = MODEL_PATH_PRETRAIN['faster_rcnn']
        det_cfg.cascade_rcnn['url'] = MODEL_PATH_PRETRAIN['cascade_rcnn']
        det_cfg.retinanet['url'] = MODEL_PATH_PRETRAIN['retinanet']
    else:
        det_cfg.faster_rcnn['cfg_path'] = MODEL_CFG_FINETUNE['faster_rcnn']
        det_cfg.cascade_rcnn['cfg_path'] = MODEL_CFG_FINETUNE['cascade_rcnn']
        det_cfg.retinanet['cfg_path'] = MODEL_CFG_FINETUNE['retinanet']
        det_cfg.detector['class_list'] = 'visible_body'
        det_cfg.faster_rcnn['url'] = MODEL_PATH_FINETUNE['faster_rcnn']
        det_cfg.cascade_rcnn['url'] = MODEL_PATH_FINETUNE['cascade_rcnn']
        det_cfg.retinanet['url'] = MODEL_PATH_FINETUNE['retinanet']
    
    detector = Detector(is_pretrained=True,detector_cfg = det_cfg)
    detector.load_detector()

    # firstly sort the image file by name
    """
    filenames = []
    for root, dirs, files in os.walk(full_path):
        for file in files:
            if file[-3:] == 'jpg':
                filenames.append(file)
    filenames.sort()
    """


    filenames = TEST_LIST
    img_root_dir = './'
    if PRETRAIN_MODEL:
        file_prefix = '_pre'
    else:
        file_prefix = ''
    # then detect on each image by sequence and save result to txt in MOT format
    detfilename = 'det22' +'_'+DETECTOR_TYPE + file_prefix + '.txt'
    with open(img_root_dir + detfilename, 'w') as f:
        for img_id, img_name in enumerate(filenames):
            temp_img_path = img_name
            image = Image_api(img_path=temp_img_path)
            image.cv2_resize()
            image.cv2_toNumpy()
            image = detector.detect_all(image)
            if DRAWRESULT:
                image.draw()
            #det_bboxes, det_labels = detect_func(temp_img_path, detector_type, ...)
            # here detect on image, do nms and return bboxes and labels (person's label is 0)
            # det_bboxes: [[x_tl, y_tl, x_br, y_br, confidence], ...]
            for r in image.result:
                x_tl = r[0]
                x_br = r[2]
                y_tl = r[1]
                y_br = r[3]
                conf = r[4]
                w = int(x_br) - int(x_tl)
                h = int(y_br) - int(y_tl)
                f.writelines([str(img_id + 1), ',', '-1', ',', str(x_tl), ',', str(y_tl), ',',
                              str(w), ',', str(h), ',', str(conf), ',', '-1', ',', '-1', ',', '-1','\n'])
            """
            for i, item in enumerate(det_bboxes):
                bbox = item.tolist()
                if det_labels[i] == 0:  # person
                    x_tl = bbox[0]
                    x_br = bbox[2]
                    y_tl = bbox[1]
                    y_br = bbox[3]
                    conf = bbox[4]

                    w = x_br - x_tl
                    h = y_br - y_tl

                    if conf > conf_thres:
                        f.writelines([str(img_id + 1), ',', '-1', ',', str(x_tl), ',', str(y_tl), ',',
                                      str(w), ',', str(h), ',', str(conf), ',', '-1', ',', '-1', ',', '-1', '\n'])
            """
    return filenames
# just for test

def detect():
    det_giga_full(img_root_dir='./', detector_type=DETECTOR_TYPE, conf_thres=0.01)

def evaluate():
    img_files_list = EVAL_LIST
    if PRETRAIN_MODEL:
        file_prefix = '_pre'
    else:
        file_prefix = ''
    evaluate_det_result(PREFIX + '/det_' + DETECTOR_TYPE + file_prefix + '.txt', JSON_PATH[0], JSON_PATH[1],
                        img_files_list)


if __name__ == '__main__':
    det_giga_full(img_root_dir='./', detector_type=DETECTOR_TYPE, conf_thres=0.01)

    img_files_list = EVAL_LIST
    if PRETRAIN_MODEL:
        file_prefix = '_pre'
    else:
        file_prefix = ''
    evaluate_det_result('/home/wangzerun/giga/det_' + DETECTOR_TYPE + file_prefix + '.txt', JSON_PATH[0], JSON_PATH[1], img_files_list)
