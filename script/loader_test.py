from sys import path
path.append(r'./')
from data.dataloader.loader_util import load_label_file

from detector.detector_entity import Detector
from data.image.image_entity import Image
from train.train_util import cal_ap, draw_fppi_mr
from script.config import *
import os
import glob
import json
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

TEST_PATH = os.path.join(ROOT, 'data/frame/test/*')
# TEST_PATH = os.path.join(ROOT, 'data/frame/[0-9]*')
TEST_SAVE_PATH = os.path.join(ROOT, 'script/results/testresult')
TEST_RESULT_PATH = os.path.join(ROOT, 'script/results/testresult/results')
DIRECTLY_LOAD = False

if __name__ ==  '__main__':
    if not os.path.exists(TEST_RESULT_PATH):
        os.mkdir(TEST_RESULT_PATH)
    img_files = glob.glob(os.path.join(TEST_PATH, '*.jpg'))
    names = [x.split('/')[-2]+'/'+x.split('/')[-1] for x in img_files]
    detector = Detector(is_pretrained=True)
    detector.load_detector()
    annos = []
    results = {}

    if not DIRECTLY_LOAD:

        for j in names:
            #anno = load_label_file(dataset_name=j.split('/')[0], filename=j.split('/')[1].split('.')[0], test=True)
            image = Image(dataset_name='test/'+j.split('/')[0] ,frame_name=j.split('/')[1].split('.')[0])
            #image = Image(dataset_name=j.split('/')[0], frame_name=j.split('/')[1].split('.')[0])
            image.cv2_resize()
            image.cv2_toNumpy()
            #image.normalize()
            #image.hwc2chw()
            image = detector.detect_all(image)

            #annos.append(anno)
            results[image.name] = image.result

            with open(os.path.join(TEST_RESULT_PATH, image.name + '.txt'), 'w') as f:
                for r in image.result:
                    f.writelines('visible_body ')
                    f.writelines(str(r[4]) + ' ')
                    f.writelines(str(int(r[0])) + ' ')
                    f.writelines(str(int(r[1])) + ' ')
                    f.writelines(str(int(r[2])) + ' ')
                    f.writelines(str(int(r[3])) + '\n')

        with open(os.path.join(TEST_SAVE_PATH, 'result.pkl', 'wb'))as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
        #np.save(os.path.join(TEST_SAVE_PATH, 'anno'), np.asarray(annos))

    # else:
        # annos_ori = np.load(os.path.join(TEST_SAVE_PATH, 'anno.npy'), allow_pickle=True)
        # annos = [x for x in annos_ori]

    #draw_fppi_mr(results, annos, filenames=[dataset_name + '/'+ i for i in jpg_names])



