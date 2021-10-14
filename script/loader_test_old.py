from sys import path
path.append(r'./')
from data.dataloader.loader_util import load_label_file

from detector.detector_entity import Detector
from data.image.image_entity import Image
from train.train_util import cal_ap, draw_fppi_mr
from script.config import *
import os
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np



if __name__ ==  '__main__':
    TEST_PATH = os.path.join(ROOT, 'data/frame/test/*')
    #TEST_PATH = os.path.join(ROOT, 'data/frame/[0-9]*')
    TEST_SAVE_PATH = os.path.join(ROOT, 'script/results/testresult')
    DIRECTLY_LOAD = False
    img_files = glob.glob(os.path.join(TEST_PATH, '*.jpg'))
    names = [x.split('/')[-2]+'/'+x.split('/')[-1] for x in img_files]
    detector = Detector(is_pretrained=True)
    detector.load_detector()
    annos = []
    results = None
    count = 0

    if not DIRECTLY_LOAD:
        for j in names:
            anno = load_label_file(dataset_name=j.split('/')[0], filename=j.split('/')[1].split('.')[0])
            image = Image(dataset_name='test/'+j.split('/')[0] ,frame_name=j.split('/')[1].split('.')[0])
            #image = Image(dataset_name=j.split('/')[0], frame_name=j.split('/')[1].split('.')[0])
            image.cv2_resize()
            image.cv2_toNumpy()
            #image.normalize()
            #image.hwc2chw()
            image = detector.detect_all(image)
            index = np.ones((image.result.shape[0], 1)) * count
            result = np.concatenate((index, image.result), axis=1)

            #index = np.ones((anno.shape(0), 1)) * count
            #anno = np.concatenate((index, anno), axis=1)
            if results is None:
                #annos = anno
                results = result
            else:
                #annos = np.concatenate((annos, anno))
                results = np.concatenate((results, result))
            annos.append(anno)
            count += 1

        #image.draw()
        np.save(os.path.join(TEST_SAVE_PATH, 'result'), results)
        np.save(os.path.join(TEST_SAVE_PATH, 'anno'), np.asarray(annos))

    else:
        results = np.load(os.path.join(TEST_SAVE_PATH, 'result.npy'))
        annos_ori = np.load(os.path.join(TEST_SAVE_PATH, 'anno.npy'), allow_pickle=True)
        annos = [x for x in annos_ori]

    _, _, ap = cal_ap(results, annos, count+1)
    print('ap {}'.format(ap))