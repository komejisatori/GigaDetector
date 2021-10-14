from sys import path
path.append(r'./')
from script.config import *
import os
import glob
import cv2
from data.image.config import image as img_cfg
from script.loader_test import TEST_PATH

TEST_RESULT_PATH = os.path.join(ROOT, 'script/results/testresult/resultsimgs')

if __name__ == "__main__":
    if not os.path.exists(TEST_RESULT_PATH):
        os.mkdir(TEST_RESULT_PATH)
    img_files = glob.glob(os.path.join(TEST_PATH, '*.jpg'))
    for x in img_files:
        #import pdb
        #pdb.set_trace()
        img = cv2.imread(x)
        img = cv2.resize(img, img_cfg['resize_shape'])
        cv2.imwrite(os.path.join(TEST_RESULT_PATH, x.split('/')[-1]), img)
