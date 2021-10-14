from sys import path
path.append(r'./')
import os
import matplotlib.pyplot as plt
from data.cocoapi.PythonAPI.pycocotools.coco import COCO
from data.cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
from script.config import *

def eval(annFile, resFile):
    annType = ['segm','bbox','keypoints']
    annType = annType[1]      #specify type here
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'

    cocoGt=COCO(annFile)
    cocoDt=cocoGt.loadRes(resFile)

    imgIds=sorted(cocoGt.getImgIds())


    cocoEval = COCOeval(cocoGt,cocoDt,annType)

    cocoEval.params.imgIds  = imgIds
    cocoEval.params.maxDets = [1,10,500]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()