# Giga Detector

## introduction

a detector for large frames

## installation

- python 3.6
- opencv-python
- pytorch 1.0.1
- matlab R2019
- mmdetection

## BEFORE USING
- you should install matlabR2019 first for drawing LAMR curves
https://blog.csdn.net/elgong/article/details/82931416
- you should clone mAP and voc2coco repo to [TODO]

## TRAIN
 Giga detector uses mmdetection provided models for detection.
 For training new models, you should:
 - make a VOC dataset (make sure the old jpgs and xmls has been cleaned)
    - crop images [TODO]
    - put annos(npy files) jpgs(jpg files) to [ROOT]/data/dataset/[anno & jpg]
    - put name.txt to [ROOT]/data/dataset
    - run [ROOT]/data/dataloader/crop_images.py, this will generate xml files for VOC dataset
    - run generate_namelist/py, this will generate train.txt for training
    - put train.txt from [ROOT]/data/dataset/train.txt to [ROOT]/data/dataset/VOCdevkit/VOC2007/ImageSets/Main
    - put xml files from [ROOT]/data/dataset/xml to [ROOT]/data/dataset/VOCdevkit/VOC2007/Annotations
    - put jpg files from [ROOT]/data/dataset/jpg to [ROOT]/data/dataset/VOCdevkit/VOC2007/JPEGImages
    
 - provide mmdetection training scripts 
    - scripts are in [ROOT]/train/train_configs, we use 
        - faster_rcnn_101_rpn.py
        - hrnet_v2p_w48.py
        - retinanet_101_FPN.py
    - make sure the class numbers is 2, you can search the word 'num_classes' in each script
    - make sure the dataset path is correct
 - provide a train script to [ROOT]/train
    -
    - we use scripts for 3 models above, 1 and 2 are provided, try to make the retinanet script
    - set a specific gpu by add os.environ["CUDA_VISIBLE_DEVICES"] = [GPU_ID]
    - run the train script, TMUX suggested by 'tmux new -s giga_train'

## dataset

## run





