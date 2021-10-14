from sys import path
path.append(r'./')
from script.config import *
import os
import glob
import json
from data.dataloader.loader_util import load_all_names_and_labels, convert_single_pascal, convert_single_pascal_v2
from data.voc2coco.voc2coco import convert

RESULT_PATH = os.path.join(ROOT, 'script/results/testresult/results')
ANNO_PATH = os.path.join(ROOT, 'data/label/labels')
ANNO_OUT_PATH = os.path.join(ROOT, 'data/label/cocoanno')
XMLLIST_PATH = os.path.join(ROOT, 'data/voc2coco/xmllist.txt')

def result_to_coco_v2(txt_file_path, output='./'):
    txt_files = txt_file_path
    json_results = []
    id_count = 1
    with open(txt_files, 'r') as f:
        while True:
            lines = f.readline()
            if not lines:
                break
            monos = lines.split(',')
            data = dict()
            data['image_id'] = int(monos[0])
            data['bbox'] = [int(float(monos[2])), int(float(monos[3])), int(float(monos[4])), int(float(monos[5]))]
            data['score'] = float(monos[6])
            data['category_id'] = 0
            json_results.append(data)
        id_count += 1
    with open(os.path.join(output, "output_result.json"), 'w', encoding='utf-8') as json_file:
        json.dump(json_results, json_file, ensure_ascii=False)

def result_to_coco_v2_single(start, end, txt_file_path, output='./'):
    txt_files = txt_file_path
    json_results = []
    id_count = 1
    count = 0
    
    with open(txt_files, 'r') as f:
        while True:
            lines = f.readline()
            if not lines:
                break       
            monos = lines.split(',')
            data = dict()
            data['image_id'] = int(monos[0])
            if start <= data['image_id'] - 1 and data['image_id'] - 1 <= end:
                data['image_id'] -= start
                data['bbox'] = [int(float(monos[2])), int(float(monos[3])), int(float(monos[4])), int(float(monos[5]))]
                data['score'] = float(monos[6])
                data['category_id'] = 0
                json_results.append(data)
        id_count += 1
    with open(os.path.join(output, "output_result.json"), 'w', encoding='utf-8') as json_file:
        json.dump(json_results, json_file, ensure_ascii=False)


def result_to_coco():
    txt_files = glob.glob(os.path.join(RESULT_PATH, '*.txt'))
    names = [x.split('/')[-2] + '/' + x.split('/')[-1] for x in txt_files]
    import pdb
    pdb.set_trace()
    json_results = []
    filelist = []
    id_count = 0
    for file in txt_files:
        with open(file, 'r') as f:
            while True:
                lines = f.readline()
                if not lines:
                    break
                monos = lines.split(' ')
                data = dict()
                data['image_id'] = id_count
                data['bbox'] = [int(monos[3]), int(monos[4]), int(monos[5])-int(monos[3])
                    , int(monos[6]) - int(monos[4])]
                data['score'] = float(monos[1].split('(')[1].split(',')[0])
                data['category_id'] = 0
                json_results.append(data)
        id_count += 1

        filelist.append(file)
    with open("./output_result.json", 'w', encoding='utf-8') as json_file:
        json.dump(json_results, json_file, ensure_ascii=False)

    return filelist

def convert_to_coco(names, labels, type='visible_body', outpath = './', img_size=[7998, 4800]):
    with open(os.path.join(outpath, 'xmllist.txt'), 'w') as f:
        index = 1
        for name, label in zip(names, labels):
            bbox_list = label
            convert_single_pascal_v2(str(index)+'.jpg', bbox_list, img_size, outpath, type)
            f.writelines(str(index) + '.xml\n')
            index += 1
    convert(os.path.join(outpath, 'xmllist.txt'), outpath, os.path.join(outpath, 'anno.json'))



def convert_to_pascal():
    txt_files = glob.glob(os.path.join(ANNO_PATH, '*.txt'))
    count = 0
    f1 = open('./fileorder.txt', 'w')
    for file in txt_files:
        bbox_list = []

        with open(file, 'r') as f:
            while True:
                lines = f.readline()
                if not lines:
                    break
                monos = lines.split(' ')
                bbox_list.append([int(monos[1]), int(monos[2]), int(monos[3]), int(monos[4])])
        f1.writelines(file.split('/')[-1] + '\n')
        name = file.split('/')[-1].split('.')[0] + '.jpg'
        convert_single_pascal(name, str(count)+'.jpg', bbox_list, ANNO_OUT_PATH)
        count += 1
    f1.close()

def generate_xmllist():
    txt_files = glob.glob(os.path.join(ANNO_OUT_PATH, '*.xml'))
    with open(XMLLIST_PATH,'w') as f:
        for file in txt_files:
            name = file.split('/')[-1]
            f.writelines(name + '\n')


if __name__ == '__main__':
    #convert_to_pascal()
    #generate_xmllist()
    result_to_coco()
