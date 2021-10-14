
import os
from sys import path
path.append(r'./')
from data.dataloader.loader_util import load_all_names_and_labels_v2
import run.config as run_cfg
PREFIX = run_cfg.ROOT
TEST_LIST_PATH = PREFIX + '/api/test_list.txt'
TEST_LIST = []
EVAL_LIST = []
with open(TEST_LIST_PATH, 'r') as f:
    while True:
        lines = f.readline()
        if not lines:
            break
        TEST_LIST.append(PREFIX + '/data/frame/test/' + lines.split('\n')[0])
        EVAL_LIST.append(lines.split('\n')[0])


def read_det_file(gt_json_path_1=PREFIX+'/data/label/gt_human_bbox_all.json', gt_json_path_2=PREFIX+'/data/label/gt_human_bbox_all_part2_v2.json', img_files_list=EVAL_LIST, pre = 'fst_pre', filepath=PREFIX+'/det_cascade_rcnn.txt'):
    names, labels = load_all_names_and_labels_v2(gt_json_path_1, gt_json_path_2, type='visible body')
    eval_dict = {}
    eval_names = []
    eval_labels = []
    for name, label in zip(names, labels):
        if name in img_files_list:
            eval_dict[name] = label
    for name in img_files_list:
        eval_names.append(name)
        eval_labels.append(eval_dict[name])
    f_name = open(PREFIX+'/api/lamr/name_' + pre +'.txt','w')
    f_score = open(PREFIX+'/api/lamr/score_' + pre + '.txt', 'w')
    f_anno = open(PREFIX+'/api/lamr/anno_' + pre +'.txt', 'w')
    f_gt = open(PREFIX+'/api/lamr/gt_' + pre +'.txt', 'w')
    id_count = 1
    f_name.writelines(str(id_count)+',\n')
    temp = 0
    with open(filepath, 'r') as f:
        while True:
            lines = f.readline()
            monos = lines.split(',')
            if len(monos) > 2:
                if monos[0] == str(id_count):
                    f_score.writelines(str(monos[6]) + ';')
                    f_anno.writelines(str(monos[2])+','+str(monos[3])+','+str(monos[4])+','+str(monos[5])+';')
                else:
                    id_count+=1
                    f_name.writelines(str(id_count) + ',\n')
                    f_score.writelines('\n')
                    f_anno.writelines('\n')
                    f_score.writelines(str(monos[6]) + ';')
                    f_anno.writelines(str(monos[2]) + ',' + str(monos[3]) + ',' + str(monos[4]) + ',' + str(monos[5]) + ';')

                temp += 1
            if not lines:
                break
    for label in eval_labels:
        for l in label:
            f_gt.writelines(str(l[0])+','+str(l[1])+','+str(l[2]-l[0])+','+str(l[3]-l[1])+';')
        f_gt.writelines('\n')
    f_name.close()
    f_score.close()
    f_anno.close()
    f_gt.close()

if __name__ == '__main__':
    read_det_file()
