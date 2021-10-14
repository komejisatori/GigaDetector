from sys import path
path.append(r'../')

from train.train_util import cal_ap
from script.config import *
import os
from data.dataloader.loader_util import load_all_names_and_labels_v2
RESULT_PATH = os.path.join(ROOT, 'det_new_fst.txt')
EVAL_NAME_PATH = os.path.join(ROOT, 'run/test_list.txt')
GT_PATH_1 = os.path.join(ROOT, 'data/label/gt_human_bbox_all.json')
GT_PATH_2 = os.path.join(ROOT, 'data/label/gt_human_bbox_all_part2_v2.json')
EVAL_LIST = []
with open(EVAL_NAME_PATH, 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        EVAL_LIST.append(line.split('\n')[0])

def run_fp_analyse():
    names, labels = load_all_names_and_labels_v2(GT_PATH_1, GT_PATH_2, type='visible body')
    annos = []
    eval_dict = {}
    eval_names = []
    eval_labels = []
    for (n,l) in zip(names, labels):
        if n in EVAL_LIST:
            eval_dict[n] = l
    for n in EVAL_LIST:
        eval_names.append(n)
        eval_labels.append(eval_dict[n])


    result_labels = []
    with open(RESULT_PATH, 'r') as f:
        #line = f.readline()
        last_id = 1
        while(True):
            line = f.readline()
            if not line:
                break
            monos = line.split(',')
            id = int(monos[0])
            result_labels.append([id, int(float(monos[2])),
                        int(float(monos[3])),
                        int(float(monos[4])+float(monos[2])),
                        int(float(monos[5])+float(monos[3])),
                        float(monos[6])])

    img_num = len(EVAL_LIST)
    ad = cal_ap(result_labels,eval_labels, img_num)

    height = []
    oriheight = []
    count = 0

    for anno in ad:
        a_count = 0
        for a in anno:
            if int(a) == 0:
                height.append(eval_labels[count][a_count][3] - eval_labels[count][a_count][1])
            oriheight.append(eval_labels[count][a_count][3] - eval_labels[count][a_count][1])
            a_count += 1
        count += 1

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages('./height.pdf')
    bins = np.arange(0,1001,50)
    x = np.asarray(height)
    y = np.asarray(oriheight)
    import pdb
    pdb.set_trace()
    plt.figure()
    plt.hist([x, y], bins, label=['TN','ALL'])
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    legend = plt.legend(prop=font)
    plt.ylabel('Amount', fontdict={'family' : 'Times New Roman', 'size'   : 15})
    plt.xlabel('Height', fontdict={'family' : 'Times New Roman', 'size'   : 15})
    pdf.savefig()
    plt.close()
    pdf.close()

