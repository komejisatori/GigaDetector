from sys import path
import os
path.append(r'../')
from data.dataloader.config import dataloader as loader_cfg
from data.dataloader.loader_util import load_all_names_and_labels
from script.config import *

CLASS_INDEX = ['10','11']
OUTPUT_PATH = os.path.join(ROOT, 'data/label/labels')


def generate_gt_files(json_path=loader_cfg['json_path']):
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    names, labels = load_all_names_and_labels(json_path)
    for name, label in zip(names, labels):
        if name[:2] in CLASS_INDEX:
            print('[gt logger] dealing with {}'.format(name))
            gt_filename = name.split('/')[1].split('.')[0]
            with open(os.path.join(OUTPUT_PATH, gt_filename + '.txt'), 'w') as f:
                for l in label:
                    f.writelines('visible_body ')
                    f.writelines(str(int(l[0])) + ' ')
                    f.writelines(str(int(l[1])) + ' ')
                    f.writelines(str(int(l[2])) + ' ')
                    f.writelines(str(int(l[3])) + '\n')


if __name__ == "__main__":
    generate_gt_files()
