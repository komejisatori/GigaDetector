import json
import glob
from sys import path
path.append(r'../')
from data.dataloader.config import dataloader as loader_cfg
from data.dataloader.loader_util import generate_name_list, convert_to_pascal, split_train_test


if __name__ == '__main__':
    # generate_name_list()
    # convert_to_pascal()
    split_train_test()