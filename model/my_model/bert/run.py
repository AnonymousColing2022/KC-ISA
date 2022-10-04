# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network, test
from importlib import import_module
import argparse
from sklearn.model_selection import StratifiedKFold
import os
import random

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'SMP2021'  # 数据集

    model_name = args.model  # bert

    if model_name == 'dt_bert_base' :
        from dt_utils import get_time_dif, build_dataset_, get_time_dif
    else:
        from utils import get_time_dif, build_dataset_

    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    random.seed(2021)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset_(config)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_data, dev_data)
    test(config, model, test_data)

