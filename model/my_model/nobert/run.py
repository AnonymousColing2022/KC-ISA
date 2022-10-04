# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network,test
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=True, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'SMP2021'  # 数据集

    # embedding.npz 分词
    embedding = 'embedding.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # TextCNN, TextRNN, GCN
    if model_name == 'GCN' or model_name == 'TextRNN_GCN' or model_name == 'TextCNN_GCN':
        from utils_gcn import build_dataset, build_iterator, get_time_dif
    elif model_name == 'd_CNN_CNN' or model_name == 'd_RNN_RNN' or model_name == 'd_RNN_CNN' or model_name == 'd_CNN_RNN' or model_name == 'd_rnn_CNN_CNN' \
            or model_name == 'd_RNNAtt_CNN' or model_name == 'd_RCNN_CNN':
        from d_utils import build_dataset, build_iterator, get_time_dif
    elif model_name == 'd_GCN_CNN' or model_name == 'd_GCN_RNN':
        from d_utils_gcn import build_dataset, build_iterator, get_time_dif
    elif model_name == 'dt_RNN_RNN2' or model_name == 'dt_RNN_RNN':
        from dt_utils import build_dataset, build_iterator, get_time_dif
    else :
        from d_utils import build_dataset, build_iterator, get_time_dif


    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    # model_Target = x_Target.Model(config_Target).to(config_Context.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter)
    test(config, model, test_iter)

