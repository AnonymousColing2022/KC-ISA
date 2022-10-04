# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, recall_score, precision_score
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter
import pandas as pd


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = 99999.9
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    best_f_score = 0.
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains_context, labels) in enumerate(train_iter):
            outputs = model(trains_context)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 10 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss, f_score = evaluate(config, model, dev_iter)
                if f_score > best_f_score:
                    best_f_score = f_score
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%}, Val acc: {3} .Val Loss: {4:>5.2}, Val F1: {5},  Time: {6} {7}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_acc, dev_loss, f_score, time_dif, improve))
                # writer.add_scalar("loss/train", loss.item(), total_batch)
                # writer.add_scalar("loss/dev", f_score, total_batch)
                # writer.add_scalar("acc/train", train_acc, total_batch)
                # writer.add_scalar("acc/dev", dev_loss, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
        scheduler.step()  # 学习率衰减
    writer.close()


def test(config, model, data_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts_context, labels in data_iter:
            outputs = model(texts_context)
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    f_score = f1_score(y_true=labels_all, y_pred=predict_all, average='macro')
    print("测试集acc:", acc, ", F1值为:", f_score)

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts_context, labels in data_iter:
            outputs = model(texts_context)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    f_score = f1_score(y_true=labels_all, y_pred=predict_all, average='macro')
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter), f_score
