# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from transformers import AdamW
import pandas as pd
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, recall_score, precision_score
from torch.utils.data import DataLoader

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
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


def train(config, model, train_data, dev_data):
    start_time = time.time()
    model.train()

    if config.model_name == 'dt_bert_base':
        from dt_utils import get_time_dif, collate_batch
    else :
        from utils import get_time_dif, collate_batch

    # 初始化dataloader
    train_dataloader = DataLoader(train_data,
                                  batch_size=config.batch_size,
                                  collate_fn=collate_batch,
                                  shuffle=True)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters,
                         lr=config.learning_rate,eps=config.adam_epsilon)

    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")  # 这里是“欧一”，不是“零一”

    # 设置 warm up学习
    t_total = len(train_dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                num_training_steps=t_total)

    # 输出训练有关的所有参数
    for k, v in config.__dict__.items():
        print("  {:18s} = {}".format(str(k), str(v)))

    total_batch = 0  # 记录进行到多少batch
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()

    logging_loss = 0.0
    optimizer.step()
    best_f_score = 0.

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, batch in enumerate(train_dataloader):
            inputs = {}
            for k, v in batch.items():
                inputs[k] = v.to(config.device)
            outputs = model(inputs)
            labels = inputs['label']
            loss = F.cross_entropy(outputs, labels)
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            logging_loss += loss.item()

            # 过gradient_accumulation_steps后才将梯度清零，不是每次更新/每过一个batch清空一次梯度，即每gradient_accumulation_steps次更新清空一次
            if (i + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()  # 更新学习率
                model.zero_grad()
                total_batch += 1

            if total_batch % 20 == 0 and i % config.gradient_accumulation_steps == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()

                train_acc = metrics.accuracy_score(true, predic)
                acc, f_score, dev_loss = evaluate(config, model, dev_data)
                if f_score > best_f_score:
                    best_f_score = f_score
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1},  Train Acc: {2:>6.2%}, Dev Loss:{3} ,Val Acc: {4},  Val F1: {5},  Time: {6} {7}'
                print(msg.format(total_batch, logging_loss / 50, train_acc, dev_loss / 50,acc, f_score, time_dif, improve))
                logging_loss = 0.0
                model.train()
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    print('验证集上最佳F1值为:', best_f_score)
    #test(config, model, test_iter)


# def test(config, model, test_data, fold_id = '__'):
#     # test
#     # 初始化dataloader
#     test_dataloader = DataLoader(test_data,
#                                   batch_size=config.batch_size,
#                                   collate_fn=collate_batch,
#                                   shuffle=False)
#     model.load_state_dict(torch.load(config.save_path))
#     model.eval()
#     start_time = time.time()
#     predict_all = np.array([], dtype=int)
#     probs = []
#     with torch.no_grad():
#         for i, batch in enumerate(test_dataloader):
#             inputs = {}
#             for k, v in batch.items():
#                 inputs[k] = v.to(config.device)
#             outputs = model(inputs)
#             #加个softmax 变成概率，保存。 用来集成学习
#             sm = torch.nn.Softmax(1)
#             probs.append(sm(outputs).detach().cpu().numpy())
#             predic = torch.max(outputs.data, 1)[1].cpu().numpy()
#             predict_all = np.append(predict_all, predic)
#     time_dif = get_time_dif(start_time)
#     probs = np.concatenate(probs, axis=0)
#     print("predict  end..........")
#     ids = pd.read_csv(config.test_path)
#
#     ids = ids.iloc[:, 0]
#     test = pd.DataFrame(predict_all)
#     test = pd.concat([ids, test], axis=1)
#
#     if(config.k_fold > 0):
#         result_name = 'result_' + config.model_name + '_' + str(config.k_fold) + 'fold_' + str(fold_id) + '.txt'
#         probs_name =  config.model_name + '_' + str(config.k_fold) + 'fold_' + str(fold_id) + '_probs'
#     else:
#         result_name = 'result_' + config.model_name + '.txt'
#         probs_name =  config.model_name + 'probs'
#     test.to_csv(result_name, sep='\t', index=False, encoding='utf8', header=None)
#     np.save(probs_name, probs)
#     print("结果保存在" + result_name)
#     print("Time usage:", time_dif)



def evaluate(config, model, dev_data):
    model.eval()
    if config.model_name == 'dt_bert_base':
        from dt_utils import get_time_dif, collate_batch
    else :
        from utils import get_time_dif, collate_batch

    # 初始化dataloader
    dev_dataloader = DataLoader(dev_data,
                                  batch_size=config.batch_size,
                                  collate_fn=collate_batch,
                                  shuffle=True)
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in dev_dataloader:
            inputs = {}
            for k, v in batch.items():
                inputs[k] = v.to(config.device)
            labels = inputs['label']
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    f_score = f1_score(y_true=labels_all, y_pred=predict_all, average='macro')
    acc = accuracy_score(y_true=labels_all, y_pred=predict_all)
    return acc, f_score, loss_total

def test(config, model, test_data):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    if config.model_name == 'dt_bert_base':
        from dt_utils import get_time_dif, collate_batch
    else :
        from utils import get_time_dif, collate_batch

    # 初始化dataloader
    test_dataloader = DataLoader(test_data,
                                  batch_size=config.batch_size,
                                  collate_fn=collate_batch,
                                  shuffle=True)
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {}
            for k, v in batch.items():
                inputs[k] = v.to(config.device)
            labels = inputs['label']
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    f_score = f1_score(y_true=labels_all, y_pred=predict_all, average='macro')
    acc = accuracy_score(y_true=labels_all, y_pred=predict_all)
    print("测试集上acc：", acc, "测试集上F1值：", f_score)
