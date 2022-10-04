# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import csv
from torch.utils.data import Dataset
import jieba
import pickle as pkl

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


# def build_dataset(config):
#
#     def load_dataset(path,context_pad_size = 256, target_pad_size = 128, is_test = False):
#         contents = []
#         with open(path, 'r', encoding='UTF-8') as f:
#             reader = csv.reader(f)
#             for line in reader:
#                 if line[0] == 'id':
#                     continue
#                 context = line[1]
#                 target = line[2]
#                 if (is_test == False):
#                     label = line[3]
#                 context_token = config.tokenizer.tokenize(context)
#                 target_token = config.tokenizer.tokenize(target)
#                 context_token = [CLS] + context_token
#                 target_token = [CLS] + target_token
#                 context_seq_len = len(context_token)
#                 target_seq_len = len(target_token)
#                 context_mask = []
#                 target_mask = []
#                 context_token_ids = config.tokenizer.convert_tokens_to_ids(context_token)
#                 target_token_ids = config.tokenizer.convert_tokens_to_ids(target_token)
#                 if target_pad_size:
#                     if len(context_token) < context_pad_size:
#                         context_mask = [1] * len(context_token_ids) + [0] * (context_pad_size - len(context_token))
#                         context_token_ids += ([0] * (context_pad_size - len(context_token)))
#                     else:
#                         context_mask = [1] * context_pad_size
#                         context_token_ids = context_token_ids[:context_pad_size]
#                         context_seq_len = context_pad_size
#                     if len(target_token) < target_pad_size:
#                         target_mask = [1] * len(target_token_ids) + [0] * (target_pad_size - len(target_token))
#                         target_token_ids += ([0] * (target_pad_size - len(target_token)))
#                     else:
#                         target_mask = [1] * target_pad_size
#                         target_token_ids = target_token_ids[:target_pad_size]
#                         target_seq_len = target_pad_size
#                 if (is_test == False):
#                     contents.append((context_token_ids, context_seq_len, context_mask,target_token_ids, target_seq_len, target_mask, int(label)))
#                 elif (is_test == True):
#                     contents.append((context_token_ids, context_seq_len, context_mask, target_token_ids,target_seq_len, target_mask))
#         return contents
#
#     train = load_dataset(config.train_path, config.context_pad_size, config.target_pad_size)
#     dev = load_dataset(config.dev_path, config.context_pad_size, config.target_pad_size)
#     test = load_dataset(config.test_path, config.context_pad_size, config.target_pad_size, is_test = True)
#     return train, dev, test


def build_dataset_(config):

    def load_dataset_(path,context_pad_size = 300, target_pad_size = 100, is_test = False):
        contents = []
        kg_tokenizer = lambda x: x.split(' ')
        kgembedding = pkl.load(open(config.kgembedding_path, 'rb'))
        with open(path, 'r', encoding='UTF-8') as f:
            reader = csv.reader(f)
            for line in reader:
                if line[0] == 'id':
                    continue
                context = line[1]
                target = line[2]
                if (is_test == False):
                    label = line[3]


                kg = target
                kg = jieba.cut(kg, cut_all=False)
                kg = " ".join(kg)
                kg_token = kg_tokenizer(kg)
                kg_len = len(kg_token)
                if len(kg_token) < config.target_pad_size:
                    kg_token.extend(" " * (target_pad_size - len(kg_token)))
                else:
                    kg_token = kg_token[:target_pad_size]
                    kg_len = target_pad_size
                kg_line = []
                for word in kg_token:
                    kg_line.append(kgembedding.get(word, [0] * 100))

                context_inputs = config.tokenizer.encode_plus(text=context, max_length=context_pad_size, truncation=True)
                target_inputs = config.tokenizer.encode_plus(text=target, max_length=target_pad_size, truncation=True)

                context_token_ids, context_mask, context_type_ids = context_inputs['input_ids'], context_inputs['attention_mask'], context_inputs['token_type_ids']
                target_token_ids, target_mask, target_type_ids = target_inputs['input_ids'], target_inputs['attention_mask'], target_inputs['token_type_ids']
                if (is_test == False):
                    contents.append((context_token_ids, context_type_ids, context_mask,target_token_ids, target_type_ids, target_mask, int(label), kg_line, kg_len))
                elif (is_test == True):
                    contents.append((context_token_ids, context_type_ids, context_mask, target_token_ids,target_type_ids, target_mask))
        return contents

    train = load_dataset_(config.train_path, config.context_pad_size, config.target_pad_size)
    dev = load_dataset_(config.dev_path, config.context_pad_size, config.target_pad_size)
    test = load_dataset_(config.test_path, config.context_pad_size, config.target_pad_size)
    return SentimentDataset(train), SentimentDataset(dev), SentimentDataset(test)

class SentimentDataset(Dataset):
    def __init__(self, contents):
        self.contents = contents
    def __len__(self):
        return len(self.contents)
    def __getitem__(self, item):
        return self.contents[item]



# class DatasetIterater(object):
#     def __init__(self, batches, batch_size, device):
#         self.batch_size = batch_size
#         self.batches = batches
#         self.n_batches = len(batches) // batch_size
#         self.residue = False  # 记录batch数量是否为整数
#         if len(batches) % self.n_batches != 0:
#             self.residue = True
#         self.index = 0
#         self.device = device
#
#     def _to_tensor(self, datas):
#         if len(datas[0]) == 7:
#             context = torch.LongTensor([_[0] for _ in datas]).to(self.device)
#             # pad前的长度(超过pad_size的设为pad_size)
#             context_seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
#             context_mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)
#
#             target = torch.LongTensor([_[3] for _ in datas]).to(self.device)
#             # pad前的长度(超过pad_size的设为pad_size)
#             target_seq_len = torch.LongTensor([_[4] for _ in datas]).to(self.device)
#             target_mask = torch.LongTensor([_[5] for _ in datas]).to(self.device)
#
#             y = torch.LongTensor([_[6] for _ in datas]).to(self.device)
#             return (context, context_seq_len, context_mask),(target, target_seq_len, target_mask), y
#
#         if len(datas[0]) == 6:
#             context = torch.LongTensor([_[0] for _ in datas]).to(self.device)
#             # pad前的长度(超过pad_size的设为pad_size)
#             context_seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
#             context_mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)
#
#             target = torch.LongTensor([_[3] for _ in datas]).to(self.device)
#             # pad前的长度(超过pad_size的设为pad_size)
#             target_seq_len = torch.LongTensor([_[4] for _ in datas]).to(self.device)
#             target_mask = torch.LongTensor([_[5] for _ in datas]).to(self.device)
#             return (context, context_seq_len, context_mask),(target, target_seq_len, target_mask)
#
#     def __next__(self):
#         if self.residue and self.index == self.n_batches:
#             batches = self.batches[self.index * self.batch_size: len(self.batches)]
#             self.index += 1
#             batches = self._to_tensor(batches)
#             return batches
#
#         elif self.index >= self.n_batches:
#             self.index = 0
#             raise StopIteration
#         else:
#             batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
#             self.index += 1
#             batches = self._to_tensor(batches)
#             return batches
#
#     def __iter__(self):
#         return self
#
#     def __len__(self):
#         if self.residue:
#             return self.n_batches + 1
#         else:
#             return self.n_batches

def collate_batch(datas):

    # 在这里pad 减少pad字符，提高显存利用率
    def pad(batch, pad_token=0):
        new_batch = []
        max_len = max(len(b) for b in batch)
        for b in batch:
            padding_len = max_len - len(b)
            new_batch.append(b + [pad_token] * padding_len)
        return new_batch

    batch = {}
    batch['context'] = torch.LongTensor(pad([_[0] for _ in datas]))
    # pad前的长度(超过pad_size的设为pad_size)
    batch['context_type_ids'] = torch.LongTensor(pad([_[1] for _ in datas]))
    batch['context_mask'] = torch.LongTensor(pad([_[2] for _ in datas]))

    batch['target'] = torch.LongTensor(pad([_[3] for _ in datas]))
    # pad前的长度(超过pad_size的设为pad_size)
    batch['target_type_ids'] = torch.LongTensor(pad([_[4] for _ in datas]))
    batch['target_mask'] = torch.LongTensor(pad([_[5] for _ in datas]))


    batch['label'] = torch.LongTensor([_[6] for _ in datas])
    batch['kg_line'] = torch.FloatTensor([_[7] for _ in datas])
    batch['kg_len'] = torch.LongTensor([_[8] for _ in datas])
    return batch



# def build_iterator(dataset, config):
#     iter = DatasetIterater(dataset, config.batch_size, config.device)
#     return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
