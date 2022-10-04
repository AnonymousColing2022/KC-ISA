# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'dt_RNN_RNN'
        self.train_path = dataset + '/data/train.csv'  # 训练集
        self.dev_path = dataset + '/data/dev.csv'  # 验证集
        self.test_path = dataset + '/data/test.csv' # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab_word.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.kgembedding_path = dataset + '/data/kgembedding.pkl'
        # self.embedding_pretrained = None
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)['arr_0'].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.6                                              # 随机失活
        self.require_improvement = 200                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        # self.num_classes = 3                        # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        # self.n_vocab = 1000                                               # 词表大小，在运行时赋值
        self.num_epochs = 50                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.context_pad_size = 300  # 每句话处理成的长度(短填长切)
        self.target_pad_size = 128  # 每句话处理成的长度(短填长切
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 256                                         # lstm隐藏层
        self.num_layers = 2                                             # lstm层数


'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm_context = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.lstm_target = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                                    bidirectional=True, batch_first=True, dropout=config.dropout)

        self.lstm_kg = nn.LSTM(100, config.hidden_size, config.num_layers,
                                    bidirectional=True, batch_first=True, dropout=config.dropout)
        # coattention
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.sentence_liner = nn.Sequential(nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
                                            nn.Tanh())
        self.knowledge_linear = nn.Sequential(nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
                                              nn.Tanh())
        self.context_linear = nn.Sequential(nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
                                            nn.Tanh())

        self.final_lstm = nn.LSTM(config.hidden_size * 6, config.hidden_size, config.num_layers,
                                  bidirectional=True, batch_first=True, dropout=config.dropout)
        bidir = 2
        # self.fc1 = nn.Linear(config.hidden_size * config.num_layers * bidir * 2 * 2, 500)
        # # self.fc1 = nn.Linear(config.hidden_size * config.num_layers * bidir * 2, 500)
        # self.fc2 = nn.Linear(500, 300)
        # self.fc3 = nn.Linear(300, config.num_classes)
        self.W1 = nn.Linear(config.hidden_size * 2, config.hidden_size * 2, bias=False)
        self.W2 = nn.Linear(config.hidden_size * 2, config.hidden_size * 2, bias=False)
        self.fc1 = nn.Linear(config.hidden_size * 2 * 2, 256)
        self.fc2 = nn.Linear(256, config.num_classes)

    @staticmethod
    def masked_softmax(L, sequence_length):
        '''
        :param L: batch size, n, m
        :param sequence_length: batch size
        :return:
        '''
        device = L.device
        batch_size, n, m = L.shape
        # batch size, m
        
        sequence_mask = (torch.arange(0, m).repeat(batch_size, 1).to(device) < sequence_length.unsqueeze(-1)).float()
        # batch size, 1, m
        sequence_mask = (1.0 - sequence_mask.unsqueeze(1)) * -9999999.9
        attention_score = L + sequence_mask
        attention_weight = torch.softmax(attention_score, dim=-1)
        return attention_weight

    def coattention(self, feature_q, feature_p, q_length, p_length):
        '''
        :param feature_q: batch size, n, hidden size   辅助信息
        :param feature_p: batch size, m, hidden size   主要信息
        :param q_length: batch size
        :param p_length: batch size
        :return:
        '''
        # batch size, n, m
        L = torch.matmul(feature_q, feature_p.permute(0, 2, 1))
        q2p = self.masked_softmax(L, p_length)
        p2q = self.masked_softmax(L.permute(0, 2, 1), q_length)
        # batch size, n, hidden size
        q2p_feature = torch.matmul(q2p, feature_p)
        concat_q = torch.cat([feature_q, q2p_feature], dim=-1)
        # batch size, m, hidden size * 2
        p2q_feature = torch.matmul(p2q, concat_q)
        final_concat = torch.cat([p2q_feature, feature_p], dim=-1)
        output, hidden = self.final_lstm(final_concat)
        hc, cs = hidden
        batch_size = hc.shape[1]
        hc = hc.permute(1, 0, 2).reshape(batch_size, -1)
        cs = cs.permute(1, 0, 2).reshape(batch_size, -1)
        concat_state = torch.cat([hc, cs], dim=-1)
        return concat_state, output

    def forward(self, train):
        context = self.embedding(train[0])  # [batch_size, seq_len, embeding]=[128, 128, 300]
        context, _ = self.lstm_context(context)
        context_length = train[1]
        # context = torch.cat((context[:,0,:], context[:,-1,:]),1)
        target = self.embedding(train[2])  # [batch_size, seq_len, embeding]=[128, 128, 300]
        target_p, _ = self.lstm_target(target)# [128, n, 512]
        target = torch.mean(target_p, dim=1)
        sentence_length = train[3]
        kg = train[4]
        kg_q, _ = self.lstm_kg(kg)# [128, m, 512]
        #co attention 128, m, 512
        # HQ = self.tanh1(self.fc_co(kg_q))
        # 128, n, 512
        knowledge_feature = self.knowledge_linear(kg_q)
        # 128, m, 512
        sentence_feature = self.sentence_liner(target_p)
        # 128, l, 512
        context_feature = self.context_linear(context)

        kg_sent_coatt, kg_sent_output = self.coattention(knowledge_feature, sentence_feature, sentence_length, sentence_length)
        context_sent_coatt, context_sent_output = self.coattention(context_feature, sentence_feature, context_length, sentence_length)

        ks_out = torch.matmul(
            torch.softmax(torch.matmul(kg_sent_output, self.W1(context_sent_output).permute(0, 2, 1)), dim=-1),
            context_sent_output)
        cs_out = torch.matmul(
            torch.softmax(torch.matmul(context_sent_output, self.W2(kg_sent_output).permute(0, 2, 1)), dim=-1),
            kg_sent_output)

        ks_out = torch.mean(ks_out, dim=1)
        cs_out = torch.mean(cs_out, dim=1)

        # # 128, 512, n
        # # tmp = target_p.permute(0, 2, 1)
        # # 128, n, m
        # L = torch.matmul(knowledge_feature, sentence_feature.permute(0, 2, 1))
        # # 128, n, m
        # AD = self.masked_softmax(L, sentence_length)
        # # 128, m, n
        # AQ = self.masked_softmax(L.permute(0, 2, 1), sentence_length)
        # # knowledge attention to sentence
        # # 128, n, 512
        # kg2sent = torch.matmul(AD, sentence_feature) #CQ
        # # 128, n, 1024
        # concat_kg = torch.cat([knowledge_feature, kg2sent], dim=-1)
        # # 128, m, 1024
        # sent2kg = torch.matmul(AQ, concat_kg)
        # # 128, m, 1536
        # concat_sent = torch.cat([sentence_feature, sent2kg], dim=-1)
        # output, hidden = self.final_lstm(concat_sent)
        # hc, cs = hidden
        # batch_size = hc.shape[1]
        # hc = hc.permute(1, 0, 2).reshape(batch_size, -1)
        # cs = cs.permute(1, 0, 2).reshape(batch_size, -1)
        # concat_state = torch.cat([hc, cs], dim=-1)

        # concat_state = torch.cat((context_sent_coatt, kg_sent_coatt), dim=1)

        concat_state = torch.cat((ks_out, cs_out), dim=1)

        out = self.fc2(self.fc1(concat_state))

        return out
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
import torch
if __name__ == '__main__':
    train = [
        [], [],
        torch.empty([128, 128]).uniform_(to=100).type(torch.int64),
        torch.tensor([100] * 128),
        torch.rand(128, 128, 100),
    ]
    config = Config(dataset=' ', embedding=' ')
    model = Model(config)
    output = model(train)
