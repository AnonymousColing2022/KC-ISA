# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer, BertConfig


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'dt_bert_base'
        self.train_path = dataset + '/data/train.csv'                                # 训练集
        self.dev_path = dataset + '/data/dev.csv'                                    # 验证集
        self.test_path = dataset + '/data/test.csv'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.kgembedding_path = dataset + '/data/kgembedding.pkl'
        self.save_path = dataset + '/saved_dict/'  + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 300                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 20                                             # epoch数
        self.batch_size = 16                                              # mini-batch大小
        self.context_pad_size = 300                                              # 每句话处理成的长度(短填长切)
        self.target_pad_size = 128                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5                                       # 学习率
        self.bert_path = './pretrain_model/bert_base'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.adam_epsilon = 1e-8
        self.warmup_steps = 400
        self.max_grad_norm = 1.0
        self.gradient_accumulation_steps = 8
        self.model_config = BertConfig.from_pretrained(self.bert_path)
        self.k_fold = -1
        self.num_layers = 2
        self.dropout = 0.5

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, config = config.model_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.bert1 = BertModel.from_pretrained(config.bert_path, config=config.model_config)
        for param in self.bert1.parameters():
            param.requires_grad = True
        self.final_lstm = nn.LSTM(config.hidden_size * 3, config.hidden_size, config.num_layers,
                                  bidirectional=True, batch_first=True, dropout=config.dropout)
        self.kg_lstm = nn.LSTM(100, 384, config.num_layers,
                                  bidirectional=True, batch_first=True, dropout=config.dropout)

        self.sentence_liner = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                            nn.Tanh())
        self.context_linear = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                            nn.Tanh())
        self.kg_linear = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                            nn.Tanh())
        # self.fc1 = nn.Linear(config.hidden_size * 2 * 2 * 2, 300)
        # self.fc2 = nn.Linear(300, 128)
        # self.fc3 = nn.Linear(128, config.num_classes)

        self.W1 = nn.Linear(config.hidden_size * 2, config.hidden_size * 2, bias=False)
        self.W2 = nn.Linear(config.hidden_size * 2, config.hidden_size * 2, bias=False)

        self.fc1 = nn.Linear(config.hidden_size * 2 * 2, 128)
        # self.fc1 = nn.Linear(config.hidden_size, 128)
        self.fc2 = nn.Linear(128, config.num_classes)

        # self.fc1 = nn.Linear(config.hidden_size, 128)
        # self.fc2 = nn.Linear(128, config.num_classes)


    @staticmethod
    def masked_softmax(L, sequence_length=None):
        '''
        :param L: batch size, n, m
        :param sequence_length: batch size
        :return:
        '''
        device = L.device
        batch_size, n, m = L.shape
        # batch size, m
        if sequence_length is not None:
            sequence_mask = (
                        torch.arange(0, m).repeat(batch_size, 1).to(device) < sequence_length.unsqueeze(-1)).float()
            # batch size, 1, m
            sequence_mask = (1.0 - sequence_mask.unsqueeze(1)) * -9999999.9
            attention_score = L + sequence_mask
        else:
            attention_score = L
        attention_weight = torch.softmax(attention_score, dim=-1)
        return attention_weight

    def coattention(self, feature_q, feature_p, q_length=None, p_length=None):
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

    def forward(self, inputs):
        context_text = inputs['context']  # 输入的句子
        target_text = inputs['target']  # 输入的句子
        context_mask = inputs['context_mask']  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        target_mask = inputs['target_mask']  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        context = self.bert1(context_text, attention_mask=context_mask, output_hidden_states=False)
        target = self.bert(target_text, attention_mask=target_mask, output_hidden_states=False)
        _context = context.get('last_hidden_state')
        _target = target.get('last_hidden_state')
        #
        kg = inputs['kg_line']
        kg_q, _ = self.kg_lstm(kg)
        kg_feature = self.kg_linear(kg_q)
        kg_len = inputs['kg_len']
        #
        #
        # context_pooler = context.get('pooler_output')
        # target_pooler = target.get('pooler_output')
        #
        _context = self.context_linear(_context)
        _target = self.sentence_liner(_target)
        #
        con_sent_state, con_sent_output = self.coattention(_context, _target)
        kg_sent_state, kg_sent_output = self.coattention(kg_feature, _target, kg_len)
        #
        ks_out = torch.matmul(
            torch.softmax(torch.matmul(kg_sent_output, self.W1(con_sent_output).permute(0, 2, 1)), dim=-1),
            con_sent_output)
        cs_out = torch.matmul(
            torch.softmax(torch.matmul(con_sent_output, self.W2(kg_sent_output).permute(0, 2, 1)), dim=-1),
            kg_sent_output)

        ks_out = torch.mean(ks_out, dim=1)
        cs_out = torch.mean(cs_out, dim=1)
        #
        #
        # # _context = torch.mean(_context, dim=1)
        # # _target = torch.mean(con_sent_output, dim=1)
        #
        # # out = self.fc3(self.fc2(self.fc1(_target)))
        concat_state = torch.cat((ks_out, cs_out), dim=1)
        out = self.fc2(self.fc1(concat_state))
        return out


