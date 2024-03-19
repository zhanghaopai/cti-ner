import torch
import torch.nn as nn

from transformers import BertModel
from torchcrf import CRF


class Bert_BiLSTM_CRF(nn.Module):

    def __init__(self, tag_to_idx, embedding_dim=768, hidden_dim=256):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.tag_to_idx=tag_to_idx
        self.tag_size=len(tag_to_idx)
        self.hidden_dim=hidden_dim
        self.embedding_dim=embedding_dim

        self.bert = BertModel.from_pretrained('dslim/bert-base-NER')
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(hidden_dim, self.tag_size)
        self.crf = CRF(self.tag_size, batch_first=True)

    def _get_features(self, sentence):
        with torch.no_grad():
            embeds, _ = self.bert(sentence)
        enc, _ = self.lstm(embeds)
        enc = self.dropout(enc)
        feats = self.linear(enc)
        return feats

    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence)
        if not is_test:  # 训练阶段，返回loss
            loss = -self.crf.forward(emissions, tags, mask, reduction='mean')
            return loss
        else:  # 测试阶段，返回decoding结果
            decode = self.crf.decode(emissions, mask)
            return decode



