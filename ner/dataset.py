from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import torch

from ner.config import TRAIN_FILE_PATH, SENT_MAX_LEN, CLS, SEP
from ner.data_process import get_labels

tokenizer = AutoTokenizer.from_pretrained('dslim/bert-base-NER')
LABELS=[CLS, SEP] + get_labels()
tag2idx={tag:idx for idx,tag in enumerate(LABELS)}
id2tag={idx:tag for idx,tag in enumerate(LABELS)}

def getTag2Ids():
    return tag2idx

class NERDataset(Dataset):
    def __init__(self, f_path):
        # 句子按照单词分词
        self.sentenses=[]
        self.tag_li=[]
        with open(f_path, mode='r', encoding='utf-8') as f:
            lines = [line.split('\n')[0] for line in f.readlines() if (len(line) > 0)]
        lines=[line for line in lines if len(line.split(' ')) >= 2]
        words=[line.split(' ')[0] for line in lines]
        tags=[line.split(' ')[1] for line in lines]

        # 组成句子，以及对应的tag
        word, tag = [], []
        for char, t in zip(words, tags):
            if char != '.':
                word.append(char)
                tag.append(t)
            else:
                if len(word) > SENT_MAX_LEN:
                    self.sentenses.append([CLS] + word[:SENT_MAX_LEN] + [SEP])
                    self.tag_li.append([CLS] + tag[:SENT_MAX_LEN] + [SEP])
                else:
                    self.sentenses.append([CLS] + word + [SEP])
                    self.tag_li.append([CLS] + tag + [SEP])
                word, tag = [], []


    def __len__(self):
        return len(self.sentenses)

    def __getitem__(self, idx):
        words, tags=self.sentenses[idx], self.tag_li[idx]
        # sentense 编码
        token_ids = tokenizer.convert_tokens_to_ids(words)
        # label 编码
        tag_ids=[tag2idx[tag] for tag in tags]
        seqlen=len(tag_ids)
        return token_ids, tag_ids, seqlen



if __name__=="__main__":
    print(tag2idx)
    nerDataset = NERDataset(TRAIN_FILE_PATH)
