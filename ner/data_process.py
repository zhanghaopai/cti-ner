from glob import glob
import os
import random
import pandas as pd
from ner.config import *


def generate_vocab():
    # 读取csv
    df = pd.read_csv(TRAIN_FILE_PATH, usecols=[0], names=['word'], delimiter=' ')
    # 生成词表
    all_vocab=df['word'].value_counts().keys().tolist()
    all_vocab=[WORD_PAD, WORD_UNK] + all_vocab
    # 生成词列表
    vocab=all_vocab[:VOCAB_SIZE]
    # 生成字典
    vocab_dict={value:index for index,value in enumerate(vocab)}
    # 生成两列
    vocab_pd=pd.DataFrame(list(vocab_dict.items()))
    vocab_pd.to_csv(VOCAB_PATH, header=None, index=None)

def generate_label():
    # 读取csv
    df = pd.read_csv(TRAIN_FILE_PATH, usecols=[1], names=['label'], delimiter=' ')
    # 生成词表
    all_label=df['label'].value_counts().keys().tolist()
    # 生成字典
    label_dict={value:index for index,value in enumerate(all_label)}
    # 生成两列
    label_pd=pd.DataFrame(list(label_dict.items()))
    label_pd.to_csv(LABEL_PATH, header=None, index=None)

def get_labels():
    df = pd.read_csv(LABEL_PATH, usecols=[0], names=['label'])
    return df['label'].tolist()


if __name__=='__main__':
    # generate_vocab()
    # generate_label()
    print(get_labels())