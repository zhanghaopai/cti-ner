import os

# 获取当前脚本所在的目录
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
# 获取当前脚本所在的项目根目录
ROOT_PATH = os.path.dirname(CURRENT_PATH)

TRAIN_FILE_PATH = ROOT_PATH + '/resource/APTNERtrain.txt'
DEV_FILE_PATH = ROOT_PATH + '/resource/APTNERdev.txt'
TEST_FILE_PATH = ROOT_PATH + '/resource/APTNERtest.txt'

VOCAB_PATH = ROOT_PATH + '/resource/vocab.txt'
LABEL_PATH = ROOT_PATH + '/resource/label.txt'

WORD_PAD = '<PAD>'
WORD_UNK = '<UNK>'
CLS='[CLS]'
SEP='[SEP]'

VOCAB_SIZE = 3000

SENT_MAX_LEN = 256
