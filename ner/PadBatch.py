import torch


class PadBatch():
    def __init__(self, batch):
        self.batch=batch

    def __call__(self, *args, **kwargs):
        maxlen = max([i[2] for i in self.batch])
        token_tensors = torch.LongTensor([i[0] + [0] * (maxlen - len(i[0])) for i in self.batch])
        label_tensors = torch.LongTensor([i[1] + [0] * (maxlen - len(i[1])) for i in self.batch])
        mask = (token_tensors > 0)
        return token_tensors, label_tensors, mask
