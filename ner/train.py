from torch.utils.data import DataLoader
from ner.config import TRAIN_FILE_PATH, DEV_FILE_PATH, TEST_FILE_PATH
from ner.dataset import NERDataset, PadBatch, getTag2Ids
from transformers import AdamW, get_linear_schedule_with_warmup
from ner.model import Bert_BiLSTM_CRF


if __name__=="__main__":
    train_dataset = NERDataset(TRAIN_FILE_PATH)  # trainset为预处理好的文本
    eval_dataset = NERDataset(DEV_FILE_PATH)  # validset为预处理好的文本
    test_dataset = NERDataset(TEST_FILE_PATH)  # testset为预处理好的文本

    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=64,
                            shuffle=True,
                            num_workers=4,
                            collate_fn=PadBatch)

    eval_iter = DataLoader(dataset=eval_dataset,
                           batch_size=32,
                           shuffle=False,
                           num_workers=4,
                           collate_fn=PadBatch)

    test_iter = DataLoader(dataset=test_dataset,
                           batch_size=32,
                           shuffle=False,
                           num_workers=4,
                           collate_fn=PadBatch)
    model = Bert_BiLSTM_CRF(getTag2Ids())
    optimizer = AdamW(model.parameters(), lr=0.001, eps=1e-6)
    len_dataset = len(train_dataset)
    epoch = 30
    batch_size = 64
    total_steps = (len_dataset // batch_size) * epoch \
        if len_dataset % batch_size == 0 else ( len_dataset // batch_size + 1) * epoch

    warm_up_ratio = 0.1  # 定义要预热的step，一般取10%
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_ratio * total_steps,
                                                num_training_steps=total_steps)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    accumulation_batchs=16
    for step, (inputs, labels, mask) in enumerate(train_iter):
        batch_loss = model(inputs, labels, mask)  # 计算 batch loss
        full_loss = batch_loss / accumulation_batchs  # 标准化 loss
        full_loss.backward()  # 反向传播，累加梯度
        if (step + 1) % accumulation_batchs == 0:
            optimizer.step()  # 更新优化器
            scheduler.step()  # 更新调度器
            optimizer.zero_grad()