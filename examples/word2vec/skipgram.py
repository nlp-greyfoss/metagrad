import numpy as np
from tqdm import tqdm

import metagrad.module as nn
from metagrad import Tensor, cuda
from metagrad.dataloader import DataLoader
from metagrad.dataset import Dataset
from metagrad.loss import CrossEntropyLoss
from metagrad.optim import SGD
from utils import BOS_TOKEN, EOS_TOKEN, WEIGHT_INIT_RANGE, load_corpus


class SkipGramDataset(Dataset):
    def __init__(self, corpus, vocab, window_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]

        for sentence in tqdm(corpus, desc='Dataset Construction'):
            sentence = [self.bos] + sentence + [self.eos]

            for i in range(1, len(sentence) - 1):
                # 模型输入：当前词
                w = sentence[i]
                # 模型输出： 窗口大小内的上下文
                # max 和 min 防止越界取到非预期的单词
                left_context_index = max(0, i - window_size)
                right_context_index = min(len(sentence), i + window_size)
                context = sentence[left_context_index:i] + sentence[i + 1:right_context_index + 1]
                self.data.extend([(w, c) for c in context])

        self.data = np.asarray(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    @staticmethod
    def collate_fn(examples):
        '''
        自定义整理函数
        :param examples:
        :return:
        '''
        inputs = Tensor([ex[0] for ex in examples])
        targets = Tensor([ex[1] for ex in examples])
        return inputs, targets


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs: Tensor) -> Tensor:
        # 得到输入词向量
        embeds = self.embeddings(inputs)
        # 根据输入词向量，对上下文进行预测
        output = self.output(embeds)
        return output


if __name__ == '__main__':
    embedding_dim = 64
    window_size = 3
    batch_size = 1024
    num_epoch = 10
    min_freq = 3  # 保留单词最少出现的次数

    # 读取文本数据，构建Skip-gram模型训练数据集
    corpus, vocab = load_corpus('../data/xiyouji.txt', min_freq)
    dataset = SkipGramDataset(corpus, vocab, window_size=window_size)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=True
    )

    loss_func = CrossEntropyLoss()
    # 构建Skip-gram模型，并加载至device
    device = cuda.get_device("cuda:1" if cuda.is_available() else "cpu")
    model = SkipGramModel(len(vocab), embedding_dim)
    model.to(device)
    optimizer = SGD(model.parameters(), lr=1)

    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
            inputs, targets = [x.to(device) for x in batch]
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss
        print(f"Loss: {total_loss.item():.2f}")
