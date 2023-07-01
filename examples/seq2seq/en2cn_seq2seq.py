import random
from typing import Tuple

from tqdm import tqdm

import metagrad.module as nn
from examples.embeddings.utils import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from examples.seq2seq.base import Encoder, Decoder
from examples.seq2seq.dataset import load_dataset_nmt, process_nmt, truncate_pad, read_nmt, cht_to_chs, tokenize_nmt
from metagrad import Tensor, cuda
from metagrad import init, debug_mode
from metagrad import functions as F
from metagrad.loss import CrossEntropyLoss
from metagrad.metrics import bleu_score
from metagrad.optim import Adam
from metagrad.tensor import no_grad
from metagrad.utils import grad_clipping


class NMTEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout, bidirectional=True):
        super().__init__()

        # 嵌入层 获取输入序列中每个单词的嵌入向量 padding_idx不需要更新嵌入
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # 基于双向GRU实现
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=num_hiddens, num_layers=num_layers, dropout=dropout,
                          bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq):
        '''
               Args:
                   input_seq:  形状 (seq_len, batch_size)
               Returns:
        '''
        # (seq_len, batch_size, embed_size)
        # embedded = self.dropout(self.embedding(input_seq))
        embedded = self.embedding(input_seq)
        # outputs (seq_len, batch_size, num_direction * num_hiddens)
        # hidden  (num_direction * num_layers, batch_size, num_hiddens)
        outputs, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            outputs = outputs[:, :, :self.rnn.hidden_size] + outputs[:, :, self.rnn.hidden_size:]
        # outputs (seq_len, batch_size, num_hiddens)
        # hidden  (num_direction * num_layers, batch_size, num_hiddens)
        return outputs, hidden


class NMTDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=num_hiddens, num_layers=num_layers, dropout=dropout)
        # 将隐状态转换为词典大小维度
        self.fc_out = nn.Linear(num_hiddens, vocab_size)
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq, hidden) -> Tuple[Tensor, Tensor]:
        input_seq = input_seq.unsqueeze(0)
        # input = (1, batch_size)
        embedded = self.dropout(self.embedding(input_seq))
        # embedded = self.embedding(input_seq)
        # embedded = (1, batch_size, embed_size)
        output, hidden = self.rnn(embedded, hidden)
        #
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_seq, target_seq, teacher_forcing_ratio=1) -> Tensor:
        '''

        Args:
            input_seq: (seq_len, batch_size)
            target_seq: (seq_len, batch_size)
            teacher_forcing_ratio: 强制教学比率
        Returns:

        '''
        max_len, batch_size = target_seq.shape
        # tgt_vocab_size = self.decoder.vocab_size

        outputs = []
        # hidden  (num_direction * num_layers, batch_size, num_hiddens)
        _, hidden = self.encoder(input_seq)
        # decoder_input (1, batch_size)
        decoder_input = target_seq[0, :]
        # 确保tgt[t]看到的是下一个label
        for t in range(1, max_len):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs.append(output)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = target_seq[t] if teacher_force else top1

        return F.stack(outputs)


def train_epoch(model, data_iter, optimizer, criterion, clip, device, tf_ratio):
    model.train()
    epoch_loss = 0

    for batch in data_iter:
        optimizer.zero_grad()
        inputs, targets = [x.to(device).T for x in batch]
        outputs = model(inputs, targets, tf_ratio)

        outputs = outputs.view(-1, outputs.shape[2])
        targets = targets[1:].view(-1)

        loss = criterion(outputs, targets)
        loss.backward()
        with no_grad():
            # 梯度裁剪
            grad_clipping(model, clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss


def evaluate(model, data_iter, criterion, device):
    model.eval()
    epoch_loss = 0
    with no_grad():
        for batch in data_iter:
            inputs, targets = [x.to(device).T for x in batch]
            outputs = model(inputs, targets, 0)  # 评估时不用teacher forcing

            outputs = outputs.view(-1, outputs.shape[2])
            targets = targets[1:].view(-1)

            loss = criterion(outputs, targets)
            epoch_loss += loss.item()

    return epoch_loss


def translate(model, src_sentence, src_vocab, tgt_vocab, max_len, device):
    """翻译一句话"""
    # 在预测时设置为评估模式
    model.eval()
    if isinstance(src_sentence, str):
        src_sentence = process_nmt(src_sentence)
        src_tokens = [src_vocab[BOS_TOKEN]] + [src_sentence.split(' ')] + [src_vocab[EOS_TOKEN]]
    else:
        src_tokens = [src_vocab[BOS_TOKEN]] + src_vocab[src_sentence] + [src_vocab[EOS_TOKEN]]

    src_tokens = truncate_pad(src_tokens, max_len, src_vocab[PAD_TOKEN])
    # 添加批量轴
    src_tensor = Tensor(src_tokens, dtype=int, device=device).unsqueeze(1)
    with no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    trg_indexes = [tgt_vocab[BOS_TOKEN]]
    for _ in range(max_len):
        trg_tensor = Tensor([trg_indexes[-1]], dtype=int, device=device)

        with no_grad():
            output, hidden = model.decoder(trg_tensor, hidden)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        pred_token = output.argmax(1).item()
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred_token == tgt_vocab[EOS_TOKEN]:
            break
        trg_indexes.append(pred_token)

    return tgt_vocab.to_tokens(trg_indexes)[1:]


def cal_bleu_score(net, data_path, src_vocab, tgt_vocab, seq_len, device):
    net.eval()
    raw_text = cht_to_chs(read_nmt(data_path))

    srcs, tgts = tokenize_nmt(process_nmt(raw_text))
    predicts = []
    labels = []

    for src, tgt in zip(srcs, tgts):
        predict = translate(net, src, src_vocab, tgt_vocab, seq_len, device)
        predicts.append(predict)
        labels.append([tgt])  # 因为reference可以有多条，所以是一个列表
        print(f"{tgt} → {predict}")

    return bleu_score(predicts, labels)


def init_weights(model):
    for name, param in model.named_parameters():
        init.uniform_(param, -0.08, 0.08)


# 参数定义
embed_size = 256
num_hiddens = 512
num_layers = 2
dropout = 0.5

batch_size = 128
max_len = 40

lr = 0.001
num_epochs = 100
min_freq = 1
clip = 1.0

tf_ratio = 0.5 # teacher force ratio

print_every = 1

device = cuda.get_device("cuda" if cuda.is_available() else "cpu")

# 加载训练集
train_iter, src_vocab, tgt_vocab = load_dataset_nmt('../data/en-cn/train_mini.txt', batch_size=batch_size,
                                                    min_freq=min_freq)

# 加载验证集
valid_iter, src_vocab, tgt_vocab = load_dataset_nmt('../data/en-cn/dev_mini.txt', batch_size=batch_size,
                                                    min_freq=min_freq, src_vocab=src_vocab, tgt_vocab=tgt_vocab)

# 构建编码器
encoder = NMTEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
# 构建解码器
decoder = NMTDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)

model = NMTModel(encoder, decoder, device)
model.apply(init_weights)
model.to(device)

optimizer = Adam(model.parameters())
criterion = CrossEntropyLoss(ignore_index=tgt_vocab[PAD_TOKEN])


def train(model, num_epochs, train_iter, valid_iter, optimizer, criterion, clip, device, tf_ratio):
    best_valid_loss = float('inf')
    for epoch in tqdm(range(1, num_epochs + 1),  desc="Training", leave=False):
        train_loss = train_epoch(model, train_iter, optimizer, criterion, clip, device, tf_ratio)
        valid_loss = evaluate(model, valid_iter, criterion, device)
        #if valid_loss < best_valid_loss:
        #    best_valid_loss = valid_loss
        #    model.save("nmt.pt")
        tqdm.write(f"epoch {epoch:3d} , train loss: {train_loss:.4f} , validate loss: {valid_loss:.4f}")

# with debug_mode():
train(model, num_epochs, train_iter, valid_iter, optimizer, criterion, clip, device, tf_ratio)


#model = model.load("nmt.pt")
print(model)
model.to(device)

import time

start = time.time()

score = cal_bleu_score(model, '../data/en-cn/train_mini.txt', src_vocab, tgt_vocab, max_len, device)
print(f"cost: {time.time() - start}")
print(f"bleu score: {score * 100:.2f}")
# 简单 单层 1.2
# 参数更多 单层 GRU 4.09
# 单层 GRU 逆序 2.13

model.save("nmt.pt")

# while True:
#    sentence = input("英文: ")
#    chinese, _ = inference(model, sentence, src_vocab, tgt_vocab, 20, device)
#    print(chinese)
