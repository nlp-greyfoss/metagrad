import random
from typing import Tuple

from examples.embeddings.utils import BOS_TOKEN, PAD_TOKEN
from examples.seq2seq.base import Encoder, Decoder
import metagrad.module as nn
from examples.seq2seq.dataset import load_dataset_nmt
from metagrad import Tensor, cuda
import metagrad.functions as F
from metagrad.optim import Adam, SGD
from metagrad.tensor import no_grad, debug_mode
from metagrad.utils import Animator, Accumulator, Timer, clip_grad_norm_
from tqdm import tqdm


class RNNEncoder(Encoder):
    '''用RNN实现编码器'''

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=.0, **kwargs):
        super(RNNEncoder, self).__init__(**kwargs)
        # 嵌入层 获取输入序列中每个单词的嵌入向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 基于单向GRU实现
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X) -> Tensor:
        '''

        Args:
            X:  形状 (batch_size, num_steps)
        Returns:

        '''

        X = self.embedding(X)  # X 的形状 (batch_size, num_steps, embed_size)
        X = X.permute((1, 0, 2))  # (num_steps, batch_size, embed_size)
        output, state = self.rnn(X)
        # num_directions = 1 todo 可以试试num_directions=2
        # output (num_steps, batch_size, num_hiddens * num_directions)
        # state (num_layers * num_directions, batch_size, num_hiddens)
        return output, state


# encoder = RNNEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
# encoder.eval()
# X = Tensor.zeros((4, 7), dtype=numpy.int32)
# output, state = encoder(X)
#
#
# print(output.shape)


class RNNDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=.0, **kwargs):
        super(RNNDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # embed_size + num_hiddens 为了处理拼接后的维度，见forward函数中的注释
        self.rnn = nn.RNN(embed_size + num_hiddens, num_hiddens, num_layers, batch_first=False, dropout=dropout)
        # 将隐状态转换为词典大小维度
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self.vocab_size = vocab_size

    def init_state(self, enc_outputs):
        return enc_outputs[1]  # 得到编码器输出中的state

    def forward(self, X, state) -> Tuple[Tensor, Tensor]:
        '''

        Args:
            X: (batch_size)
            state:

        Returns:

        '''
        X = X.unsqueeze(0)  # (1, batch_size)
        X = self.embedding(X)  # (1, batch_size, embed_size)

        # context 形状 (1, batch_size, num_hiddens )
        context = state[-1].unsqueeze(0)
        # 为了每个解码时间步都能看到上下文，拼接context与X
        # (1, batch_size, embed_size) + (1, batch_size, num_hiddens)
        #                           => (1, batch_size, embed_size + num_hiddens)
        concat_context = F.cat((X, context), 2)

        output, state = self.rnn(concat_context, state)

        output = F.softmax(self.dense(output.squeeze(0)), axis=1)

        return output, state


# decoder = RNNDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
# decoder.eval()
# state = decoder.init_state(encoder(X))
# output, state = decoder(X, state)
# print(output.shape, state.shape)


class Seq2Seq(nn.Module):
    '''合并编码器和解码器'''

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, tf_ratio=0.5) -> Tensor:
        '''

        Args:
            src: (batch_size, num_steps)
            tgt: (batch_size, num_steps)
            tf_ratio: teacher forcing 阈值

        Returns:

        '''
        enc_outputs = self.encoder(src)
        hidden = self.decoder.init_state(enc_outputs)  # (num_layers  * num_directions, batch_size, num_hiddens)

        batch_size, num_steps = tgt.shape

        outputs = []

        inp = Tensor([tgt_vocab[BOS_TOKEN]] * batch_size, device=self.device)  # bos
        for t in range(1, num_steps):
            output, hidden = self.decoder(inp, hidden)

            outputs.append(output)
            tf = random.random() < tf_ratio
            pred = output.argmax(axis=1)
            inp = tgt[:, t] if tf else pred

        outputs = F.stack(outputs, axis=0)
        return outputs


def nll_mask_loss(inp, target, mask):
    num_total = mask.sum()
    ce_loss = -inp.gather(1, target.view(-1, 1)).squeeze(1).log()
    loss = F.masked_select(ce_loss, mask).sum() / len(ce_loss)
    return loss, num_total.item()


def batch_2_train(batch, padding_value):
    input_batch, output_batch = batch

    mask = output_batch != padding_value

    return input_batch, output_batch, mask


def dataloader_stream(padding_value, dataloader):
    while True:
        # 每次重新构造一个迭代器，保证一直会有数据
        it = iter(dataloader)
        batch = next(it)
        yield batch_2_train(batch, padding_value)


def train(input, target, mask, encoder, decoder, encoder_optimizer, decoder_optimizer,
          device, tgt_vocab, teacher_forcing_ratio):
    """
    在单个批次上的训练
    Returns:

    """
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input, target, mask = input.to(device), target.to(device), mask.to(device)

    print_losses = []

    num_totals = 0  # 单词数

    loss = 0

    enc_outputs = encoder(input)

    hidden = decoder.init_state(enc_outputs)

    batch_size, num_steps = target.shape

    # 创建初始的解码器输入，初始时输入BOS_TOKEN
    inp = Tensor([tgt_vocab[BOS_TOKEN]] * batch_size, device=device)  # bos

    # 是否开启teacher forcing; 若开启，则用真实输出来预测下一个输出，否则用预测的输出来预测下一个输出。
    use_tf = True if random.random() < teacher_forcing_ratio else False

    if use_tf:
        for t in range(num_steps):  # 最多生成num_steps
            output, hidden = decoder(inp, hidden)
            # 使用目标单词作为下一个输入
            inp = target[:, t]
            mask_loss, num_total = nll_mask_loss(output, target[:, t], mask[:, t])
            loss += mask_loss

            print_losses.append(mask_loss.item() * num_total)

            num_totals += num_total
    else:
        for t in range(num_steps):
            output, hidden = decoder(inp, hidden)
            inp = output.argmax(axis=1)

            mask_loss, num_total = nll_mask_loss(output, target[:, t], mask[:, t])

            loss += mask_loss
            print_losses.append(mask_loss.item() * num_total)
            num_totals += num_total

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / num_totals


def train_seq2seq(encoder, decoder, data_iter, lr, n_iteration, tgt_vocab, device, teacher_forcing_ratio, print_every):
    """训练序列到序列模型"""

    encoder.train()
    decoder.train()

    encoder_optimizer = Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = Adam(decoder.parameters(), lr=lr)

    print_loss = 0

    print('Beginning Training...')

    # animator = Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])

    PAD_IDX = tgt_vocab[PAD_TOKEN]

    for iteration in tqdm(range(1, n_iteration + 1)):
        input_batch, output_batch, mask = next(dataloader_stream(PAD_IDX, data_iter))

        loss = train(input_batch, output_batch, mask, encoder, decoder, encoder_optimizer, decoder_optimizer, device,
                     tgt_vocab, teacher_forcing_ratio)

        print_loss += loss

        # 打印训练信息
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every

            print(
                f"(Iteration: {iteration}/{n_iteration} {iteration / n_iteration * 100 :.1f}% "
                f"Loss: {print_loss_avg:.4f})")
            print_loss = 0


def predict_seq2seq(encoder, decoder, data_iter, src_vocab, tgt_vocab, num_steps, device):
    encoder.eval()
    decoder.eval()

    output = []
    with no_grad():
        for batch in data_iter:
            output_seq = []

            X, Y = [x.to(device) for x in batch]

            enc_outputs = encoder(X)
            hidden = decoder.init_state(enc_outputs)  # (num_layers  * num_directions, batch_size, num_hiddens)

            inp = Tensor([tgt_vocab[BOS_TOKEN]], device=device)
            for t in range(1, num_steps):
                hat_y, hidden = decoder(inp, hidden)

                inp = hat_y.argmax(axis=1)
                pred = inp.squeeze(axis=0).item()

                if pred == tgt_vocab['<eos>']:
                    break
                output_seq.append(pred)

            output.append(
                f" {' '.join(src_vocab.token(X.squeeze().data.tolist()))} => {''.join(tgt_vocab.token(output_seq))}")

    return output


# 参数定义
embed_size = 512
num_hiddens = 512
num_layers = 4
dropout = 0.15

batch_size = 64
num_steps = 20

lr = 0.0001
n_iters = 10000
min_freq = 1

print_every = 100

teacher_forcing_ratio = 0.5

device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")

# 加载数据集
train_iter, src_vocab, tgt_vocab = load_dataset_nmt('../data/en-cn/train.txt', batch_size=batch_size,
                                                    min_freq=min_freq, max_len=num_steps)

# 构建编码器
encoder = RNNEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
# 构建解码器
decoder = RNNDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)

encoder.to(device)
decoder.to(device)

# 训练
train_seq2seq(encoder, decoder, train_iter, lr, n_iters, tgt_vocab, device, teacher_forcing_ratio, print_every)

test_iter, _, _ = load_dataset_nmt('../data/en-cn/test_mini.txt', batch_size=1, min_freq=min_freq, max_len=num_steps,
                                   src_vocab=src_vocab, tgt_vocab=tgt_vocab, shuffle=False)

output = predict_seq2seq(encoder, decoder, test_iter, src_vocab, tgt_vocab, num_steps=num_steps, device=device)
encoder.save("encoder.pt")
decoder.save("decoder.pt")
print(output)
