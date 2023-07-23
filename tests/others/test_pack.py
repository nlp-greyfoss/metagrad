import metagrad.module as nn
from metagrad.tensor import Tensor
from metagrad.rnn_utils import pack_padded_sequence, pad_packed_sequence
from metagrad.utils import pad_sequence


def test_pack_padded_sequence():
    sentences = ['crazy', 'complicate', 'medium', 'hard', 'rookie']
    # 词典
    vocab = ['<pad>'] + sorted(set([char for sentence in sentences for char in sentence]))
    # 转换成数值
    vectorized = [Tensor([vocab.index(t) for t in seq]) for seq in sentences]
    # 每个序列的长度
    lengths = Tensor(list(map(len, vectorized)))
    print(lengths)

    padded_seq = pad_sequence(vectorized)

    print(padded_seq)

    sorted_lengths, sorted_indices = lengths.sort(0, descending=True)

    padded_seq = padded_seq[sorted_indices]

    print(padded_seq)

    embed = nn.Embedding(len(vocab), 3)  # embedding_dim = 3
    embedded_seq = embed(padded_seq)
    print(embedded_seq)  # (batch_size, seq_len, embed_size)

    result = pack_padded_sequence(embedded_seq, sorted_lengths.tolist(), batch_first=True)

    rnn = nn.RNN(input_size=3, hidden_size=4, batch_first=True, bidirectional=True)

    out, _ = rnn(result)

    print(out)
    print(out.data.shape)
    print(out.batch_sizes)
    output, input_sizes = pad_packed_sequence(out, batch_first=True)

    print(output.shape)
    print(input_sizes)
    print(output)
