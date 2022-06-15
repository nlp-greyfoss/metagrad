import metagrad.module as nn
from metagrad.tensor import Tensor

if __name__ == '__main__':
    lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
    #inputs = [Tensor.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

    # initialize the hidden state.
    # hidden = (Tensor.randn(1, 1, 3),
    #           Tensor.randn(1, 1, 3))
    # for i in inputs:
    #     # Step through the sequence one element at a time.
    #     # after each step, hidden contains the hidden state.
    #     i = i.reshape((1, 1, -1))
    #     out, hidden = lstm(i, hidden)
    #
    # print(out)
    # print(hidden)

    # alternatively, we can do the entire sequence all at once.
    # the first value returned by LSTM is all of the hidden states throughout
    # the sequence. the second is just the most recent hidden state
    # (compare the last slice of "out" with "hidden" below, they are the same)
    # The reason for this is that:
    # "out" will give you access to all hidden states in the sequence
    # "hidden" will allow you to continue the sequence and backpropagate,
    # by passing it as an argument  to the lstm at a later time
    # Add the extra 2nd dimension
    inputs = Tensor.randn(5, 1, 3)
    hidden = (Tensor.randn(1, 1, 3), Tensor.randn(1, 1, 3))  # clean out hidden state
    out, hidden = lstm(inputs, hidden)
    print(out)
    print(hidden)
