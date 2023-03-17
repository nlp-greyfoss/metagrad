import os
import random
from string import ascii_letters

import metagrad.module as nn
from metagrad.tensor import Tensor, cuda, debug_mode

import metagrad.functions as F
from unidecode import unidecode
import numpy as np
from metagrad.optim import Adam, SGD
from metagrad.loss import CrossEntropyLoss
from tqdm import tqdm

device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")

data_dir = "../data/rnn/names"

lang2label = {
    file_name.split(".")[0]: Tensor([i], dtype=np.int32, device=device)
    for i, file_name in enumerate(os.listdir(data_dir))
}

num_langs = len(lang2label)

char2idx = {letter: i for i, letter in enumerate(ascii_letters + " .,:;-'")}
num_letters = len(char2idx)
print(num_letters)


def name2tensor(name):
    tensor = Tensor.zeros((len(name), 1, num_letters), device=device)
    for i, char in enumerate(name):
        tensor[i, 0, char2idx[char]] = 1
    return tensor


# print(name2tensor("abc"))

tensor_names = []
target_langs = []

for file in os.listdir(data_dir):
    with open(os.path.join(data_dir, file), encoding="utf-8") as f:
        lang = file.split(".")[0]
        names = [unidecode(line.rstrip()) for line in f]
        for name in names:
            try:
                tensor_names.append(name2tensor(name))
                target_langs.append(lang2label[lang])
            except KeyError:
                pass

from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(
    range(len(target_langs)),
    test_size=0.1,
    shuffle=True,
    stratify=target_langs
)

train_dataset = [
    (tensor_names[i], target_langs[i])
    for i in train_idx
]

test_dataset = [
    (tensor_names[i], target_langs[i])
    for i in test_idx
]


class GRUModel(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(GRUModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=num_letters,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.fc = nn.Linear(hidden_size, num_langs)

    def forward(self, x):
        hidden_state = self.init_hidden()
        output, hidden_state = self.gru(x, hidden_state)
        output = self.fc(output[-1])
        return output

    def init_hidden(self):
        return Tensor.zeros((self.num_layers, 1, self.hidden_size)).to(device)


hidden_size = 256
learning_rate = 0.001
num_epochs = 2
print_interval = 3000

model = GRUModel(num_layers=2, hidden_size=hidden_size)
model.to(device)

optimizer = SGD(model.parameters(), lr=learning_rate)
criterion = CrossEntropyLoss()

import time

start = time.time()

for epoch in tqdm(range(num_epochs)):
    random.shuffle(train_dataset)
    for i, (name, label) in enumerate(train_dataset):
        output = model(name)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % print_interval == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Step [{i + 1}/{len(train_dataset)}], "
                f"Loss: {loss.item():.4f}"
            )

print(f"Training cost {time.time() - start}s")

num_correct = 0
num_samples = len(test_dataset)

model.eval()
model.save("simple.pt")
# model.load("simple.pt")
model.to_gpu(device)

for name, label in test_dataset:
    name, label = name.to(device), label.to(device)
    output = model(name)
    pred = output.argmax(axis=1)
    num_correct += int(pred.item() == label.item())

print(f"Accuracy: {num_correct / num_samples * 100:.4f}%")
