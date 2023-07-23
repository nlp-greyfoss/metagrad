import metagrad.module as nn
from metagrad import cuda


def test_save_load():
    device = cuda.get_device("cuda" if cuda.is_available() else "cpu")

    model = nn.Linear(3, 2)
    model.to(device)

    print(list(model.parameters()))

    model.save("test_model.pkl")

    new_model = nn.Linear(3, 2)
    new_model.load("test_model.pkl")

    print(list(new_model.parameters()))

    assert model.weight.tolist() == new_model.weight.tolist()
