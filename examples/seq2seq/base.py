from typing import Tuple

import metagrad.module as nn
from metagrad import Tensor


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X) -> Tensor:
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs):
        raise NotImplementedError

    def forward(self, X, state) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
