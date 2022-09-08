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


class EncoderDecoder(nn.Module):
    '''合并编码器和解码器'''

    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X) -> Tensor:
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder(dec_X, dec_state)
