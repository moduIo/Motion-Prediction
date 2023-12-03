# Copyright (c) Facebook, Inc. and its affiliates.

import torch.nn as nn

class Seq2Seq(nn.Module):
    """Seq2Seq model for sequence generation. The interface takes predefined
    encoder and decoder as input.

    Attributes:
        encoder: Pre-built encoder
        decoder: Pre-built decoder
    """

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, src, tgt, max_len=None):
        """
        Inputs:
            src: Source sequence provided as input to the encoder.
                Expected shape: (batch_size, seq_len, input_dim)
            tgt: Target sequence provided as input to the decoder. During
                training, provide reference target sequence. For inference,
                provide only last frame of source.
                Expected shape: (batch_size, seq_len, input_dim)
            max_len: Optional; Length of sequence to be generated. By default,
                the decoder generates sequence with same length as `tgt`
                (training).
        """
        hidden, cell, outputs = self.encoder(src)
        outputs = self.decoder(tgt, hidden, cell, max_len, outputs)
        return outputs