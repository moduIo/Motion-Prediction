# Copyright (c) Facebook, Inc. and its affiliates.

import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(
        self, input_dim=None, hidden_dim=1024, num_layers=1):
        """LSTMEncoder encodes input vector using LSTM cells.

        Attributes:
            input_dim: Size of input vector
            hidden_dim: Size of hidden state vector
            num_layers: Number of layers of LSTM units

        """
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,num_layers=num_layers,batch_first=True)

    def forward(self, input):
        """
        Input:
            input: Input vector to be encoded.
                Expected shape is (batch_size, seq_len, input_dim)
        """
        outputs, (lstm_hidden, lstm_cell) = self.lstm(input)
        return lstm_hidden, lstm_cell, outputs
