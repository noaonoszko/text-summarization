import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import time
import numpy as np
import sys

import random

import matplotlib.pyplot as plt


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, A, H_shifted):
        A_enc = self.encoder(A)
        A_mask = A != 0
        return self.decoder(A_enc, A_mask, H_shifted)


class Encoder(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.rnn = nn.GRU(
            param.emb_dim,
            param.enc_rnn_dim,
            param.enc_rnn_depth,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, article):
        output, _ = self.rnn(article)
        return output


class Decoder(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.rnn = nn.GRU(
            param.emb_dim,
            param.dec_rnn_dim,
            param.dec_rnn_depth,
            batch_first=True,
            bidirectional=False,
        )
        voc_size = 400000
        self.output_layer = nn.Linear(
            param.dec_rnn_dim + param.emb_dim, voc_size, bias=False
        )
        self.rnn_dim = param.dec_rnn_dim

    def forward_step(self, prev_embed, enc_out, src_mask, decoder_hidden):
        rnn_input = prev_embed
        rnn_output, decoder_hidden = self.rnn(rnn_input, decoder_hidden)
        pre_output = torch.cat([prev_embed, rnn_output], dim=2)
        T_output = self.output_layer(pre_output)
        return decoder_hidden, T_output

    # Apply the decoder in teacher forcing mode.
    def forward(self, enc_out, src_mask, H_shifted):

        n_sen, n_words, _ = H_shifted.shape

        # Initialize the hidden state of the GRU.
        decoder_hidden = torch.zeros(1, n_sen, self.rnn_dim, device=src_mask.device)

        all_out = []

        # For each position in the target sentence:
        for i in range(n_words):

            # Embedding for the previous word.
            prev_embed = T_emb[:, i].unsqueeze(1)

            # Run the decoder one step.
            # This returns a new hidden state, and the output
            # scores (over the target vocabulary) at this position.
            _, decoder_hidden, T_output = self.forward_step(
                prev_embed, enc_out, src_mask, decoder_hidden
            )
            all_out.append(T_output)

        # Combine the output scores for all positions.
        return torch.cat(all_out, dim=1)