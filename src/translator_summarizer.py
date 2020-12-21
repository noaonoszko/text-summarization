import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import time
import numpy as np
import sys

import random

import matplotlib.pyplot as plt
from common_utils import *


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
    def __init__(self, params, wordvecs):
        super().__init__()
        self.params = params
        self.wordvecs = wordvecs
        self.rnn = nn.GRU(
            params.emb_dim,
            params.dec_rnn_dim,
            params.dec_rnn_depth,
            batch_first=True,
            bidirectional=False,
        )
        self.output_layer = nn.Linear(
            params.dec_rnn_dim + params.emb_dim, params.vocab_size, bias=False
        )
        self.rnn_dim = params.dec_rnn_dim

    def forward_step(self, prev_embed, enc_out, src_mask, decoder_hidden):
        rnn_input = prev_embed
        rnn_output, decoder_hidden = self.rnn(rnn_input, decoder_hidden)
        pre_output = torch.cat([prev_embed, rnn_output], dim=2)
        T_output = self.output_layer(pre_output)
        return decoder_hidden, T_output

    # Apply the decoder in teacher forcing mode.
    def forward(self, enc_out, src_mask, H_shifted):

        n_sen, n_words = H_shifted.shape

        # Initialize the hidden state of the GRU.
        decoder_hidden = torch.zeros(1, n_sen, self.rnn_dim, device=src_mask.device)

        all_out = []

        # For each position in the target sentence:
        for i in range(n_words):
            
            # Embedding for the previous word.
            # print("H_shifted[:, i]:", H_shifted[:, i], type(H_shifted[0, i]))
            # print("words_to_embs(self.wordvecs, H_shifted[:, i]):", words_to_embs(self.wordvecs, list(H_shifted[:, i])))
            # print(type(H_shifted[:, i]))
            H_word_emb = torch.tensor(words_to_embs(self.wordvecs, list(H_shifted[:, i])), device=self.params.device)
            prev_embed = H_word_emb.unsqueeze(1)

            # Run the decoder one step.
            # This returns a new hidden state, and the output
            # scores (over the target vocabulary) at this position.
            decoder_hidden, H_output = self.forward_step(
                prev_embed, enc_out, src_mask, decoder_hidden
            )
            all_out.append(H_output)

        # Combine the output scores for all positions.
        return torch.cat(all_out, dim=1)


class SummarizerParameters:
    device = "cuda"

    random_seed = 0
    vocab_size = 400000

    n_epochs = 30

    batch_size = 128

    learning_rate = 5e-4
    weight_decay = 0

    emb_dim = 50
    enc_rnn_dim = 512
    enc_rnn_depth = 1

    dec_rnn_dim = 512
    dec_rnn_depth = 1

    max_train_sentences = 20000
    max_valid_sentences = 1000


class Summarizer:
    def __init__(self, params, wordvecs, word_int_dict):
        self.params = params
        self.wordvecs = wordvecs
        self.word_int_dict = word_int_dict

    def train(self, train_set):

        p = self.params

        # Setting a fixed seed for reproducibility.
        torch.manual_seed(p.random_seed)
        random.seed(p.random_seed)

        # Build the encoder and decoder.
        encoder = Encoder(p)
        decoder = Decoder(p, self.wordvecs)
        self.model = EncoderDecoder(encoder, decoder)

        self.model.to(p.device)
        print(" done.")

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay
        )

        # The loss function is a cross-entropy loss at the token level.
        # We don't include padding tokens when computing the loss.
        loss_func = torch.nn.CrossEntropyLoss()
        
        t = trange(1, p.n_epochs + 1)
        for epoch in t:
            t0 = time.time()

            loss_sum = 0
            for i, (Abatch, Hbatch) in enumerate(train_loader(self.wordvecs, self.word_int_dict, train_set), 1):
                # We use teacher forcing to train the decoder.
                # This means that the input at each decoding step will be the
                # *gold-standard* word at the previous position.
                # We create a tensor Hbatch_shifted that contains the previous words.
                batch_size, sen_len = Hbatch.shape
                zero_pad = np.array(["" for i in range(Hbatch.shape[0])])
                zero_pad = np.expand_dims(zero_pad, 1)
                Hbatch_shifted = np.concatenate([zero_pad, Hbatch[:, :-1]], axis=1)
                self.model.train()
                scores = self.model(Abatch.to(device=p.device), Hbatch_shifted)
                
                # Convert highlight words to ints
                Hbatch_int = torch.zeros(Hbatch.shape, device=self.params.device)
                for batch in range(Hbatch.shape[0]):
                    Hbatch_int[batch] = words_to_ints(word_int_dict=self.word_int_dict, words=Hbatch[batch])
                Hbatch = Hbatch_int
                loss = loss_func(scores.view(-1, p.vocab_size), Hbatch.view(-1).type(torch.LongTensor).to(device=self.params.device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

                t.set_description("Epoch {}".format(epoch))
                t.set_postfix(
                    loss=loss.item()
                )