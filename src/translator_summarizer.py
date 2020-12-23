import torch
from torch import nn
import torch.nn.functional as F
# from torch.utils.train_data import train_DataLoader

import time
import numpy as np
import sys

import random

import matplotlib.pyplot as plt
from common_utils import *

class BahdanauAttention(nn.Module):
   
    def __init__(self, hidden_size, key_size, query_size):
        super().__init__()
       
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
       
    def precompute_key(self, encoder_out):
        # encoder_out shape: (n_sentences, n_src_words, encoder_rnn_size)
        return self.key_layer(encoder_out)
   
    def forward(self, query, precomputed_key, value, mask):
        # query shape: (n_sentences, 1, decoder_rnn_size)
        # precomputed_key shape: (n_sentences, n_src_words, hidden_size)

        query = self.query_layer(query)
           
        # Calculate scores.
        scores = self.energy_layer(query + precomputed_key).squeeze(2)

        #####
       
        # scores.data.masked_fill_(~mask, -float('inf'))
               
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=1)       
       
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas.unsqueeze(1), value)
       
        return context, alphas       
       
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
        attn_dim = 2*params.enc_rnn_dim
        self.rnn = nn.GRU(
            params.emb_dim+attn_dim,
            params.dec_rnn_dim,
            params.dec_rnn_depth,
            batch_first=True,
            bidirectional=False,
        )
        self.output_layer = nn.Linear(
            params.dec_rnn_dim + attn_dim + params.emb_dim, params.vocab_size, bias=False
        )
        self.rnn_dim = params.dec_rnn_dim
        self.attention = BahdanauAttention(hidden_size=attn_dim,
                                           query_size=params.dec_rnn_dim,
                                           key_size=2*params.enc_rnn_dim)

    def forward_step(self, prev_embed, enc_out, src_mask, precomputed_key, decoder_hidden):
        # The "query" for attention is the hidden state of the decoder.
        query = decoder_hidden[-1].unsqueeze(1)  
        
        # Apply the attention model to compute a "context", summary of the encoder output
        # based on the current query.
        # Also returns the attention weights, which we might want to visualize or inspect.
        # print("query.shape, precomputed_key.shape, enc_out.shape, src_mask.shape:\n", query.shape, precomputed_key.shape, enc_out.shape, src_mask.shape, "\n")
        context, attn_probs = self.attention(
            query=query, precomputed_key=precomputed_key,
            value=enc_out, mask=src_mask)
        
        # Feed through rnn
        rnn_input = torch.cat([prev_embed, context], dim=2)
        rnn_output, decoder_hidden = self.rnn(rnn_input, decoder_hidden)

        # Feed through output layer
        pre_output = torch.cat([prev_embed, rnn_output, context], dim=2)
        T_output = self.output_layer(pre_output)
        return decoder_hidden, T_output

    # Apply the decoder in teacher forcing mode.
    def forward(self, enc_out, src_mask, H_shifted):

        n_sen, n_words = H_shifted.shape

        # Initialize the hidden state of the GRU.
        decoder_hidden = torch.zeros(1, n_sen, self.rnn_dim, device=src_mask.device)

        all_out = []

        # For each position in the target sentence:
        for b in range(n_words):
            
            # Embedding for the previous word.
            H_word_emb = words_to_embs(self.wordvecs, list(H_shifted[:, b])).to(self.params.device)
            prev_embed = H_word_emb.unsqueeze(1)

            # Precompute attention keys (if needed).
            precomputed_key = self.attention.precompute_key(enc_out)

            # Run the decoder one step.
            # This returns a new hidden state, and the output
            # scores (over the target vocabulary) at this position.
            decoder_hidden, H_output = self.forward_step(
                prev_embed, enc_out, src_mask, precomputed_key, decoder_hidden
            )
            all_out.append(H_output)

        # Combine the output scores for all positions.
        return torch.cat(all_out, dim=1)


class SummarizerParameters:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random_seed = 0
    vocab_size = 400000


    batch_size = 128

    learning_rate = 1e-1
    weight_decay = 0

    emb_dim = 50
    enc_rnn_dim = 512
    enc_rnn_depth = 1

    dec_rnn_dim = 512
    dec_rnn_depth = 1


class Summarizer:
    def __init__(self, params, wordvecs, word_int_dict):
        self.params = params
        self.wordvecs = wordvecs
        self.word_int_dict = word_int_dict
    
    def validate(self, data, generate=False):
        p = self.params
        self.model.eval()
        for b, (Abatch, Hbatch, lengths) in enumerate(train_loader(self.wordvecs, self.word_int_dict, data, batch_size=3)):
            batch_size, sen_len = Hbatch.shape
            zero_pad = np.array(["" for i in range(Hbatch.shape[0])])
            zero_pad = np.expand_dims(zero_pad, 1)
            Hbatch_shifted = np.concatenate([zero_pad, Hbatch[:, :-1]], axis=1)
            scores = self.model(Abatch.to(device=p.device), Hbatch_shifted)
            loss_func = torch.nn.CrossEntropyLoss()
            
            # Convert highlight words to ints
            Hbatch_int = torch.zeros(Hbatch.shape, device=self.params.device)
            for batch in range(Hbatch.shape[0]):
                Hbatch_int[batch] = words_to_ints(word_int_dict=self.word_int_dict, words=Hbatch[batch])
            val_loss = loss_func(scores.view(-1, p.vocab_size), Hbatch_int.view(-1).type(torch.LongTensor).to(device=self.params.device))
            if generate and b < 5:
                print("\n\ni =", b)
                print("----------------------------output----------------------------")
                for w in range(Hbatch.shape[1]):
                    best_word_int = torch.argmax(scores[b,w])
                    print(list(self.wordvecs.keys())[best_word_int.item()], end=" ")
                print("\n----------------------------target----------------------------")
                for w in range(Hbatch.shape[1]):
                    print(Hbatch[b,w], end=" ")        
            return val_loss

    def train(self, train_data, val_data=False, n_epochs=200, val_every=50, generate_every=50):
        p = self.params

        # Setting a fixed seed for reproducibility.
        torch.manual_seed(p.random_seed)
        random.seed(p.random_seed)

        # Build the encoder and decoder.
        encoder = Encoder(p)
        decoder = Decoder(p, self.wordvecs)
        self.model = EncoderDecoder(encoder, decoder)
        self.model.to(p.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay
        )

        # The loss function is a cross-entropy loss at the token level.
        # We don't include padding tokens when computing the loss.
        loss_func = torch.nn.CrossEntropyLoss()
        
        train_losses = torch.zeros(n_epochs, requires_grad=False)
        val_losses = torch.zeros(n_epochs, requires_grad=False)
        t = trange(1, n_epochs + 1)
        for epoch in t:
            # torch.cuda.empty_cache()
            for b, (Abatch, Hbatch) in enumerate(train_loader(self.wordvecs, self.word_int_dict, train_data)):
                self.model.train()
                batch_size, sen_len = Hbatch.shape
                zero_pad = np.array(["" for i in range(Hbatch.shape[0])])
                zero_pad = np.expand_dims(zero_pad, 1)
                Hbatch_shifted = np.concatenate([zero_pad, Hbatch[:, :-1]], axis=1)
                scores = self.model(Abatch.to(device=p.device), Hbatch_shifted)
                
                # Convert highlight words to ints
                Hbatch_int = torch.zeros(Hbatch.shape, device=self.params.device)
                for batch in range(Hbatch.shape[0]):
                    Hbatch_int[batch] = words_to_ints(word_int_dict=self.word_int_dict, words=Hbatch[batch])
                Hbatch = Hbatch_int
                
                # Backprop
                loss = loss_func(scores.view(-1, p.vocab_size), Hbatch.view(-1).type(torch.LongTensor).to(device=self.params.device))
                train_losses[epoch-1] = loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Validate
                if epoch % val_every == 0 and b == 0:
                    generate = True if epoch % generate_every == 0 or epoch == n_epochs else False
                    val_loss = self.validate(val_data, generate)
                    val_losses[epoch-1] = val_loss
                    
                    # Save loss plot
                    if generate:
                        plt.plot(train_losses.detach())
                        plt.plot(val_losses.detach())
                        plt.savefig("loss.png")

                    t.set_postfix(
                        loss=loss.item(), val_loss=val_loss.item()
                    )

                t.set_description("Epoch {}".format(epoch))
                sys.stdout.flush()
            del Hbatch_shifted, Hbatch_int