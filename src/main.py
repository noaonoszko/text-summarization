from tqdm import tqdm
from tqdm.auto import trange
from pathlib import Path

from datasets import load_dataset

import numpy as np
import matplotlib.pyplot as plt

import glove
import nltk
from rouge_score import rouge_scorer
import torch

from models import *
from common_utils import *


class SummarizerParameters:
    device = "cpu"

    random_seed = 0

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
    def __init__(self, params):
        self.params = params

    def train(self, train_data):

        p = self.params

        # Setting a fixed seed for reproducibility.
        torch.manual_seed(p.random_seed)
        random.seed(p.random_seed)

        # Build the encoder and decoder.
        encoder = Encoder(p)
        decoder = Decoder(p)
        self.model = EncoderDecoder(encoder, decoder)

        self.model.to(p.device)
        print(" done.")

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay
        )

        # The loss function is a cross-entropy loss at the token level.
        # We don't include padding tokens when computing the loss.
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.T_voc.get_pad_idx())

        for epoch in range(1, p.n_epochs + 1):

            t0 = time.time()

            loss_sum = 0
            for i, (Sbatch, Tbatch) in enumerate(train_loader, 1):

                # We use teacher forcing to train the decoder.
                # This means that the input at each decoding step will be the
                # *gold-standard* word at the previous position.
                # We create a tensor Tbatch_shifted that contains the previous words.
                batch_size, sen_len = Tbatch.shape
                zero_pad = torch.zeros(
                    batch_size, 1, dtype=torch.long, device=Tbatch.device
                )
                Tbatch_shifted = torch.cat([zero_pad, Tbatch[:, :-1]], dim=1)

                self.model.train()
                scores = self.model(Sbatch, Tbatch_shifted)
                if i == 1:
                    print("scores.shape:", scores.shape, "Tbatch.shape:", Tbatch.shape)
                    print(
                        "scores.view(-1, len(self.T_voc)).shape:",
                        scores.view(-1, len(self.T_voc)).shape,
                        "Tbatch.view(-1):",
                        Tbatch.view(-1).shape,
                    )
                    print("Tbatch", Tbatch)
                loss = loss_func(scores.view(-1, len(self.T_voc)), Tbatch.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

                print(".", end="")
                sys.stdout.flush()


# Required download for tokenization
nltk.download("punkt")

# Load cnn data
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_set = dataset["train"]


# summarizer = Summarizer(SummarizerParameters())
# summarizer.train(train_data=train_set)

# Load glove
p = SummarizerParameters()
glove = glove.Glove(
    glove_dir=str(Path(__file__).resolve().parents[3]) + "/Assignment 3/data/glove/"
)
wordvecs = glove.load_glove(p.emb_dim)
print("Done loading glove")

# Try out the encoder
a, h = train_loader(wordvecs, train_set)
encoder = Encoder(p)
enc_out = encoder(torch.unsqueeze(a, 0))
print(type(enc_out))
print(enc_out.shape)

# # Evaluate the baseline
# rouge_scores = evaluate(
#     data=train_set.select(range(len(train_set))), model=lead_3_baseline
# )
# print("Averages:", np.mean(rouge_scores, axis=0))
# rouge_hist(rouge_scores)

# Try out the inverse embedding
# print(emb_to_word(wordvecs["apple"])