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

from translator_summarizer import *
from common_utils import *

# Required download for tokenization
nltk.download("punkt")

# Load cnn data
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_set = dataset["train"]


# summarizer = Summarizer(SummarizerParameters())
# summarizer.train(train_data=train_set)

# Load glove
p = SummarizerParameters()
# glove_path = str(Path(__file__).resolve().parents[3]) + "/Assignment 3/data/glove/"
glove_path = str(Path(__file__).resolve().parents[2]) + "/glove/"
glove = glove.Glove(
    glove_dir=glove_path
)
vocab_size = 100
wordvecs = glove.load_glove(p.emb_dim, vocab_size=vocab_size)
print("Done loading glove")
word_int_dict = {}
for w, word in enumerate(wordvecs):
    word_int_dict[word] = w

# # Try out the encoder
# a, h = train_loader(wordvecs, train_set)
# encoder = Encoder(p)
# enc_out = encoder(torch.unsqueeze(a, 0))
# print(type(enc_out))
# print(enc_out.shape)

# Try out the full network
summarizer = Summarizer(p, wordvecs=wordvecs, word_int_dict=word_int_dict)
summarizer.train(train_set=train_set)

# # Evaluate the baseline
# rouge_scores = evaluate(
#     data=train_set.select(range(len(train_set))), model=lead_3_baseline
# )
# print("Averages:", np.mean(rouge_scores, axis=0))
# rouge_hist(rouge_scores)

# Try out the inverse embedding
# print(emb_to_word(wordvecs["apple"])