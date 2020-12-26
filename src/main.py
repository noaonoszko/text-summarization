from tqdm import tqdm
from tqdm.auto import trange
from pathlib import Path
import sys
import argparse

from datasets import load_dataset

import numpy as np
import matplotlib.pyplot as plt

import glove
import nltk
from rouge_score import rouge_scorer
import torch

from translator_summarizer import *
from rl_summarizer import *
from common_utils import *

# Required download for tokenization
nltk.download("punkt")

# Load cnn data
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_set = dataset["train"]
val_set = dataset["validation"]
test_set = dataset["test"]

# Load glove
param = SummarizerParameters()
# glove_path = str(Path(__file__).resolve().parents[3]) + "/Assignment 3/data/glove/"
glove_path = str(Path(__file__).resolve().parents[2]) + "/glove/"
glove = glove.Glove(
    glove_dir=glove_path
)
wordvecs = glove.load_glove(param.word_emb_dim, vocab_size=param.vocab_size)
print("Done loading glove")
word_int_dict = {}
for w, word in enumerate(wordvecs):
    word_int_dict[word] = w

# Argparsers
parser = argparse.ArgumentParser()
parser.add_argument(
    "--epochs",
    help="Number of epochs",
    type=int,
    default=200,
)
parser.add_argument(
    "--n_train",
    help="Number of training examples",
    type=int,
    default=50,
)
parser.add_argument(
    "--val_every",
    help="Validation frequency",
    type=int,
    default=1,
)
parser.add_argument(
    "--generate_every",
    help="How often to generate a sentence",
    type=int,
    default=10,
)
args = parser.parse_args()

# Use less training data
n_epochs = args.epochs
n_train = args.n_train
n_val = int(n_train/10)
train_subset = train_set.select(range(n_train))
val_subset = val_set.select(range(n_val))

# Train and validate the RL summarizer
summarizer = Summarizer(param)
summarizer.train(train_data=train_subset, val_data=val_subset, val_every=args.val_every, n_epochs=args.epochs, generate_every=args.generate_every)

# Train and validate the translator summarizer
# summarizer = Summarizer(param, wordvecs=wordvecs, word_int_dict=word_int_dict)
# summarizer.train(train_data=train_subset, val_data=val_subset, val_every=args.val_every, n_epochs=args.epochs, generate_every=args.generate_every)


# # Evaluate the baseline
# rouge_scores = evaluate(
#     data=train_set.select(range(len(train_set))), model=lead_3_baseline
# )
# print("Averages:", np.mean(rouge_scores, axis=0))
# rouge_hist(rouge_scores)

# Try out the inverse embedding
# print(emb_to_word(wordvecs["apple"])