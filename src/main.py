from tqdm import tqdm
from tqdm.auto import trange
from pathlib import Path
import sys
import argparse

from datasets import load_dataset, logging

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
parser.add_argument( #add --use_combinations as a flag to use combinations
    "--use_combinations",
    help="Consider combinations of the p best sentences and choose the param.n_sent best ones according to rouge scores.",
    action="store_true",
)
args = parser.parse_args()

# Load cnn data
logging.set_verbosity(logging.ERROR)
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_set = dataset["train"]
val_set = dataset["validation"]
test_set = dataset["test"]

# Load glove
param = SummarizerParameters()
param.use_combinations = args.use_combinations
# glove_path = str(Path(__file__).resolve().parents[3]) + "/Assignment 3/data/glove/"
# glove_path = str(Path(__file__).resolve().parents[2]) + "/glove/"
# glove = glove.Glove(
#     glove_dir=glove_path
# )
# wordvecs = glove.load_glove(param.word_emb_dim, vocab_size=param.vocab_size)
# print("Done loading glove")
# word_int_dict = {}
# for w, word in enumerate(wordvecs):
#     word_int_dict[word] = w


# Use less training data
print("before preprocessing")
n_epochs = args.epochs
n_train = args.n_train
n_val = int(n_train/10)
train_subset = train_set.select(range(4*n_train)).filter(lambda example: len(nltk.tokenize.sent_tokenize(example["article"])) >= param.n_sent)
train_subset = train_subset.select(range(n_train))
val_subset = val_set.select(range(4*n_val)).filter(lambda example: len(nltk.tokenize.sent_tokenize(example["article"])) >= param.n_sent)
val_subset = val_subset.select(range(n_val))
print("after preprocessing")

# Train and validate the RL summarizer
summarizer = Summarizer(param)
summarizer.train(train_data=train_subset, val_data=val_subset, val_every=args.val_every, n_epochs=args.epochs, generate_every=args.generate_every)

# Train and validate the translator summarizer
# summarizer = Summarizer(param, wordvecs=wordvecs, word_int_dict=word_int_dict)
# summarizer.train(train_data=train_subset, val_data=val_subset, val_every=args.val_every, n_epochs=args.epochs, generate_every=args.generate_every)


# Evaluate the baseline
rouge_scores = evaluate(
    data=val_subset, model=lead_3_baseline
)
print("Average F1 for LEAD-3:", np.mean(rouge_scores))
# rouge_hist(rouge_scores)

# Try out the inverse embedding
# print(emb_to_word(wordvecs["apple"])