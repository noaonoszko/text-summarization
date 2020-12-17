from tqdm import tqdm
from tqdm.auto import trange

from datasets import load_dataset

import numpy as np
import matplotlib.pyplot as plt

import nltk
import torch
from rouge_score import rouge_scorer

# Required download for tokenization
nltk.download("punkt")

# Load cnn data
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_set = dataset["train"]


def evaluate_baseline(n_iter=False):
    if not n_iter:
        n_iter = len(train_set)
    results = np.zeros((n_iter, 3))
    for t, text in enumerate(tqdm(train_set.select(range(n_iter)))):
        tokens = nltk.tokenize.sent_tokenize(text["article"])

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        scores = scorer.score(
            text["highlights"],
            tokens[0],
        )
        results[t] = np.array(
            [scores["rouge1"][2], scores["rouge2"][2], scores["rougeL"][2]]
        )

    fig, axs = plt.subplots(3)
    titles = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    for i in range(3):
        axs[i].hist(results[:, i])
        axs[i].axvline(results[:, i].mean(), color="k", linestyle="dashed", linewidth=1)
        axs[i].set_title(titles[i])
    plt.show()


evaluate_baseline(10000)