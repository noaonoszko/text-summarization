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
from sklearn.metrics.pairwise import cosine_similarity


def first_sent_baseline(text):
    """
    First sentence baseline. Returns the first sentence of the article in the datapoint text.
    """
    return nltk.tokenize.sent_tokenize(text["article"])


def evaluate(data, model):
    """
    Evaluates a text summarizer model by returning the rouge-1, rouge-2 and rouge-L scores.
    """
    rouge_scores = np.zeros((len(data), 3))
    for t, text in enumerate(tqdm(data)):
        tokens = model(text)

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        scores = scorer.score(
            text["highlights"],
            tokens[0],
        )
        rouge_scores[t] = np.array(
            [scores["rouge1"][2], scores["rouge2"][2], scores["rougeL"][2]]
        )
    return rouge_scores


def rouge_hist(rouge_scores):
    """
    Draws histograms for the three rouge scores and adds a vertical line for the average.
    """
    fig, axs = plt.subplots(3)
    titles = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    for i in range(3):
        axs[i].hist(rouge_scores[:, i])
        axs[i].axvline(
            rouge_scores[:, i].mean(), color="k", linestyle="dashed", linewidth=1
        )
        axs[i].set_title(titles[i])
    plt.show()


def cos_sim(x, y):
    """
    Returns the cosine similarity of x and y.
    """
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def emb_to_word(emb):
    """
    Returns the word with the highest cosine similarity with the input embedding emb.
    """
    max_cosine_similarity = 0
    for w, wordvec in enumerate(tqdm(wordvecs.items())):
        wordemb_with_noise = emb
        cosine_sim = cos_sim(wordemb_with_noise, wordvec[1])
        if cosine_sim > max_cosine_similarity:
            most_similar_word = wordvec[0]
            max_cosine_similarity = cosine_sim
    return most_similar_word, max_cosine_similarity


# Required download for tokenization
nltk.download("punkt")

# Load cnn data
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_set = dataset["train"]

# Load glove
glove = glove.Glove(
    glove_dir=str(Path(__file__).resolve().parents[3]) + "/Assignment 3/data/glove/"
)
wordvecs = glove.load_glove(50)
print("Done loading glove")

# Evaluate the baseline
# rouge_scores = evaluate(data=train_set.select(range(1000)), model=first_sent_baseline)
# print("Averages:", np.mean(rouge_scores, axis=0))
# rouge_hist(rouge_scores)

# Try out the inverse embedding
# print(emb_to_word(wordvecs["apple"])
