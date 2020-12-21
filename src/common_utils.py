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


def lead_3_baseline(text):
    """
    Lead-3 baseline. Returns the first three sentences of the article in the datapoint text.
    """
    return nltk.tokenize.sent_tokenize(text["article"])[:3]


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


def words_to_embs(wordvecs, words):
    """
    Returns the embedding of
    """
    return torch.tensor(
        [
            wordvecs[word] if word in wordvecs else np.zeros(wordvecs["the"].shape[0])
            for word in words
        ],
        dtype=torch.float32,
    )


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


def train_loader(wordvecs, train_set):
    article_emb = words_to_embs(
        wordvecs, nltk.tokenize.word_tokenize(train_set[0]["article"])
    )
    highlights_emb = words_to_embs(
        wordvecs, nltk.tokenize.word_tokenize(train_set[0]["highlights"])
    )
    return article_emb, highlights_emb
