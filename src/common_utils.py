from tqdm import tqdm
from tqdm.auto import trange
from pathlib import Path

from datasets import load_dataset

import numpy as np
import matplotlib.pyplot as plt
from collections import *

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
    if type(words) == str:
        wordvecs[words] if words in wordvecs else np.zeros(wordvecs["the"].shape[0]) # perhaps random
    elif type(words) == list:
        return torch.tensor(
            [
                wordvecs[word] if word in wordvecs else np.zeros(wordvecs["the"].shape[0])
                for word in words
            ],
            dtype=torch.float32,
        )
def words_to_ints(word_int_dict, words):
    return torch.tensor([word_int_dict[word] if word in word_int_dict else 0 for word in words]) # else 0, needs to be addressed!

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


def train_loader(wordvecs, word_int_dict, train_set, batch_size=16, emb_size=50):
    n_batches = int(train_set.shape[0]/batch_size) + 1
    for b in range(n_batches):
        # Calculate max length of articles and highlights in batch
        max_article_len = 0
        max_highlights_len = 0
        actual_batch_size = batch_size if b < n_batches - 1 else train_set.shape[0]-batch_size*b
        for i in range(actual_batch_size):
            A_tokens = nltk.tokenize.word_tokenize(train_set[batch_size*b+i]["article"])
            max_article_len = (
                len(A_tokens) if len(A_tokens) > max_article_len else max_article_len
            )
            H_tokens = nltk.tokenize.word_tokenize(train_set[batch_size*b+i]["highlights"])
            max_highlights_len = (
                len(H_tokens)
                if len(H_tokens) > max_highlights_len
                else max_highlights_len
            )
        
        # Prepare batches
        actual_batch_size = batch_size if b < n_batches - 1 else train_set.shape[0]-b*batch_size # last batch is sometimes smaller
        article_emb_all = torch.zeros((actual_batch_size, max_article_len, emb_size))
        highlights_words_all = np.empty((actual_batch_size, max_highlights_len), dtype=object)
        for i in range(actual_batch_size):
            article_emb = words_to_embs(
                wordvecs, nltk.tokenize.word_tokenize(train_set[batch_size*b+i]["article"])
            )
            highlights_words = np.array(nltk.tokenize.word_tokenize(train_set[batch_size*b+i]["highlights"]))
            article_emb_all[i, : len(article_emb), :] = article_emb
            highlights_words_all[i, : len(highlights_words)] = highlights_words
        yield (article_emb_all, highlights_words_all)