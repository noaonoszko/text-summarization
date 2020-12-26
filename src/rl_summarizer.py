import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer

import time
import numpy as np
from scipy.special import comb
import sys

import random

import matplotlib.pyplot as plt
from common_utils import *


class SummarizerParameters:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random_seed = 0
    vocab_size = 400000
    n_sent = 120
    p = 5
    sentences_per_summary = 3

    batch_size = 64

    learning_rate = 1e-2
    weight_decay = 0

    word_emb_dim = 50
    sent_emb_dim = 768

class SentenceExtractor(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.output_layer = nn.Linear(param.n_sent * param.sent_emb_dim, param.n_sent)
        self.output_activation = nn.Softmax(1)

    def reward(self, sentence):
        pass

    def forward(self, sentences):
        n_batches, n_sentences = sentences.shape
        sentence_embeddings = torch.zeros((n_batches, n_sentences, self.param.sent_emb_dim), device=self.param.device)
        for dp, datapoint in enumerate(sentences):
            for s, sentence in enumerate(datapoint):
                if sentence is not None:
                    sentence_embeddings[dp, s] = torch.tensor(self.sbert_model.encode(sentence), device=self.param.device)
                else:
                    sentence_embeddings[dp, s] = torch.zeros(self.param.sent_emb_dim, device=self.param.device)
        flat_embs = torch.flatten(sentence_embeddings, start_dim=1)
        output = self.output_layer(torch.flatten(sentence_embeddings, start_dim=1))
        output = self.output_activation(output)
        return output

    def rouge_score(self, text, highlights):
        """
        Return: the mean of R1, R2 and RL scores.
        text and highlights are strings
        """
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        scores = scorer.score(
            text,
            highlights
        )

        rouge_mean = np.mean(
            [scores["rouge1"][2], scores["rouge2"][2], scores["rougeL"][2]]
        )
        return rouge_mean

class Summarizer:
    def __init__(self, param):
        self.param = param

    def validate(self, data, generate=False): # should sample from network output instead
        highlights = [data[dp]["highlights"] for dp in range(len(data))]
        sentences = np.empty((len(data), self.param.n_sent), dtype=object)
        for dp in range(len(data)):
            sents = nltk.tokenize.sent_tokenize(data[dp]["article"])
            sentences[dp, :len(sents)] = sents
        loss, outputs = self.forward_prop(sentences, highlights)
        
        
        
        # Check combinations
        n_datapoints = sentences.shape[0]
        best_p_sentences_idx = torch.topk(outputs, k=self.param.p, dim=1).indices
        n_combinations = int(comb(self.param.p, self.param.sentences_per_summary))
        scores = torch.zeros((n_datapoints, n_combinations), device=self.param.device)
        chosen_summaries = np.empty(n_datapoints, dtype=object)
        rewards = torch.zeros(n_datapoints, device=self.param.device)
        rouge_score = 0
        for dp, datapoint in enumerate(sentences):
            combinations = torch.combinations(best_p_sentences_idx[dp], self.param.sentences_per_summary)
            for c, combination in enumerate(combinations):
                try:
                    summary = ""
                    for sentence_idx in combination:
                        summary += " "+sentences[dp, sentence_idx]
                    scores[dp, c] = self.model.rouge_score(summary, highlights[dp])
                except TypeError:
                    continue
                scores[dp] /= torch.sum(scores[dp].type(torch.float64)) # normalize
                chosen_summary_idx = torch.multinomial(scores[dp], 1)[0].item()
                chosen_summaries[dp] = ""
                for c, sentence_idx in enumerate(combinations[chosen_summary_idx]):
                    if c == 0:
                        chosen_summaries[dp] += sentences[dp, sentence_idx] 
                    else:
                        chosen_summaries[dp] += " "+sentences[dp, sentence_idx] 
            if chosen_summaries[dp] is not None:
                rouge_score += self.model.rouge_score(chosen_summaries[dp], highlights[dp]) / n_datapoints

        if generate:
            for dp in range(len(data)):
                print("----------------------------output----------------------------")
                print(chosen_summaries[dp])
                print("\n----------------------------target----------------------------")
                print(highlights[dp])
        return loss, rouge_score

    def train(self, train_data, val_data=False, n_epochs=200, val_every=50, generate_every=50, batch_size=5):
        # Setting a fixed seed for reproducibility.
        torch.manual_seed(self.param.random_seed)
        random.seed(self.param.random_seed)

        self.model = SentenceExtractor(self.param).to(self.param.device)
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.param.learning_rate, weight_decay=self.param.weight_decay
        )

        train_losses = torch.zeros(n_epochs, requires_grad=False)
        val_losses = torch.zeros(n_epochs, requires_grad=False)
        val_rouge_scores = torch.zeros(n_epochs, requires_grad=False)
        t = trange(1, n_epochs + 1)
        for epoch in t:
            for b, (sentences, highlights) in enumerate(train_loader_rl(self.param, train_data)):
                loss, _ = self.forward_prop(sentences, highlights)
                
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses[epoch-1] = loss

                # Validate
                if epoch % val_every == 0 and b == 0:
                    generate = True if epoch % generate_every == 0 or epoch == n_epochs else False
                    val_loss, val_rouge_score = self.validate(val_data, generate)
                    val_losses[epoch-1] = val_loss
                    val_rouge_scores[epoch-1] = val_rouge_score
                    
                    # Save loss plot
                    if generate:
                        plt.plot(train_losses.detach())
                        plt.plot(val_losses.detach())
                        plt.legend(["train_loss", "val_loss"])
                        plt.savefig("loss.png")
                        plt.clf()
                        plt.plot(val_rouge_scores)
                        plt.savefig("rouge_scores.png")
                    t.set_postfix(
                        loss=loss.item(), val_loss=val_loss.item(), val_rouge_score=val_rouge_score
                    )

    def forward_prop(self, sentences, highlights):
        n_datapoints = sentences.shape[0]
        
        # Feed forward
        outputs = self.model(sentences)

        # Calculate rouge score for every sentence
        rouge_scores = torch.zeros(sentences.shape)
        for dp, datapoint in enumerate(sentences):
            for s, sentence in enumerate(datapoint):
                if sentence is not None:
                    rouge_scores[dp, s] = self.model.rouge_score(sentence, highlights[dp])

        # Calculate scores for combinations of p sentences, sample one and calculate reward
        best_p_sentences_idx = torch.topk(rouge_scores, k=self.param.p).indices
        n_combinations = int(comb(self.param.p, self.param.sentences_per_summary))
        scores = torch.zeros((n_datapoints, n_combinations), device=self.param.device)
        chosen_summaries = np.empty(n_datapoints, dtype=object)
        rewards = torch.zeros(n_datapoints, device=self.param.device)
        for dp, datapoint in enumerate(sentences):
            combinations = torch.combinations(best_p_sentences_idx[dp], self.param.sentences_per_summary)
            for c, combination in enumerate(combinations):
                try:
                    summary = ""
                    for sentence_idx in combination:
                        summary += " "+sentences[dp, sentence_idx]
                    scores[dp, c] = self.model.rouge_score(summary, highlights[dp])
                except TypeError:
                    continue
                scores[dp] /= torch.sum(scores[dp].type(torch.float64)) # normalize
                chosen_summary_idx = torch.multinomial(scores[dp], 1)[0].item()
                chosen_summaries[dp] = ""
                for c, sentence_idx in enumerate(combinations[chosen_summary_idx]):
                    if c == 0:
                        chosen_summaries[dp] += sentences[dp, sentence_idx] 
                    else:
                        chosen_summaries[dp] += " "+sentences[dp, sentence_idx] 
                rewards[dp] = self.model.rouge_score(chosen_summaries[dp], highlights[dp])
        loss = -torch.dot(rewards, torch.sum(torch.log(outputs), dim=1)) # expected_gradient loss
        loss = torch.tensor(loss, requires_grad=True)
        return loss, outputs