import torch
from torch import nn
import torch.nn.functional as F
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
    
    eps_max = 1 # 1, 1 means no greedy epsilon
    eps_min = 1
    sent_emb_dim = 768
    n_sent = 40
    sentences_per_summary = 3
    use_combinations = False
    p = 10
    k = 5

    batch_size = 2
    learning_rate = 1e-3

    
    # Sentence extractor
    sent_ext_hidden_size = 600
    sent_ext_num_layers = 1

    # Document encoder
    doc_enc_hidden_size = 600
    doc_enc_num_layers = 1


class SentenceEncoder(nn.Module):
    def __init__(self, param, sent_emb_dict):
        super().__init__()
        self.param = param
        self.sent_emb_dict = sent_emb_dict

    def forward(self, sentences):
        n_batches, n_sentences = sentences.shape
        sentence_embeddings = torch.zeros((n_batches, n_sentences, self.param.sent_emb_dim), device=self.param.device)
        for dp, datapoint in enumerate(sentences):
            for s, sentence in enumerate(datapoint):
                sentence_embeddings[dp, s] = self.sent_emb_dict[sentence]
        return sentence_embeddings

class DocumentEncoder(nn.Module):
    def __init__(self, param):
            super().__init__()
            self.param = param
            self.lstm = nn.LSTM(input_size=param.sent_emb_dim, hidden_size=param.doc_enc_hidden_size, num_layers=param.doc_enc_num_layers, batch_first=True)

    def forward(self, sent_embs):
        _, (h_n, _) = self.lstm(sent_embs)
        return h_n
        

class SentenceExtractor(nn.Module):
    def __init__(self, param, sent_encoder, doc_encoder):
        super().__init__()
        self.param = param
        self.sent_encoder = sent_encoder
        self.doc_encoder = doc_encoder
        self.lstm = nn.LSTM(input_size=param.sent_emb_dim, hidden_size=param.sent_ext_hidden_size, num_layers=param.sent_ext_num_layers, batch_first=True)
        self.ll_size = param.n_sent * param.sent_ext_hidden_size 
        self.ll = nn.Linear(self.ll_size, param.n_sent)

    def forward(self, sentences):
        # TODO: Feed sentences in reverse order
        batch_size, _ = sentences.shape
        sent_embs = self.sent_encoder(sentences)
        doc_encoding = self.doc_encoder(sent_embs)
        c0 = torch.zeros(doc_encoding.shape, device=self.param.device)

        h_t, _ = self.lstm(sent_embs, (doc_encoding, c0))
        h_t = h_t.contiguous().view(batch_size, self.ll_size)
        
        out = self.ll(h_t)
        out = F.log_softmax(out, dim=1)
        return out

class Summarizer:
    def __init__(self, param):
        self.param = param

    def rouge_score(self, highlights, text):
        """
        Return: the mean of R1, R2 and RL scores.
        text and highlights are strings
        """
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        scores = scorer.score(
            highlights,
            text
        )

        rouge_mean = np.mean(
            [scores["rouge1"][2], scores["rouge2"][2], scores["rougeL"][2]]
        )
        return rouge_mean
        
    def validate(self, data, generate=False): # should sample from network output instead
        n_datapoints = len(data)
        random_indices = np.random.randint(0, len(data), size=n_datapoints)
        highlights = data.select(random_indices)["highlights"]
        sentences = np.empty((n_datapoints, self.param.n_sent), dtype=object)
        for i, dp in enumerate(random_indices):
            sents = nltk.tokenize.sent_tokenize(data[dp.item()]["article"])
            sentences[i, :len(sents[:self.param.n_sent])] = sents[:self.param.n_sent]
        loss, outputs, chosen_summaries = self.forward_prop(sentences, highlights, eps=1, use_combinations=False)
        
        # Calculate rouge score
        n_datapoints = sentences.shape[0]
        rouge_score = 0
        for dp, datapoint in enumerate(sentences):
            if chosen_summaries[dp] is not None:
                rouge_score += self.rouge_score(highlights[dp], chosen_summaries[dp]) / n_datapoints

        if generate:
            for dp in range(len(data)):
                print("----------------------------output----------------------------")
                print(chosen_summaries[dp])
                print("----------------------------target----------------------------\n")
                print(highlights[dp])
                if dp == 5:
                    break
        return loss, rouge_score

    def train(self, sent_emb_dict, train_data, val_data=False, n_epochs=200, val_every=50, generate_every=50, batch_size=5, baseline_rouge_score=False):
        # Setting a fixed seed for reproducibility.
        torch.manual_seed(self.param.random_seed)
        random.seed(self.param.random_seed)

        sent_encoder = SentenceEncoder(self.param, sent_emb_dict)
        doc_encoder = DocumentEncoder(self.param)
        self.model = SentenceExtractor(self.param, sent_encoder, doc_encoder).to(self.param.device)
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.param.learning_rate
        )
        
        n_batches = int(len(train_data)/self.param.batch_size) + 1
        # n_plot_points = int(n_epochs * n_batches / val_every) # plot one point per val_every examples
        n_plot_points = n_epochs
        train_losses = torch.zeros(n_plot_points, requires_grad=False)
        val_losses = torch.zeros(n_plot_points, requires_grad=False)
        val_rouge_scores = torch.zeros(n_plot_points, requires_grad=False)
        t = trange(1, n_epochs + 1)
        for epoch in t:
            if n_epochs == 1:
                eps = self.param.eps_min
            else:
                eps = self.param.eps_max-(self.param.eps_max-self.param.eps_min)*(epoch-1)/(n_epochs-1)
            mean_loss = 0
            for b, (sentences, highlights) in enumerate(train_loader_rl(self.param, train_data)):
                loss, _, _ = self.forward_prop(sentences, highlights, eps=eps, use_combinations=self.param.use_combinations)
                mean_loss += loss/sentences.shape[0]

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # train_losses[n_batches*(epoch-1)+b] = loss
                
                # Add mean loss over batches to the list if we're at the last batch
                if b == n_batches-1:
                    train_losses[epoch-1] = mean_loss

                # Validate
                if sentences.shape[0] == self.param.batch_size:
                    examples_seen = (epoch-1) * len(train_data) + (b+1)*self.param.batch_size
                else:
                    examples_seen = epoch * len(train_data)
                # if examples_seen % val_every == 0:
                if epoch % val_every == 0 and b == n_batches-1:
                    generate = True if examples_seen % generate_every == 0 or (epoch == n_epochs and b == n_batches - 1) else False
                    val_loss, val_rouge_score = self.validate(val_data, generate)
                    
                    # val_losses[n_batches*(epoch-1)+b] = val_loss
                    # val_rouge_scores[n_batches*(epoch-1)+b] = val_rouge_score
                    val_losses[epoch-1] = val_loss
                    val_rouge_scores[epoch-1] = val_rouge_score

                    # Save loss plot
                    examples = np.linspace(1, n_epochs*len(train_data), n_plot_points)
                    # plt.scatter(examples, train_losses.detach(), label="train_loss", color="blue", s=0.2)
                    # plt.scatter(examples, val_losses.detach(), label="val_loss", color="red", s=0.2)
                    fig, (ax1, ax2) = plt.subplots(2)
                    ax1.plot(range(n_epochs), train_losses.detach(), label="train_loss", color="blue", linewidth=0.5)
                    ax2.plot(range(n_epochs), val_losses.detach(), label="val_loss", color="red", linewidth=0.5)
                    ax1.set_title("train_loss")
                    ax2.set_title("val_loss")
                    plt.savefig("loss.png")
                    plt.close(fig)
                    # plt.plot(examples, val_rouge_scores)
                    plt.plot(range(n_epochs), val_rouge_scores, label="Average rouge F1 model")
                    if baseline_rouge_score:
                        plt.plot(range(n_epochs), baseline_rouge_score*np.ones(n_epochs), color="purple", label="Average rouge F1 LEAD-3")
                    plt.legend()
                    plt.savefig("rouge_scores.png")
                    plt.clf()
                    t.set_postfix(
                        loss=mean_loss.item(), val_loss=val_loss.item(), val_rouge_score=val_rouge_score
                    )

    def forward_prop(self, sentences, highlights, eps, use_combinations=True):
        n_datapoints = sentences.shape[0]
        
        # Feed forward
        outputs = self.model(sentences)

        # Calculate rouge score for every sentence
        rouge_scores = torch.zeros(sentences.shape)
        for dp, datapoint in enumerate(sentences):
            for s, sentence in enumerate(datapoint):
                if sentence is not None:
                    rouge_scores[dp, s] = self.rouge_score(highlights[dp], sentence)

        chosen_summaries = np.empty(n_datapoints, dtype=object)
        chosen_sents_idx = torch.zeros((n_datapoints, self.param.sentences_per_summary), dtype=torch.long, device=self.param.device)
        rewards = torch.zeros(n_datapoints, device=self.param.device)

        if use_combinations:
            # Calculate scores for combinations of p sentences, sample one and calculate reward
            best_p_sentences_idx = torch.topk(outputs, k=self.param.p).indices
            n_combinations = int(comb(self.param.p, self.param.sentences_per_summary))
            scores = torch.zeros((n_datapoints, n_combinations), device=self.param.device)
            for dp, datapoint in enumerate(sentences):
                combinations = torch.combinations(best_p_sentences_idx[dp], self.param.sentences_per_summary)
                for c, combination in enumerate(combinations):
                    try:
                        summary = ""
                        for sentence_idx in combination:
                            summary += " "+sentences[dp, sentence_idx]
                        scores[dp, c] = 1e-5+self.rouge_score(highlights[dp], summary)
                    except TypeError:
                        print("TypeError occured due to a sentence being None")
                        exit()
                scores[dp] /= torch.sum(scores[dp].type(torch.float64)) # normalize
                values, best_k_summaries_comb_idx = torch.topk(scores[dp], k=self.param.k)#.indices
                scores_best_k = scores[dp, best_k_summaries_comb_idx]
                scores_best_k = scores_best_k/torch.sum(scores_best_k)
                if torch.rand(1).item() < eps:
                    chosen_summary_idx = torch.multinomial(scores_best_k, 1)[0].item()
                else:
                    chosen_summary_idx = torch.argmax(scores_best_k).item()
                chosen_sents_idx[dp] = combinations[chosen_summary_idx]
                chosen_summaries[dp] = ""
                for c, sentence_idx in enumerate(chosen_sents_idx[dp]):
                    if c == 0:
                        chosen_summaries[dp] += sentences[dp, sentence_idx] 
                    else:
                        chosen_summaries[dp] += " "+sentences[dp, sentence_idx] 
                rewards[dp] = self.rouge_score(highlights[dp], chosen_summaries[dp])
                
        else:
            for dp, datapoint in enumerate(sentences):
                chosen_sents_idx[dp] = torch.topk(outputs[dp], k=self.param.sentences_per_summary).indices
                try:
                    chosen_summaries[dp] = ""
                    for sentence_idx in chosen_sents_idx[dp]:
                        chosen_summaries[dp] += " "+sentences[dp, sentence_idx]
                except TypeError:
                    print("TypeError occured due to a sentence being None")
                    exit()
                rewards[dp] = self.rouge_score(highlights[dp], chosen_summaries[dp])
        chosen_outputs = torch.zeros(n_datapoints, self.param.sentences_per_summary, device=self.param.device)
        for dp, datapoint in enumerate(sentences):
           chosen_outputs[dp] = outputs[dp, chosen_sents_idx[dp]]
        loss = -torch.dot(rewards, torch.sum(chosen_outputs, dim=1)) / n_datapoints # expected_gradient loss
        return loss, outputs, chosen_summaries