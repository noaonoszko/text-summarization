import torch
from torch import nn
import torch.nn.functional as F
from rouge_score import rouge_scorer
from torch.nn import Parameter

import time
import numpy as np
from scipy.special import comb
import sys
from torch.autograd import Variable

import random
import subprocess

import matplotlib.pyplot as plt
from common_utils import *

from config import FlagsClass
FLAGS = FlagsClass()


class SummarizerParameters:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random_seed = 0
    
    eps_max = 1  # 1, 1 means no greedy epsilon
    eps_min = 1
    sent_emb_dim = 350
    n_sent = 40
    sentences_per_summary = 3
    use_combinations = False
    p = 10
    k = 5

    batch_size = 20
    learning_rate = 1e-3

    
    # Sentence extractor
    sent_ext_hidden_size = 600
    sent_ext_num_layers = 1

    # Document encoder
    doc_enc_hidden_size = 600
    doc_enc_num_layers = 1

    ## 
    out_channels = 50
    # sent_emb_size = 350
    word_emb_size = 200    # should be <=200
    doc_emb_size = 600
    max_sent_length = 100
    max_doc_length = 40
    kernel_widths = [1, 2, 3, 4, 5, 6, 7]
    word_emb_file_name = "1-billion-word-language-modeling-benchmark-r13output.word2vec.vec"
    ##


#  WordEmbedding class taken from https://github.com/spookyQubit/refreshExtractiveNeuralSummarizer
class WordEmbedding(nn.Module):
    def __init__(self, word_embedding_array):
        super(WordEmbedding, self).__init__()

        word_embed_size = word_embedding_array.shape[1]
        weight_pad = Parameter(torch.from_numpy(np.zeros((1, word_embed_size))).float(), requires_grad=False)
        weight_unk = Parameter(torch.from_numpy(np.zeros((1, word_embed_size))).float(), requires_grad=True)
        weight_vocab = Parameter(torch.from_numpy(word_embedding_array).float(), requires_grad=False)

        weight_all = torch.cat([weight_pad, weight_unk, weight_vocab], 0)

        """
        With the current implementation, vocab as well as unk both have requires_grad=True.
        This is not the behavior in the paper where only unk has requires_grad=True.
        In pytorch, cannot find a way to have some index to have
        requires_grad=True and others requires_grad=False (except for padding_index).
        """
        self.all_embeddings = nn.Embedding(weight_all.shape[0], word_embed_size, padding_idx=0)
        self.all_embeddings.weight = Parameter(weight_all)  # Overrides the requires_grad set in Parameters above

    def forward(self, word_input):
        """
        :param word_input: [batch_size, seq_len] tensor of Long type
        :return: input embedding with shape of [batch_size, seq_len, word_embed_size]
        """
        return self.all_embeddings(word_input)


class SentenceEncoder(nn.Module):
    def __init__(self, param):
        super(SentenceEncoder, self).__init__()
        self.out_channels = param.out_channels
        self.in_channels = param.word_emb_size
        self.sentembedding_size = param.sent_emb_dim
        self.kernel_widths = param.kernel_widths

        if self.sentembedding_size != (self.out_channels * len(self.kernel_widths)):
            raise ValueError("sent embed != out_chan * kW")

        self.kernels = [Parameter(torch.Tensor(self.out_channels,
                                           self.in_channels,
                                           kW).normal_(0, 0.05)) for kW in self.kernel_widths]
        self._add_to_parameters(self.kernels, 'SentEncoderKernel')
        self.bias = Parameter(torch.Tensor(self.out_channels).normal_(0, 0.05))

        self.lrn = nn.LocalResponseNorm(self.out_channels)

    def forward(self, sentences):
        """
        :param sentences: batch_size (each being a different sentence), each_sent_length, wordembed_size
        :return: batch_size, sentembedding_size
        """
        sentences = sentences.transpose(1, 2).contiguous()  # in_channel (i.e. word embed) has to be at 1
        xs = [F.relu(F.conv1d(sentences, kernel, bias=self.bias)) for kernel in self.kernels]  # [(batch_size, out_channels, k-h+1)] * len(self.kernels)
        xs = [F.max_pool1d(x, x.shape[2]) for x in xs]  # [(batch_size, out_channels, 1)] * len(self.kernels)
        xs = [self.lrn(x) for x in xs]  # [(batch_size, out_channels, 1)] * len(self.kernels)
        xs = [x.squeeze(2) for x in xs]  # [(batch_size, out_channels)] * len(self.kernels)
        xs = torch.cat(xs, 1)  # batch_size, out_channels * len(self.kernels) == batch_size, sentembedding_size
        return xs

    def _add_to_parameters(self, parameters, name):
            for i, parameter in enumerate(parameters):
                self.register_parameter(name='{}-{}'.format(name, i), param=parameter)


class DocumentEncoder(nn.Module):
    def __init__(self, param):
            super().__init__()
            self.param = param
            self.lstm = nn.LSTM(input_size=param.sent_emb_dim, hidden_size=param.doc_enc_hidden_size, num_layers=param.doc_enc_num_layers, batch_first=True)

    def forward(self, sent_embs):
        _, (h_n, _) = self.lstm(sent_embs)
        return h_n
        
# Sentence Extractor taken from https://github.com/spookyQubit/refreshExtractiveNeuralSummarizer
class SentenceExtractor(nn.Module):
    def __init__(self, param, word_embedding, sent_encoder, doc_encoder, vocab_dict):
        super().__init__()
        self.param = param
        self.vocab_dict = vocab_dict
        self.word_embedding = word_embedding
        self.sent_encoder = sent_encoder
        self.doc_encoder = doc_encoder
        self.lstm = nn.LSTM(input_size=param.sent_emb_dim, hidden_size=param.sent_ext_hidden_size, num_layers=param.sent_ext_num_layers, batch_first=True)
        self.ll_size = param.n_sent * param.sent_ext_hidden_size 
        self.ll = nn.Linear(self.ll_size, param.n_sent)

        # PAD_ID = 0     # padding done by default
        self.UNK_ID = 1  # used for encoding unknown words

    def forward(self, sentences):

        batch_size, n_sentences = sentences.shape
        # print('batch_size, n_sentences sizes: ', batch_size, n_sentences)
        # print("GPU MEMORY 1 : %s" % get_gpu_memory_map())

        # word tokenize sentences and pad so that every sentence has the same number of words AND create word embeddings
        sentence_embeddings = torch.zeros((batch_size, n_sentences, self.param.max_sent_length), dtype=torch.long, device=self.param.device)
        sentence_embeddings = Variable(sentence_embeddings).cuda()


        for b_i in range(batch_size):

            temp = torch.zeros((n_sentences, self.param.max_sent_length), dtype=torch.long, device=self.param.device)
            temp = Variable(temp).cuda()

            tokenized_sents = [nltk.tokenize.word_tokenize(sentence) for sentence in sentences[b_i]]

            for s_i in range(len(tokenized_sents)):

                if len(tokenized_sents[s_i]) <= self.param.max_sent_length:
                    for t_i in range(len(tokenized_sents[s_i])):
                        token = tokenized_sents[s_i][t_i]
                        if token in self.vocab_dict:
                            temp[s_i][t_i] = self.vocab_dict[token]
                        
                        else:
                            temp[s_i][t_i] = self.UNK_ID
                else:
                    # for s_i in range(self.param.max_sent_length):
                    for t_i in range(self.param.max_sent_length):
                        token = tokenized_sents[s_i][t_i]
                        if token in self.vocab_dict:
                            temp[s_i][t_i] = self.vocab_dict[token]
                        
                        else:
                            temp[s_i][t_i] = self.UNK_ID

            
            sentence_embeddings[b_i] = temp
            
        # print('sentence_embeddings after processing shape: ', sentence_embeddings.shape)
        sentence_embeddings = sentence_embeddings.view(-1, self.param.max_sent_length)
        # print('sentence_embeddings before word embeddings shape: ', sentence_embeddings.shape)
        sentence_embeddings = self.word_embedding(sentence_embeddings)              
        # print('sentence_embeddings after word embeddings shape: ', sentence_embeddings.shape)

        sentence_embeddings = self.sent_encoder(sentence_embeddings)
        # print('sentence_embeddings after encoder shape: ', sent_embs.shape)

        # sent_embs = sent_embs.flip(1)
        sentence_embeddings = sentence_embeddings.view(batch_size, n_sentences, self.param.sent_emb_dim)
        # Initial hidden state and cell state
        
        # print('sentence_embeddings before doc encoder shape: ', sent_embs.shape)

        doc_encoding = self.doc_encoder(sentence_embeddings)
        c0 = torch.zeros(doc_encoding.shape, device=self.param.device)

        h_t, _ = self.lstm(sentence_embeddings, (doc_encoding, c0))
        h_t = h_t.contiguous().view(batch_size, self.ll_size)
        
        out = self.ll(h_t)
        out = F.log_softmax(out, dim=1)
        # print(torch.cuda.memory_summary(device=None, abbreviated=True))
        sentence_embeddings = 0 
        doc_encoding = 0 
        # print("GPU MEMORY 2 : %s" % get_gpu_memory_map())
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

        vocab_dict, word_embedding_array = create_vocab_emb_dict(self.param.word_emb_file_name, self.param.word_emb_size)
        word_embedding = WordEmbedding(word_embedding_array)

        sent_encoder = SentenceEncoder(self.param)
        doc_encoder = DocumentEncoder(self.param)
        self.model = SentenceExtractor(self.param, word_embedding, sent_encoder, doc_encoder, vocab_dict).to(self.param.device)
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
            best_p_sentences_idx = torch.topk(torch.exp(outputs), k=self.param.p).indices
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
                chosen_sents_idx[dp] = torch.topk(torch.exp(outputs)[dp], k=self.param.sentences_per_summary).indices
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