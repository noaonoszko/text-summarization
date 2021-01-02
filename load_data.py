import sys
import torch
from torch import nn
import time
import torchtext
import numpy as np
import matplotlib.pyplot as plt

import gensim.downloader
from gensim.models import KeyedVectors
from datasets import load_dataset

def load_gensim_vectors(model_file, builtin=False, limit=None):
    print(f"Loading model '{model_file}' via gensim...", end='')
    sys.stdout.flush()
    if builtin:
        gensim_model = gensim.downloader.load(model_file)
    else:
        gensim_model = KeyedVectors.load_word2vec_format(model_file, binary=True, limit=limit)
    if not limit:
        limit = len(gensim_model.index2word)
    vectors = torch.FloatTensor(gensim_model.vectors[:limit])
    voc = gensim_model.index2word[:limit]

    is_cased = False
    for w in voc:
        w0 = w[0]
        if w0.isupper():
            is_cased = True
            break
    
    print(' done!')
    return vectors, voc, is_cased


dataset = load_dataset("cnn_dailymail", "3.0.0")
# dataset is a dict with 3 entries:     {'train': (287113, 3), 'validation': (13368, 3), 'test': (11490, 3)}
# each entry is a dict with 3 entries:  'article', 'highlights', 'id'

#print(dataset["train"][1]["article"])
#print("")
#print(dataset["train"][1]["highlights"])
#print("")
#print(dataset["train"][1]["id"])

gensim_vectors = load_gensim_vectors(model_file='glove-wiki-gigaword-100', builtin=True, limit=10)

print(gensim_vectors[0].shape)
print(len(gensim_vectors))
print(gensim_vectors)
print(type(gensim_vectors))