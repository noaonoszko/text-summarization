# 3
import pickle
import pandas as pd
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from datetime import datetime as dt


### Helper function

def text_to_sent_list(text, 
                      nlp = spacy.load("en_core_web_lg"), 
                      embedder = SentenceTransformer('distilbert-base-nli-mean-tokens'),
                      min_len=2):
    
    ''' Returns cleaned article sentences and BERT sentence embeddings'''
    
    #convert to list of sentences
    text = nlp(text)
    sents = list(text.sents)
    #remove short sentences by threshhold                                                                                                
    sents_clean = [sentence.text for sentence in sents if len(sentence)> min_len]
    #remove entries with empty list
    sents_clean = [sentence for sentence in sents_clean if len(sentence)!=0]
    #embed sentences (deafult uses BERT SentenceTransformer)
    sents_embedding= np.array(embedder.encode(sents_clean, convert_to_tensor=True))
    
    return sents_clean, sents_embedding



### Script

output_file = '/Users/gustavmolander/local/newsroom_data/train_stats_df_processed_extr_5000.pickle'  
#load full extractive df
df = pd.read_pickle('/Users/gustavmolander/local/newsroom_data/train_stats_df_extractive_no_spacy.pickle')

#truncate for local computation
df= df.head(5000).reset_index(drop=True)

#load nlp and embedder
# en_core_web_lg pretrained statistical models for English
nlp = spacy.load("en_core_web_lg")
embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')

t1 = dt.now()
print(t1)

#extract clean sentence list and sentence embedding for each article TEXT
f = lambda text: text_to_sent_list(text, nlp=nlp, embedder=embedder, min_len=2)
s_interim_tuple = df['text'].apply(f)

df['text_clean'] = s_interim_tuple.apply(lambda x: x[0])
df['text_embedding'] = s_interim_tuple.apply(lambda x: x[1])

#extract clean sentence list and sentence embedding for each article SUMMARY
f = lambda summ: text_to_sent_list(summ, nlp=nlp, embedder=embedder, min_len=0)
s_interim_tuple = df['summary'].apply(f)

df['summary_clean'] = s_interim_tuple.apply(lambda x: x[0])
df['summary_embedding'] = s_interim_tuple.apply(lambda x: x[1])

with open(output_file, 'wb') as handle:                                     
    pickle.dump(df, handle)

t2=dt.now()
print(t2)
print(t2-t1)

#500 articles took 14mins on macbookpro i7