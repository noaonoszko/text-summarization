# 6
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

output_file = '/Users/gustavmolander/local/newsroom_data/extractive_all_domain_labels.pickle' 

input_file = '/Users/gustavmolander/local/newsroom_data/train_stats_dict_processed_extr_final_5000_.pickle' 
data = pd.read_pickle(input_file )

embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')

#Make single df with only Embeddings and doc label
df_embed = data['df_X'].loc[:,'Doc_BERT_D_0': 'Doc_BERT_D_767']
df_doc_label = pd.DataFrame(data['Xy_doc_label_array'],columns=['doc_label'])
df = pd.concat([df_doc_label, df_embed], axis=1)
df = df.drop_duplicates().set_index('doc_label', drop=True)

#embed lambda function
embed = lambda x: embedder.encode(x, convert_to_tensor=False)

#define subject domains
domains = ['entertainment','politics', 'business', 'crime']
#find domain word embeddings using BERT
domain_embed = [embed(dom) for dom in domains]
#wrap in dataframe
df_dom_embed = pd.DataFrame(domain_embed, index = domains,
                            columns = df.columns)
#calculate cosine similarity between article and each subject
cos_matrix = cosine_similarity(df, df_dom_embed)

#return subject word from index number function
f = np.vectorize(lambda x: domains[x])
#find max cos sim and return matching subject
doc_domain = f(np.argmax(cos_matrix, axis=1))
#Add to primary dataframe
df['domain'] = doc_domain

#Add to primary dictionary for storage
data.update({'domain_labels_arr': df['domain'].values})

#save to pickle file
with open(output_file, 'wb') as handle:                                     
    pickle.dump(data, handle)