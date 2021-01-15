# 1 
import json
import pandas as pd
import pickle

input_file = '/Users/gustavmolander/local/newsroom_data/train.jsonl'
output_file = '/Users/gustavmolander/local/newsroom_data/train_df_no_spacy.pickle'

#read jsonl file into list of sample rows
counter=0
data=[]
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.rstrip('\n|\r')))
        counter +=1
        
#wrap in dataframe        
df = pd.DataFrame(data)

#save to pickle
with open(output_file, 'wb') as handle:                                     
    pickle.dump(df, handle)