#2
import pickle
import pandas as pd
 
pickle_file = '/Users/gustavmolander/local/newsroom_data/train_df_no_spacy.pickle'
output_file = '/Users/gustavmolander/local/newsroom_data/train_stats_df_extractive_no_spacy.pickle'

#load all data
df = pd.read_pickle(pickle_file)

#filter for extractive summaries only
df = df[df.density_bin == 'extractive']
print(df.shape)

#save to pickle file
with open(output_file, 'wb') as handle:                                     
    pickle.dump(df, handle)