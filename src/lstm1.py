'''lstm1.py'''
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.callbacks import LambdaCallback
#from sklearn.metrics import classification_report, confusion_matrix
from functions import rouge_score
from functions import calc_rouge_scores

class PredictionCallback(tf.keras.callbacks.Callback):    
  def on_epoch_end(self, epoch, logs={}):
    y_pred_epoch[epoch] = self.model.predict(X_val)
    print('prediction: {} at epoch: {}'.format(y_pred_epoch, epoch))

n_train = 4800
n_val = 200
n_sum = 3
n_sent = 60
epochs = 20
y_pred_epoch = np.zeros([epochs, n_val, n_sent, 1])

input_filename = '/Users/gustavmolander/local/newsroom_data/train_stats_dict_processed_extr_final_5000_.pickle'
output_file = '/Users/gustavmolander/local/newsroom_data/lstm1.pickle'

data_dict = pd.read_pickle(input_filename)
df = data_dict['df_original']

#convert to numpy array
to_array = lambda x: np.array(x)
df.text_embedding = df.text_embedding.apply(to_array)
df.labels= df.labels.apply(to_array)
df.text_embedding = df.text_embedding.apply(lambda x: x.reshape(1, x.shape[0],x.shape[1]))
df.labels = df.labels.apply(lambda x: x.reshape(1, len(x),1))



X_train = np.zeros([n_train, n_sent, 768])
y_train = np.zeros([n_train, n_sent, 1])
X_val = np.zeros([n_val, n_sent, 768])
y_val = np.zeros([n_val, n_sent, 1])

        
for i in range(n_train):
    n_copy = df.text_embedding[i].shape[1]
    if n_copy > n_sent:
        n_copy = n_sent
    X_train[i, :n_copy] = df.text_embedding[i][0][:n_copy]
    y_train[i, :n_copy] = df.labels[i][0][:n_copy]
for i in range(n_val):
    n_copy = df.text_embedding[n_train+i].shape[1]
    if n_copy > n_sent:
        n_copy = n_sent
    X_val[i, :n_copy] = df.text_embedding[n_train+i][0][:n_copy]
    y_val[i, :n_copy] = df.labels[n_train+i][0][:n_copy]



# define LSTM
model = Sequential()
model.add(LSTM(25, input_shape=(None, 768), return_sequences=True, dropout=0))
#model.add(LSTM(25, input_shape=(None, 768), return_sequences=True, dropout=0))
#model.add(Bidirectional(LSTM(50, return_sequences=True, dropout=0), input_shape=(None, 768)))
#model.add(Bidirectional(LSTM(50, return_sequences=True, dropout=0), input_shape=(None, 768)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))

opt = keras.optimizers.Adam(learning_rate=0.00005)
model.compile(loss='binary_crossentropy', optimizer=opt, 
              metrics=[tf.keras.metrics.SensitivityAtSpecificity(0.5, num_thresholds=1)])
#weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))

avg_f_epoch = np.zeros(epochs)
for j in range(epochs):
    # train LSTM
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=8 )

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.show()
        
    # evaluate LSTM
    #print(y_pred[0].flatten()[0:10])
    #print(y_pred.shape)



    y_pred = model.predict(X_val, verbose=0)    
    avg_f = 0
    counts = 0
    for i in range(n_val):
        gold_sum_tot = df.labels_idx_list[n_train+i]
        gold_sum = gold_sum_tot[gold_sum_tot < n_sent] 
        gold_sum_len = len(gold_sum)
        LEADN = np.array([i for i in range(gold_sum_len)])
        pred_sum = np.sort(np.argpartition(y_pred[i].flatten(), -gold_sum_len, axis=0)[-gold_sum_len:])
        #print(pred_sum)
        #print(gold_sum)
        if (pred_sum == LEADN).all():
            #print("LEADN - Discard!")
            continue 
        pred_summaries = [df.text_clean[n_train+i][j] for j in pred_sum if j < len(df.text_clean[n_train+i])]
        gold_summaries = [df.text_clean[n_train+i][j] for j in gold_sum]
        #print(rouge_score(' '.join(pred_summaries), ' '.join(gold_summaries)))
        avg_f += rouge_score(' '.join(pred_summaries), ' '.join(gold_summaries))
        counts += 1

    avg_f_epoch[j] = avg_f/counts
    print(avg_f_epoch[j])

plt.plot(avg_f_epoch)
plt.show()


    
# #retrieve summary pairs
# doc_index = df.index[~mask_train]
# pred_summaries = [' '.join(np.array(df.text_clean[doc_index].iloc[j])[np.array(idx_list[j])].tolist()) 
#                   for j in range(len(idx_list))]
# df_gold = df.summary_clean[doc_index]
# gold_summaries = [' '.join(df_gold .iloc[j]) for j in range(len(pred_summaries))]
# summaries_comp = tuple(zip(pred_summaries, gold_summaries))


# print(pred_summaries[0])
# print(summaries_comp)
#calculate rouge score
# scores = calc_rouge_scores(pred_summaries, gold_summaries, 
#                                   keys=['rouge1', 'rougeL'], use_stemmer=True)

# results_dict = {'summaries_comp': summaries_comp,
#                'sent_index_number': idx, 'Rouge': scores, 'mod_summary': model.summary()}

# with open(output_file, 'wb') as handle:                                     
#     pickle.dump(results_dict, handle)