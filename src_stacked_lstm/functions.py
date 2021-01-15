import pandas as pd
import numpy as np
import random
from rouge_score import rouge_scorer

###Sub-function used in return_pred_summarie
def rouge_score(text, highlights):
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

        #print(scores["rouge1"][2], scores["rouge2"][2], scores["rougeL"][2])
        rouge_mean = np.mean(
            [scores["rouge1"][2], scores["rouge2"][2], scores["rougeL"][2]]
        )
        return rouge_mean

def return_greater_than_min_num(arr, thresh=0.5, min_num=1, fix_num_flag=False, fix_num=3):
    
    '''returns top sentences by index numbers in ascending format and according to input
    specifications
    '''
    #want fixed number sentences?
    if fix_num_flag == True:
        idx = np.argsort(arr)[-fix_num:]
        
    #return above model threshold only    
    else:
        idx_prelim = np.where(arr>= thresh)
        
        #filter for minimum number required
        if idx_prelim[0].shape[0] <= min_num:
            idx = np.argsort(arr)[-min_num:]
        else:
            idx = idx_prelim
            
    #return in ascending order
    return sorted(idx)


###Main helper function    
def return_df_pred_summaries( Xy_doc_label, y_pred, df_text, thresh, min_num,
                             return_all=False, fix_num_flag=False, fix_num=3):
    
    '''return list of predicted summaries and additional information if required
    and according to inout specifications'''
    
    #Wrangle to doc label and flattened array of predictions for each article
    df_label_pred = pd.DataFrame({'doc_label': Xy_doc_label.flatten(),
                                                 'y_pred': y_pred.flatten()}) 
    df_label_pred = df_label_pred.groupby('doc_label').agg(list) 

    df_label_pred = df_label_pred.applymap(lambda x: np.array(x))

    #subfunction to lambda
    f = lambda arr: return_greater_than_min_num(arr, thresh=thresh, 
                                    min_num=min_num,fix_num_flag = fix_num_flag, 
                                                            fix_num=fix_num)
    #get sorted index sentence numbers to include in article
    df_label_pred = df_label_pred.applymap(f) 

    #Return predicted summary
          #index is doc label
    df_doc = df_text[df_label_pred.index]     
    
          # return article sentences as list
    pred_summaries = [np.array(df_doc.iloc[j])       
                               [df_label_pred.iloc[j][0]].tolist()                      
                                          for j in range(len(df_label_pred))]
          #join into summary as single string
    pred_summaries = [summ_list if type(summ_list) == str else   
                      ' '.join(summ_list) for summ_list in pred_summaries]  
    
    if return_all == True:
        answer = df_label_pred.values, df_label_pred.index, pred_summaries
    else:
        answer = pred_summaries
    
    return answer

def gen_train_test_split_doc_level(Xy_doc_label, X, y, 
                                         test_ratio, folds=1, rand_seed=42):

    '''returns train doc labels, test doc labels, and train and test sets
    for features X and target Y'''
    
    
    random.seed(rand_seed)
    
    #index is doc label 
    total_docs = Xy_doc_label.max()
    train_docs_num = int(total_docs*(1-test_ratio))
    
    #for k >1, want to ensure different seeds
    rand_state_list = random.sample(range(2*folds), folds)
    
    #look through k folds
    train_test_set = []
    for state in rand_state_list:
    
        random.seed(state)
        #sample random training set and mask
        train_docs = random.sample(range(1, total_docs+1), train_docs_num)
        train_mask = np.array([x in train_docs for x in list(Xy_doc_label)])
        
        #use mask to define train and test sets
        X_train = X[train_mask]
        y_train = y[train_mask]
    
        X_test = X[~train_mask]
        y_test = y[~train_mask]
    
        Xy_doc_label_train = Xy_doc_label[train_mask]
        Xy_doc_label_test = Xy_doc_label[~train_mask]
        
        #assign all data to tuple for each pass
        data_pass = (Xy_doc_label_train, Xy_doc_label_test,
                                             X_train, X_test, y_train, y_test)
        #append results for ith fold to set 
        train_test_set.append(data_pass)
    
    #set answer tuples to final tuple as container
    train_test_set = tuple(train_test_set)

    return train_test_set


def calc_rouge_scores(pred_summaries, gold_summaries, 
                                 keys=['rouge1', 'rougeL'], use_stemmer=True):
    #Calculate rouge scores
    scorer = rouge_scorer.RougeScorer(keys, use_stemmer= use_stemmer)
    
    n = len(pred_summaries)
    
    scores = [scorer.score(pred_summaries[j], gold_summaries[j]) for 
              j in range(n)] 
    
    dict_scores={}                                                            
    for key in keys:
        dict_scores.update({key: {}})
        
    
    for key in keys:
        
        precision_list = [scores[j][key][0] for j in range(len(scores))]
        recall_list = [scores[j][key][1] for j in range(len(scores))]
        f1_list = [scores[j][key][2] for j in range(len(scores))]

        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        f1 = np.mean(f1_list)
        
        dict_results = {'recall': recall, 'precision': precision, 'f1': f1}
        
        dict_scores[key] = dict_results
        
    return dict_scores