import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, confusion_matrix, f1_score, accuracy_score

import vars_file

##############
# First load the feature engineered train and test files
#############

STAGE1_TRAIN_FE2_NPY_FILE = '../cache/train_stage1_fe2'
STAGE1_TEST_FE2_NPY_FILE = '../cache/test_stage1_fe2'
STAGE1_LABELS_CSV_FILE = '../cache/stage1_labels.csv'
STAGE1_TEST_ID_CSV_FILE = '../cache/stage1_test_id.csv'
STAGE1_WTS_PER_CLASS_NPY_FILE = '../cache/stage1_train_weights_per_class'

# load stage1 train and test feature engineered files
df_train = np.load(STAGE1_TRAIN_FE2_NPY_FILE+'.npy')

df1 = pd.read_csv(STAGE1_LABELS_CSV_FILE)
y = df1['y'].values

df_test = np.load(STAGE1_TEST_FE2_NPY_FILE+'.npy')
df = pd.read_csv(STAGE1_TEST_ID_CSV_FILE)
pid = df.ID.values

y = y - 1 #fix for zero bound array in XGBoost

# get the weights per class
wts_per_class = np.load(STAGE1_WTS_PER_CLASS_NPY_FILE)
wts_per_class = wts_per_class.tolist()

# train XGBoost on stage1 train set and make submission
denom = 0
fold = 10
for i in range(fold):
    params = {
        'eta': 0.03333,
        'max_depth': 6,
        'subsample' : 0.8,
        'colsample_bytree':0.8,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'lambda': 700,
        'seed': i,
        'silent': True,
        'tree_method': 'gpu_hist',
        'n_jobs': 4
    }
    x1, x2, y1, y2 = train_test_split(df_train, y, test_size=0.2, random_state=i)
    w1 = [wts_per_class[j+1] for j in y1]
    w2 = [wts_per_class[j+1] for j in y2]
    watchlist = [(xgb.DMatrix(x1, y1,weight=w1), 'train'), (xgb.DMatrix(x2, y2,weight=w2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1,weight=w1), 1000,  watchlist, verbose_eval=50, 
                      early_stopping_rounds=100)
    pred_val = model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit)
    score1 = log_loss(y2, pred_val, labels = list(range(9)))
    
    print('fold = {:d}'.format(i))
    print('val multi_log_loss: {}'.format(score1))
    
    fscore = f1_score(y2, pred_val.argmax(axis=1), labels = list(range(9)), average='macro')
    print('val f1_score: {}'.format(fscore))
    
    acc = accuracy_score(y2, pred_val.argmax(axis=1))
    print('val accuracy: {}'.format(acc))
    
    print(confusion_matrix(y2, pred_val.argmax(axis=1), labels = list(range(9))))
    print('----------')
    print('\n\n')
    if denom != 0:
        pred = model.predict(xgb.DMatrix(df_test))
        preds += pred
    else:
        pred = model.predict(xgb.DMatrix(df_test))
        preds = pred.copy()
    denom += 1
    submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = pid
    submission.to_csv('../submissions/sub_stage1_2_2_wt_xgb_fold_'  + str(i) + '.csv', index=False)
    
    # make submission for stage 1
    submission = pd.DataFrame(preds/denom, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = pid
    submission.to_csv('../submissions/sub_stage1_all_2_2_wt_xgb.csv', index=False)
    # scores 0.4571 on stage1 public leader board
    
# now train and test with partial solution

STAGE_PARTIAL_SOLN_FILE = '../data/stage1_solution_filtered.csv'

STAGE1_PARTIAL_TEST_IDS_CSV_FILE = '../cache/stage1_p_test_id.csv'

STAGE1_PARTIAL_TRAIN_FE2_NPY_FILE = '../cache/train_p_stage1_fe2'
STAGE1_PARTIAL_TEST_FE2_NPY_FILE = '../cache/train_p_stage1_fe2'

STAGE1_PARTIAL_TEST_SOLN_CSV_FILE = '../cache/stage1_p_test_labels.csv'


# get stage1 partial test and train feature engineered filesfor testing the model
df_train = np.load(STAGE1_PARTIAL_TRAIN_FE2_NPY_FILE+'.npy')

df1 = pd.read_csv(STAGE1_LABELS_CSV_FILE)
y = df1['y'].values

df_test = np.load(STAGE1_PARTIAL_TEST_FE2_NPY_FILE+'.npy')
df = pd.read_csv(STAGE1_PARTIAL_TEST_IDS_CSV_FILE)
pid = df.ID.values

# read the partial test solution
df = pd.read_csv(STAGE1_PARTIAL_TEST_SOLN_CSV_FILE)
test_labels = df['y'].values-1

y = y - 1 #fix for zero bound array

denom = 0
fold = 10 
for i in range(fold):
    params = {
        'eta': 0.03333,
        'max_depth': 6,
        'subsample' : 0.8,
        'colsample_bytree':0.8,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'seed': i,
        'silent': True
    }
    x1, x2, y1, y2 = train_test_split(df_train, y, test_size=0.2, random_state=i)
    w1 = [wts_per_class[j+1] for j in y1]
    w2 = [wts_per_class[j+1] for j in y2]
    watchlist = [(xgb.DMatrix(x1, y1,weight=w1), 'train'), (xgb.DMatrix(x2, y2,weight=w2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1,weight=w1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
    score1 = log_loss(y2, model.predict(xgb.DMatrix(x2), 
                                        ntree_limit=model.best_ntree_limit),  
                      labels = list(range(9)))
    print('fold = {:d}'.format(i))
    print('val multi_log_loss: {}'.format(score1))

    #if score < 0.9:
    if denom != 0:
        pred = model.predict(xgb.DMatrix(df_test))
        preds += pred
        else:
            pred = model.predict(xgb.DMatrix(df_test))
            preds = pred.copy()

            denom += 1
            score2 = log_loss(test_labels, pred, labels = list(range(9)))
            print('test multi_log_loss: {}'.format(score2))

            fscore = f1_score(test_labels, pred.argmax(axis=1), labels = list(range(9)), average='macro')
            print('test f1_score: {}'.format(fscore))

            acc = accuracy_score(test_labels, pred.argmax(axis=1))
            print('test accuracy: {}'.format(acc))

            print(confusion_matrix(test_labels, pred.argmax(axis=1), labels = list(range(9))))

            print('-------------------')
            print('\n\n')
        
        
score2 = log_loss(test_labels, pred, labels = list(range(9)))
print('test multi_log_loss: {}'.format(score2))

fscore = f1_score(test_labels, pred.argmax(axis=1), labels = list(range(9)), average='macro')
print('test f1_score: {}'.format(fscore))

acc = accuracy_score(test_labels, pred.argmax(axis=1))
print('test accuracy: {}'.format(acc))

print(confusion_matrix(test_labels, pred.argmax(axis=1), labels = list(range(9))))


####### 
# The following results were obtained with stage1 partial test data
###########

# final multi_log_loss: 1.2342428618070225
# final f1_score: 0.36933577459893246
# final accuracy: 0.5652173913043478
# Confusion Matrix:
# [[57  2  1 18 13  1  2  0  0]
# [ 3 11  0  4  0  1 27  0  0]
# [ 1  0  2  1  0  1  2  0  0]
# [17  1  1 42  0  0  4  0  0]
# [ 0  0  0  6 16  1  2  0  0]
# [ 2  1  1  3  3  5  7  0  0]
# [ 4 12  2  6  0  2 75  0  0]
# [ 0  1  0  0  0  0  1  0  0]
# [ 2  0  0  3  0  0  1  0  0]]
