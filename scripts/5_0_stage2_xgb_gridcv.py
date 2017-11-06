import sys
sys.path.insert(0, '/home/watts/Software/xgboost/python-package') # for xgboost


import numpy as np
import pandas as pd
import feather
import xgboost as xgb
import feather

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV

import pickle

# load the stage2 train and test data
df_train = np.load('../cache/train_stage2_fe2.npy')

df1 = pd.read_csv('../cache/stage2_labels.csv')
y = df1['y'].values

df_test = np.load('../cache/test_stage2_fe2.npy')
df = pd.read_csv('../cache/stage2_test_id.csv')
pid = df.ID.values

# load sample weights
wts = np.load('../cache/stage2_train_weights.npy')


xgb_model = xgb.XGBClassifier(objective='multi:softprob', sample_weight=wts)
xgbParams = xgb_model.get_xgb_params()
xgbParams['num_class'] = 9
parameters = {
    'learning_rate': [0.03, 0.035, 0.05], #so called `eta` value
    'max_depth': [5,6,10],
    'min_child_weight': [1,5,10],
    'silent': [1],
    'subsample': [0.8, 0.7],
    'colsample_bytree': [0.8, 0.7],
    'missing':[-999],
    'seed': [1337]
}


clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 
                   cv=StratifiedKFold(y, 5),
                   scoring = 'neg_log_loss',
                   verbose=2, refit=True)
clf.fit(df_train, y)

print(clf.best_score_)
print(clf.best_params_)

# 0.5538086202222825
# {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 10, 'min_child_weight': 1, 'missing': -999, 'seed': 1337, #'silent': 1, 'subsample': 0.8}


test_probs = clf.predict_proba(df_test)

submission = pd.DataFrame(test_probs, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('../submissions/subm_xgb_stage2_with_wt_gridcv.csv', index=False)
# scored 1.72781 on the public leaderboard, 2.48738 on the private leaderboard

#########################################################################################################################
# now get CV scores
wts_per_class = np.load('../cache/stage2_train_weights_per_class.npy')
wts_per_class = wts_per_class.tolist()

denom = 0
fold = 10 
for i in range(fold):
    params = {
        'eta': 0.05,
        'max_depth': 10,
        'subsample' : 0.8,
        'colsample_bytree':0.8,
        'min_child_weight': 1,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'seed': i,
        'tree_method': 'gpu_hist',
        'silent': True
    }
    x1, x2, y1, y2 = train_test_split(df_train, y, test_size=0.2, random_state=i)
    
    w1 = [wts_per_class[j+1] for j in y1]
    w2 = [wts_per_class[j+1] for j in y2]
    watchlist = [(xgb.DMatrix(x1, y1, weight=w1), 'train'), (xgb.DMatrix(x2, y2, weight=w2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1, weight=w1), 1000,  watchlist, 
                      verbose_eval=50, early_stopping_rounds=100)
    pred_val =  model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit)
    score1 = log_loss(y2, pred_val, labels = list(range(9)))
    
    print('fold = {:d}'.format(i))
    print('val multi_log_loss: {}'.format(score1))
    
    fscore = f1_score(y2, pred_val.argmax(axis=1), labels = list(range(9)), average='macro')
    print('val f1_score: {}'.format(fscore))
    
    acc = accuracy_score(y2, pred_val.argmax(axis=1))
    print('val accuracy: {}'.format(acc))
    
    print(confusion_matrix(y2, pred_val.argmax(axis=1), labels = list(range(9))))
    
    print('-------------------')
    print('\n\n')
    #if score < 0.9:
    if denom != 0:
        pred = model.predict(xgb.DMatrix(df_test), ntree_limit=model.best_ntree_limit+80)
        preds += pred
    else:
        pred = model.predict(xgb.DMatrix(df_test), ntree_limit=model.best_ntree_limit+80)
        preds = pred.copy()
    denom += 1
    submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = pid
    submission.to_csv('../submissions/sub5_0_stage2_xgb_fold_'  + str(i) + '.csv', index=False)


# fold = 0
# val multi_log_loss: 0.9857176823950395
# val f1_score: 0.6555427505688386
# val accuracy: 0.6842818428184282
# [[ 88   3   0  15  10   9   5   1   0]
#  [  0  62   2   1   2   0  28   0   0]
#  [  0   0  15   3   2   0   5   0   0]
#  [ 30   2   7 118   9   2   4   0   0]
#  [  7   3   1   4  35   3   2   0   0]
#  [  5   3   1   1   4  33   5   0   0]
#  [  2  24  12   4   5   2 144   0   0]
#  [  0   0   0   0   0   0   0   3   2]
#  [  0   1   0   1   1   0   0   0   7]]
# -------------------



# fold = 1
# val multi_log_loss: 0.8403991454389441
# val f1_score: 0.7053603630916921
# val accuracy: 0.7344173441734417
# [[ 90   1   1  23  12   3   2   0   0]
#  [  4  67   0   3   1   0  27   0   0]
#  [  0   1  16   7   3   0   2   0   0]
#  [ 17   0   3 109   4   1   2   0   0]
#  [  3   0   0   2  37   1   6   0   0]
#  [  3   5   0   2   4  40   5   0   0]
#  [  2  25   7   5   5   0 175   0   0]
#  [  0   0   0   0   0   0   2   2   0]
#  [  0   1   0   0   0   0   0   1   6]]
# -------------------



# fold = 2
# val multi_log_loss: 0.9251795399499941
# val f1_score: 0.6566511272579925
# val accuracy: 0.6951219512195121
# [[ 85   3   1  26  14   5   6   0   0]
#  [  0  61   1   3   2   1  39   0   0]
#  [  0   0   7   5   0   0   0   0   0]
#  [ 19   0   4 113   8   1  10   0   2]
#  [  6   3   1   1  29   2   4   0   0]
#  [  3   1   0   3   5  44   5   0   0]
#  [  1  25   7   2   2   0 161   0   0]
#  [  0   1   0   0   0   0   0   3   1]
#  [  0   0   0   0   0   0   0   2  10]]
# -------------------


# fold = 3
# val multi_log_loss: 0.9256658498803533
# val f1_score: 0.6267212975469193
# val accuracy: 0.7127371273712737
# [[ 97   1   1  16  14   4   1   0   0]
#  [  2  68   0   0   2   2  19   2   0]
#  [  0   0  17   2   0   0   1   0   0]
#  [ 22   3   2  97   6   0   2   1   0]
#  [ 10   3   0   2  30   0   5   0   0]
#  [  6   9   2   2   3  42   4   0   0]
#  [  0  33  12   6   3   1 169   0   2]
#  [  0   1   0   0   0   0   1   1   2]
#  [  0   1   0   1   0   0   0   0   5]]
# -------------------


# fold = 4
# val multi_log_loss: 0.8819861687987356
# val f1_score: 0.6494354038032708
# val accuracy: 0.6924119241192412
# [[ 87   1   2  25  11   3   1   1   0]
#  [  2  65   0   0   5   3  26   0   0]
#  [  0   0  13   2   0   0   1   0   0]
#  [ 17   3   4 113   4   2   4   1   1]
#  [  9   0   1   3  37   3   4   0   0]
#  [  5   6   0   2   4  43   5   0   0]
#  [  0  39   8   4   5   3 145   0   2]
#  [  0   1   0   0   0   1   1   3   0]
#  [  0   0   0   0   1   0   0   1   5]]
# -------------------


# fold = 5
# val multi_log_loss: 0.9902080776558578
# val f1_score: 0.5875846041267859
# val accuracy: 0.6531165311653117
# [[ 90   4   3  21  15   2   2   0   0]
#  [  2  61   0   1   1   1  33   0   0]
#  [  0   1   8   4   2   0   4   0   0]
#  [ 30   1   4 101  16   3   9   1   4]
#  [  8   4   2   3  31   5   2   0   1]
#  [  8   3   1   1   2  35   4   0   0]
#  [  2  25   5   8   2   3 148   0   0]
#  [  0   1   0   0   0   0   0   1   0]
#  [  0   1   0   0   0   0   0   1   7]]
# -------------------

# fold = 6
# val multi_log_loss: 0.9942134959285014
# val f1_score: 0.6335098141628573
# val accuracy: 0.7018970189701897
# [[100   3   1  20  13   3   5   0   0]
#  [  1  62   1   1   0   2  27   0   0]
#  [  1   0  17   1   1   0   0   0   0]
#  [ 22   3   2 112   1   1   2   0   1]
#  [  7   4   1   3  32   3   2   0   0]
#  [  6   4   3   2   5  36   4   0   0]
#  [  1  34  13   4   2   2 152   0   2]
#  [  0   2   0   0   0   0   1   1   1]
#  [  0   1   0   0   0   0   1   0   6]]
# -------------------

# fold = 7
# val multi_log_loss: 0.9966859809983633
# val f1_score: 0.5512098089322359
# val accuracy: 0.6571815718157181
# [[ 75   2   2  19  12   6   2   0   0]
#  [  3  60   0   1   4   4  31   0   0]
#  [  0   0   9   0   1   0   2   0   0]
#  [ 27   3   7 112   9   1   7   0   1]
#  [ 10   3   2   2  31   4   2   0   0]
#  [  4   5   1   4   4  32   6   0   0]
#  [  0  31  13   6   2   3 156   0   0]
#  [  0   2   0   0   0   0   0   0   1]
#  [  2   0   0   0   0   0   2   0  10]]
# -------------------

# fold = 8
# val multi_log_loss: 0.9932491372372239
# val f1_score: 0.5926026132921262
# val accuracy: 0.6693766937669376
# [[ 89   2   0  19  19   5   4   0   0]
#  [  3  60   1   3   4   1  32   0   0]
#  [  0   0  16   3   2   0   2   0   0]
#  [ 25   3   4  92   4   0   6   0   1]
#  [  7   2   3   4  31   0   4   0   0]
#  [  4   3   0   1   7  47   5   0   0]
#  [  1  31  11   4   8   0 152   0   0]
#  [  2   0   0   1   0   0   1   0   1]
#  [  0   0   0   1   0   0   0   0   7]]
# -------------------

# fold = 9
# val multi_log_loss: 1.1401607413602068
# val f1_score: 0.5292003158354467
# val accuracy: 0.6395663956639567
# [[ 72   1   1  22  18   5   4   0   0]
#  [  3  67   0   3   0   2  32   0   0]
#  [  0   0  10   3   2   0   2   0   0]
#  [ 34   3   7  95   4   1  12   0   0]
#  [  3   7   1   3  31   0   8   0   0]
#  [  3   2   0   2   3  36   7   0   0]
#  [  1  35   7   4   4   2 157   0   1]
#  [  0   5   0   0   0   0   1   0   0]
#  [  0   0   0   4   1   0   1   2   4]]
# -------------------
