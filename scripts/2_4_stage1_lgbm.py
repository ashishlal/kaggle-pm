import numpy as np
import pandas as pd
import sys

from sklearn.metrics import log_loss, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split


from lightgbm import LGBMClassifier


# get partial stage1 weights and labels
df_train = np.load('../cache/train_p_stage1_fe2.npy')
df_test = np.load('../cache/test_p_stage1_fe2.npy')
df = pd.read_csv('../cache/stage1_labels.csv')
y = df['y'].values

df1 = pd.read_csv('../cache/stage1_p_test_labels.csv')
test_labels = df1['y'].values

wts_per_class = np.load('../cache/stage1_train_weights_per_class.npy')
wts_per_class = wts_per_class.tolist()
print(wts_per_class)

# get sample weight from weights per class
w = np.array([wts_per_class[j] for j in y],)


# do a GRID search
params = {'num_leaves':[150,200], 'max_depth':[7,10],
          'learning_rate':[0.05,0.1],'max_bin':[100,200], 'n_estimators': [200]}
lgbm = LGBMClassifier(objective='multiclass', sample_weight=w)

clf = GridSearchCV(lgbm, params, n_jobs=4, 
                   cv=5,
                   scoring = 'neg_log_loss',
                   verbose=2, refit=True)
clf.fit(df_train, y)

print(clf.best_score_)
print(clf.best_params_)

# -1.87263813814
# {'learning_rate': 0.05, 'max_bin': 100, 'max_depth': 7, 'n_estimators': 200, 'num_leaves': 150}

test_preds = clf.predict_proba(df_test)
print(test_preds)

score2 = log_loss(test_labels, test_preds, labels = list(range(1,10)))
print('partial stage1 LGBM multi_log_loss: {}'.format(score2))

fscore = f1_score(test_labels, test_preds.argmax(axis=1)+1, labels = list(range(1,10)), average='micro')
print('partial stage1 LGBM f1_score: {}'.format(fscore))

acc = accuracy_score(test_labels, test_preds.argmax(axis=1)+1)
print('partial stage1 LGBM accuracy: {}'.format(acc))

print(confusion_matrix(test_labels, test_preds.argmax(axis=1)+1, labels = list(range(1,10))))

# partial stage1 LGBM multi_log_loss: 1.2407789319265683
# partial stage1 LGBM f1_score: 0.5135869565217391
# partial stage1 LGBM accuracy: 0.5135869565217391
# [[53  1  1 24 11  0  4  0  0]
#  [ 3 11  0  1  0  0 31  0  0]
#  [ 1  0  1  2  0  0  3  0  0]
#  [12  2  0 38  0  2 11  0  0]
#  [ 6  0  0  6  7  5  1  0  0]
#  [ 4  2  0  2  7  1  6  0  0]
#  [ 1 13  1  7  4  0 75  0  0]
#  [ 1  0  0  0  0  0  1  0  0]
#  [ 1  0  0  1  0  0  1  0  3]]


# now get CV scores
x1,x2,y1,y2 = train_test_split(df_train, y, test_size=0.2, random_state=42)

w1 = np.array([wts_per_class[j] for j in y1], )
w2 = np.array([wts_per_class[j] for j in y2], )

clf1 = LGBMClassifier(objective='multiclass', learning_rate= 0.03, 
                     max_bin= 100, max_depth= 7, n_estimators= 200, num_leaves= 150)

clf1.fit(x1, y1, sample_weight=w1, eval_set=[(x2, y2)], eval_sample_weight=[w2], 
        eval_metric='multi_logloss', early_stopping_rounds=100)



test_preds = clf1.predict_proba(df_test)
print(test_preds)


score2 = log_loss(test_labels, test_preds, labels = list(range(1,10)))
print('partial stage1 LGBM CV multi_log_loss: {}'.format(score2))

fscore = f1_score(test_labels, test_preds.argmax(axis=1)+1, labels = list(range(1,10)), average='micro')
print('partial stage1 LGBM CV f1_score: {}'.format(fscore))

acc = accuracy_score(test_labels, test_preds.argmax(axis=1)+1)
print('partial stage1 LGBM CV accuracy: {}'.format(acc))

print(confusion_matrix(test_labels, test_preds.argmax(axis=1)+1, labels = list(range(1,10))))

# partial stage1 LGBM CV multi_log_loss: 1.4125463050940268
# partial stage1 LGBM CV f1_score: 0.47554347826086957
# partial stage1 LGBM CV accuracy: 0.47554347826086957
# [[51  3  3 12  9 10  6  0  0]
# [ 8 12  0  2  1  1 22  0  0]
# [ 1  0  2  1  1  1  1  0  0]
# [16  2  2 35  1  2  7  0  0]
# [ 2  0  2  6  6  7  2  0  0]
# [ 2  3  0  2  2  7  6  0  0]
# [ 4 22  0  7  5  2 61  0  0]
# [ 0  1  0  0  0  0  1  0  0]
# [ 0  0  0  3  1  0  1  0  1]]





