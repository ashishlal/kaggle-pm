import re
import numpy as np
import pandas as pd
import feather
import xgboost as xgb
import feather

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss

from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.utils import shuffle
from sklearn.metrics import log_loss, confusion_matrix, f1_score, accuracy_score
import pickle

# load train and test feature engineered files
STAGE1_TRAIN_FE2_NPY_FILE = '../cache/train_stage1_fe2'
STAGE1_TEST_FE2_NPY_FILE = '../cache/test_stage1_fe2'
STAGE1_LABELS_CSV_FILE = '../cache/stage1_labels.csv'
STAGE1_TEST_ID_CSV_FILE = '../cache/stage1_test_id.csv'
STAGE1_WTS_PER_CLASS_NPY_FILE = '../cache/stage1_train_weights_per_class'

df_train = np.load(STAGE1_TRAIN_FE2_NPY_FILE + '.npy')
df1 = pd.read_csv(STAGE1_LABELS_CSV_FILE)
y = df1['y'].values
df_test = np.load(STAGE1_TEST_FE2_NPY_FILE + '.npy')
df = pd.read_csv(STAGE1_TEST_ID_CSV_FILE)
pid = df.ID.values

# get the weights per class
wts_per_class = np.load(STAGE1_WTS_PER_CLASS_NPY_FILE + '.npy')
wts_per_class = wts_per_class.tolist()
print(wts_per_class)

# train a random forest classifier
param_grid = { 
    'n_estimators': [50, 200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}

model_RF = RandomForestClassifier(random_state=0, class_weight=wts_per_class)
clf0 = GridSearchCV(model_RF, param_grid=param_grid, scoring='neg_log_loss',cv=5,n_jobs=4, refit=True)
clf0.fit(df_train, y)

print(clf0.best_score_)
print(clf0.best_params_)

# The following result was obtained -
# -1.7355450181453207
# {'max_features': 'log2', 'n_estimators': 700}

pred = clf0.predict_proba(df_test)
submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('../submissions/sub_stage1_wt_RF'  + str(i) + '.csv', index=False)

# now get stage1 partial test and train feature engineered filesfor testing the model
df_train = np.load(STAGE1_PARTIAL_TRAIN_FE2_NPY_FILE + '.npy')
df1 = pd.read_csv(STAGE1_LABELS_CSV_FILE)
y = df1['y'].values
df_test = np.load(STAGE1_PARTIAL_TEST_FE2_NPY_FILE + '.npy')
df = pd.read_csv(STAGE1_PARTIAL_TEST_IDS_CSV_FILE)
pid = df.ID.values

# read the partial test solution
df = pd.read_csv(STAGE1_PARTIAL_TEST_SOLN_CSV_FILE)
test_labels = df['y'].values

param_grid = { 
    'n_estimators': [700, 7000, 12000],
    'max_features': ['auto', 'sqrt', 'log2']
}

# train a RF model and check against the partial solution
model_RF = RandomForestClassifier(random_state=0, class_weight=wts_per_class)
clf = GridSearchCV(model_RF, param_grid=param_grid, scoring='neg_log_loss',
                   cv=StratifiedKFold(y, 5, shuffle=True),n_jobs=4, 
                   refit=True)
clf.fit(df_train, y)


print(clf.best_score_)
print(clf.best_params_)
#-0.9071905760632368
#{'max_features': 'auto', 'n_estimators': 12000}

preds_proba = clf.predict_proba(df_test)

score2 = log_loss(test_labels, preds_proba, labels = list(range(1,10)))
print('Partial test multi_log_loss: {}'.format(score2))

fscore = f1_score(test_labels, preds_proba.argmax(axis=1)+1, labels = list(range(1,10)), average='macro')
print('Partial test f1_score: {}'.format(fscore))

acc = accuracy_score(test_labels, preds_proba.argmax(axis=1)+1)
print('Partial test accuracy: {}'.format(acc))

print(confusion_matrix(test_labels, preds_proba.argmax(axis=1)+1, labels = list(range(1,10))))

# The following result was obtained
# Partial test multi_log_loss: 1.2902553093141045
# Partial test f1_score: 0.4242180806500055
# Partial test accuracy: 0.6086956521739131
# [[ 49   0   0  30   4   0  11   0   0]
# [  1   3   0   1   0   0  41   0   0]
# [  0   0   0   2   0   0   5   0   0]
# [  2   0   0  51   0   0  12   0   0]
# [  3   0   0  10   7   0   5   0   0]
# [  1   0   0   3   0  11   7   0   0]
# [  0   0   0   1   0   0 100   0   0]
# [  0   1   0   0   0   0   1   0   0]
# [  0   0   0   2   0   0   1   0   3]]

                          
