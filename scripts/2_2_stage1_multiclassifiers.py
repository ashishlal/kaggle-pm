
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import feather
import xgboost as xgb

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
  
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
  
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_raises
  
from sklearn.utils.testing import assert_greater
import pickle

from sklearn.metrics import log_loss, confusion_matrix, f1_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

from utils import multi_log_loss


# load the partial stage1 train and test datasets
df_train = np.load('../cache/train_p_stage1_fe2.npy')
df_test = np.load('../cache/test_p_stage1_fe2.npy')
df = pd.read_csv('../cache/stage1_labels.csv')
y = df['y'].values
df1 = pd.read_csv('../cache/stage1_p_test_labels.csv')
test_labels = df1['y'].values

wts_per_class = np.load('../cache/stage1_train_weights_per_class.npy')
wts_per_class = wts_per_class.tolist()
print(wts_per_class)

###################################################################################################
ovo = OneVsOneClassifier(svm.SVC(probability=True))
Cs = [0.01, 0.02, 0.05, 0.09]
cv = GridSearchCV(ovo, {'estimator__C': Cs})
cv.fit(df_train, y)
best_C = cv.best_estimator_.estimators_[0].C
assert_true(best_C in Cs)
print(best_C)

ovo = CalibratedClassifierCV(OneVsOneClassifier(svm.SVC(probability=True, C=0.01)))
ovo.fit(df_train, y)
test_preds = ovo.predict(df_test)
print(test_preds)

score2 = log_loss(test_labels, test_preds)
print('stage1 partial multi_log_loss: {}'.format(score2))

fscore = f1_score(test_labels, test_preds.argmax(axis=1)+1, labels = list(range(1,10)), average='micro')
print('stage1 partial f1_score: {}'.format(fscore))

acc = accuracy_score(test_labels, test_preds.argmax(axis=1)+1)
print('stage1 partial accuracy: {}'.format(acc))

print(confusion_matrix(test_labels, test_preds.argmax(axis=1)+1, labels = list(range(1,10))))

# stage1 partial multi_log_loss: 1.8186399268986517
# stage1 partial f1_score: 0.27445652173913043
# stage1 partial accuracy: 0.27445652173913043
# [[  0   0   0   0   0   0  94   0   0]
# [  0   0   0   0   0   0  46   0   0]
# [  0   0   0   0   0   0   7   0   0]
# [  0   0   0   0   0   0  65   0   0]
# [  0   0   0   0   0   0  25   0   0]
# [  0   0   0   0   0   0  22   0   0]
# [  0   0   0   0   0   0 101   0   0]
# [  0   0   0   0   0   0   2   0   0]
# [  0   0   0   0   0   0   6   0   0]]

#####################################################################################################
clf = OneVsRestClassifier(CalibratedClassifierCV(LinearSVC())).fit(df_train, df['y'].values)
test_preds = clf.predict(df_test)

score2 = log_loss(test_labels, test_preds)
print('stage1 partial multi_log_loss: {}'.format(score2))

fscore = f1_score(test_labels, test_preds.argmax(axis=1)+1, labels = list(range(1,10)), average='micro')
print('stage1 partial f1_score: {}'.format(fscore))

acc = accuracy_score(test_labels, test_preds.argmax(axis=1)+1)
print('stage1 partial accuracy: {}'.format(acc))

print(confusion_matrix(test_labels, test_preds.argmax(axis=1)+1, labels = list(range(1,10))))

# stage1 partial multi_log_loss: 1.7874586410646618
# stage1 partial f1_score: 0.28804347826086957
# stage1 partial accuracy: 0.28804347826086957
# [[ 0  0  0  5  0  0 89  0  0]
# [ 0  0  0  1  0  0 45  0  0]
# [ 0  0  0  1  0  0  6  0  0]
# [ 0  0  0  7  0  0 58  0  0]
# [ 0  0  0  2  0  0 23  0  0]
# [ 0  0  0  2  0  0 20  0  0]
# [ 0  0  0  2  0  0 99  0  0]
# [ 0  0  0  0  0  0  2  0  0]
# [ 0  0  0  1  0  0  5  0  0]]

########################################################################################################
MultiNomialNB
############################################################################################################
ovo = OneVsOneClassifier(MultinomialNB())
Y_pred =ovo.fit(abs(df_train), Y).predict_proba(df_test)

Y_pred = Y_pred.clip(min=0.05, max=0.95)

score2 = log_loss(test_labels, Y_pred, labels=range(1,10))
print('stage1 partial multi_log_loss: {}'.format(score2))

fscore = f1_score(test_labels, Y_pred.argmax(axis=1)+1, labels = list(range(1,10)), average='micro')
print('stage1 partial f1_score: {}'.format(fscore))

acc = accuracy_score(test_labels, Y_pred.argmax(axis=1)+1)
print('stage1 partial accuracy: {}'.format(acc))

print(confusion_matrix(test_labels, Y_pred.argmax(axis=1)+1, labels = list(range(1,10))))

# stage1 partial multi_log_loss: 2.9597867651212026
# stage1 partial f1_score: 0.11413043478260869
# stage1 partial accuracy: 0.11413043478260869
# [[ 9  0 23  6  0 19  3  9 25]
# [ 1  4 15  0  2 11  5  0  8]
# [ 2  0  4  0  0  0  0  0  1]
# [ 3  1 26  5  0  4  2  2 22]
# [ 2  0 10  0  2  8  2  0  1]
# [ 1  0  7  0  0  5  1  1  7]
# [ 1  6 19  1  5 14 10  6 39]
# [ 0  0  0  0  0  0  0  0  2]
# [ 0  0  3  0  0  0  0  0  3]]