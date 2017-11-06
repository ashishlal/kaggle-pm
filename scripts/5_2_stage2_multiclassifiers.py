import numpy as np
import pandas as pd

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
  
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV

from sklearn.grid_search import GridSearchCV

from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.testing import assert_greater
import pickle

from sklearn.metrics import log_loss, confusion_matrix, f1_score, accuracy_score

from utils import multi_log_loss

# load stage2 train and test data sets
df_train = np.load('../cache/train_stage2_fe2.npy')
df_test = np.load('../cache/test_stage2_fe2.npy')

# load stage2 labels
df = pd.read_csv('../cache/stage2_labels.csv')
y = df['y'].values

# load stage2 test id
df = pd.read_csv('../cache/stage2_test_id.csv')
pid = df.ID.values


# load the stage2 split data for CV
x1 = np.load('../cache/train_stage2_x1.npy')
x2 = np.load('../cache/train_stage2_x2.npy')
y1 = np.load('../cache/train_stage2_y1.npy')
y2 = np.load('../cache/train_stage2_y2.npy')

# load the weights per class
wts_per_class = np.load('../cache/stage2_train_weights_per_class.npy')
wts_per_class = wts_per_class.tolist()
print(wts_per_class)

# get sample weight for x1, x2 and df_train
w1 = np.load('../cache/stage2_x1_weights_per_class.npy').tolist()
w2 = np.load('../cache/stage2_x2_weights_per_class.npy').tolist()
w = np.array([wts_per_class[j] for j in y], )

print('\n')
print(w1)
print('\n')
print(w2)

############################
# OneVsOne
############################

ovo = OneVsOneClassifier(svm.SVC(probability=True))
Cs = [0.01, 0.02, 0.05, 0.09]
cv = GridSearchCV(ovo, {'estimator__C': Cs})
cv.fit(x1, y1)
best_C = cv.best_estimator_.estimators_[0].C
print(best_C)
# 0.01


ovo = CalibratedClassifierCV(OneVsOneClassifier(svm.SVC(probability=True, C=0.01, class_weight='balanced')))
ovo.fit(x1,y1)
test_preds = ovo.predict_proba(x2)


score2 = log_loss(y2, test_preds, labels = range(1,10))
print('stage2 CV multi_log_loss: {}'.format(score2))

fscore = f1_score(y2, test_preds.argmax(axis=1)+1, labels = list(range(1,10)), average='micro')
print('stage2 CV f1_score: {}'.format(fscore))

acc = accuracy_score(y2, test_preds.argmax(axis=1)+1)
print('stage2 CV accuracy: {}'.format(acc))

print(confusion_matrix(y2, test_preds.argmax(axis=1)+1, labels = list(range(1,10))))

# stage2 CV multi_log_loss: 1.8288854205271148
# stage2 CV f1_score: 0.2859078590785908
# stage2 CV accuracy: 0.2859078590785908
# [[  0   0   0   0   0   0 133   0   0]
#  [  0   0   0   0   0   0 100   0   0]
#  [  0   0   0   0   0   0  19   0   0]
#  [  0   0   0   0   0   0 150   0   0]
#  [  0   0   0   0   0   0  53   0   0]
#  [  0   0   0   0   0   0  59   0   0]
#  [  0   0   0   0   0   0 211   0   0]
#  [  0   0   0   0   0   0   4   0   0]
#  [  0   0   0   0   0   0   9   0   0]]


# train the ovo model on all data
ovo = CalibratedClassifierCV(OneVsOneClassifier(svm.SVC(probability=True, C=0.01, class_weight='balanced')))
ovo.fit(df_train, y)
test_preds = ovo.predict_proba(df_test)
test_preds = test_preds.clip(min=0.05, max=0.95)

# make submissions
df = pd.read_csv('../cache/stage2_test_id.csv')
pid = df.ID.values

submission = pd.DataFrame(test_preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('../submissions/sub_ovo.csv', index=False)
# 2.31770 on stage2 private LB, 1.91969 on stage2 public LB


############################
# OneVsRest
############################
clf = OneVsRestClassifier(CalibratedClassifierCV(LinearSVC(class_weight='balanced'))).fit(x1, y1)
test_preds = clf.predict_proba(x2)
test_preds = test_preds.clip(min=0.05, max=0.95)

score2 = log_loss(y2, test_preds, labels=[1,2,3,4,5,6,7,8,9])
print('stage2 CV multi_log_loss: {}'.format(score2))

fscore = f1_score(y2, test_preds.argmax(axis=1)+1, labels = list(range(1,10)), average='micro')
print('stage2 CV f1_score: {}'.format(fscore))

acc = accuracy_score(y2, test_preds.argmax(axis=1)+1)
print('stage2 CV accuracy: {}'.format(acc))

print(confusion_matrix(y2, test_preds.argmax(axis=1)+1, labels = list(range(1,10))))

# stage2 CV multi_log_loss: 1.8493883574649366
# stage2 CV f1_score: 0.2967479674796748
# stage2 CV accuracy: 0.2967479674796748
# [[  0   0   0  12   0   0 121   0   0]
#  [  0   0   0   5   0   0  95   0   0]
#  [  0   0   0   1   0   0  18   0   0]
#  [  0   0   0  18   0   0 132   0   0]
#  [  0   0   0   2   0   0  51   0   0]
#  [  0   0   0   1   0   0  58   0   0]
#  [  0   0   0  10   0   0 201   0   0]
#  [  0   0   0   0   0   0   4   0   0]
#  [  0   0   0   0   0   0   9   0   0]]


clf = OneVsRestClassifier(CalibratedClassifierCV(LinearSVC(class_weight='balanced'))).fit(df_train, y)
test_preds = clf.predict_proba(df_test)
test_preds = test_preds.clip(min=0.05, max=0.95)

submission = pd.DataFrame(test_preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('../submissions/sub_ovr.csv', index=False)
# 2.31000 on stage2 private LB, 1.89228 on stage2 public LB


#################################
# Naive Bayes MultinomialNB
################################

Y = df['y'].values

ovo = MultinomialNB()
Y_pred =ovo.fit(abs(df_train), Y).predict_proba(df_test)

Y_pred = Y_pred.clip(min=0.05, max=0.95)

score2 = log_loss(test_labels, Y_pred, labels=range(1,10))
print('stage2 CV multi_log_loss: {}'.format(score2))

fscore = f1_score(test_labels, Y_pred.argmax(axis=1)+1, labels = list(range(1,10)), average='micro')
print('stage2 CV f1_score: {}'.format(fscore))

acc = accuracy_score(test_labels, Y_pred.argmax(axis=1)+1)
print('stage2 CV accuracy: {}'.format(acc))

print(confusion_matrix(test_labels, Y_pred.argmax(axis=1)+1, labels = list(range(1,10))))

# stage1 partial multi_log_loss: 2.9597867651212026
# stage1 partial f1_score: 0.11413043478260869
# stage1 partial accuracy: 0.11413043478260869
# [[ 9  0 23  6  0 19  3  9 25]
#  [ 1  4 15  0  2 11  5  0  8]
#  [ 2  0  4  0  0  0  0  0  1]
#  [ 3  1 26  5  0  4  2  2 22]
#  [ 2  0 10  0  2  8  2  0  1]
#  [ 1  0  7  0  0  5  1  1  7]
#  [ 1  6 19  1  5 14 10  6 39]
#  [ 0  0  0  0  0  0  0  0  2]
#  [ 0  0  3  0  0  0  0  0  3]]



