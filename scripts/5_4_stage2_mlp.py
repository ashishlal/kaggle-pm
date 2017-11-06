import numpy as np
import pandas as pd

from sklearn.metrics import log_loss, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from utils import *

from sklearn.metrics import log_loss, confusion_matrix, f1_score, accuracy_score


# load the stage2 feature engineered datasets, labels, ids and weights
df_train = np.load('../cache/train_stage2_fe2.npy')
df_test = np.load('../cache/test_stage2_fe2.npy')
df = pd.read_csv('../cache/stage2_labels.csv')
y = df['y'].values

df = pd.read_csv('../cache/stage2_test_id.csv')
pid = df.ID.values

wts_per_class = np.load('../cache/stage2_train_weights_per_class.npy')
wts_per_class = wts_per_class.tolist()
print(wts_per_class)

##########################
# MLPCLassifier submission
############################

# try with a 0 based array. Not needed. just compat with xgboost
y = y-1

# 0 based weights
wts_per_class2 = {}
wts_per_class2 = {i:wts_per_class[i+1] for i in y}
w = np.array([wts_per_class[j+1] for j in y], )


mlp_model = MLPClassifier()

parameters = {
    'solver': ['adam', 'sgd'], 
    'max_iter': [50,250,500],
    'learning_rate_init': [0.001, 0.0001, 0.01, 0.1],
    'alpha': [0.001],
    'power_t': [0.1],
    'random_state': [1337],
}

clf = GridSearchCV(mlp_model, parameters, n_jobs=4, 
                   cv=StratifiedKFold(y, 5, True),
                   scoring = 'neg_log_loss',
                   verbose=2, refit=True)
clf.fit(df_train, y)

print(clf.best_score_)
print(clf.best_params_)
# -1.8296726568
# {'alpha': 0.001, 'learning_rate_init': 0.1, 'max_iter': 50, 'power_t': 0.1, '
#  random_state': 1337, 'solver': 'adam'}

test_preds = clf.predict_proba(df_test)
print(test_preds)

test_preds = test_preds.clip(min=0.05, max=0.95) # hack for kaggle since kaggle punishes severly if you predict 
# 1 and it is 0

submission = pd.DataFrame(test_preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('../submissions/subm_xgb_stage2_mlp.csv', index=False)
# scores 2.34559 on stage2 private LB and 1.91137 on stage2 public LB

##########################
# MLPCLassifier CV scores
############################

# get CV values, load CV train and test data
x1 = np.load('../cache/train_stage2_x1.npy')
x2 = np.load('../cache/train_stage2_x2.npy')
y1 = np.load('../cache/train_stage2_y1.npy')
y2 = np.load('../cache/train_stage2_y2.npy')

best_params_ = {'alpha': 0.001, 'learning_rate_init': 0.1, 'max_iter': 50, 'power_t': 0.1, 
                'random_state': 1337, 'solver': 'adam'}

clf1 = MLPClassifier(**best_params_)


y1 = y1-1
y2 = y2-1


clf1.fit(x1, y1)
test_preds = clf1.predict_proba(x2)
test_preds = test_preds.clip(min=0.05, max=0.95)

score2 = log_loss(y2, test_preds, labels = list(range(9)))
print('CV multi_log_loss: {}'.format(score2))

fscore = f1_score(y2, test_preds.argmax(axis=1), labels = list(range(9)), average='weighted')
print('CV f1_score: {}'.format(fscore))

acc = accuracy_score(y2, test_preds.argmax(axis=1))
print('CV accuracy: {}'.format(acc))

print(confusion_matrix(y2, test_preds.argmax(axis=1), labels = list(range(9))))
# CV multi_log_loss: 1.884022720626415
# CV f1_score: 0.12713710909501086
# CV accuracy: 0.2859078590785908
# [[  0   0   0   0   0   0 133   0   0]
#  [  0   0   0   0   0   0 100   0   0]
#  [  0   0   0   0   0   0  19   0   0]
#  [  0   0   0   0   0   0 150   0   0]
#  [  0   0   0   0   0   0  53   0   0]
#  [  0   0   0   0   0   0  59   0   0]
#  [  0   0   0   0   0   0 211   0   0]
#  [  0   0   0   0   0   0   4   0   0]
#  [  0   0   0   0   0   0   9   0   0]]

