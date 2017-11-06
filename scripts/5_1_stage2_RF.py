
import numpy as np
import pandas as pd
import feather

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, confusion_matrix, f1_score, accuracy_score

from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.utils import shuffle

import pickle


# load stage2 train and test data, labels and test_id
df_train = np.load('../cache/train_stage2_fe2.npy')
df1 = pd.read_csv('../cache/stage2_labels.csv')
y = df1['y'].values

df_test = np.load('../cache/test_stage2_fe2.npy')
df = pd.read_csv('../cache/stage2_test_id.csv')
pid = df.ID.values

# load wts per class
wts_per_class = np.load('../cache/stage2_train_weights_per_class.npy')
wts_per_class = wts_per_class.tolist()
print(wts_per_class)

# get CV scores
x1 = np.load('../cache/train_stage2_x1.npy')
x2 = np.load('../cache/train_stage2_x2.npy')
y1 = np.load('../cache/train_stage2_y1.npy')
y2 = np.load('../cache/train_stage2_y2.npy')

w1 = np.array([wts_per_class[j] for j in y1], )
w2 = np.array([wts_per_class[j] for j in y2], )

clf1 = RandomForestClassifier(random_state=0, max_features='log2', n_estimators=700)

clf1.fit(x1, y1, sample_weight=w1)

test_preds = clf1.predict_proba(x2)
print(test_preds)


score2 = log_loss(y2, test_preds, labels = list(range(1,10)))
print('CV multi_log_loss: {}'.format(score2))

fscore = f1_score(y2, test_preds.argmax(axis=1)+1, labels = list(range(1,10)), average='micro')
print('CV f1_score: {}'.format(fscore))

acc = accuracy_score(y2, test_preds.argmax(axis=1)+1)
print('CV accuracy: {}'.format(acc))

print(confusion_matrix(y2, test_preds.argmax(axis=1)+1, labels = list(range(1,10))))

# CV multi_log_loss: 1.0051730722455978
# CV f1_score: 0.7073170731707317
# CV accuracy: 0.7073170731707317
# [[ 97   0   1  19   5   1   6   0   0]
#  [  4  43   0   1   0   2  56   0   0]
#  [  0   0   4   6   1   0  13   0   0]
#  [ 12   1   1 114   0   0  13   0   0]
#  [ 16   1   0   4  26   2   5   0   0]
#  [  8   0   0   2   2  34   9   0   0]
#  [  1  13   2   3   2   0 200   0   0]
#  [  0   0   0   0   0   0   2   0   0]
#  [  0   0   0   0   0   0   2   0   4]]


# train a RandomForest model using GridSearchCV
param_grid = { 
    'n_estimators': [50, 200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}

model_RF = RandomForestClassifier(random_state=0, class_weight=wts_per_class)
clf = GridSearchCV(model_RF, param_grid=param_grid, scoring='neg_log_loss',cv=5,n_jobs=4)
clf.fit(df_train, y)


print(clf.best_score_)
print(clf.best_params_)
# -1.5031283568737968
# {'max_features': 'log2', 'n_estimators': 700}

pred_proba = clf.predict_proba(df_test)

# make submission
df = pd.DataFrame(pred_proba)
l = []
for cls in range(len(n_classes)):
    l.append('class'+str(cls+1))
df.columns = l
df['ID'] = pid
df.to_csv('../submissions/sub_stage2_rf.csv', index=False)
# scored 2.60615 on stage2 private LB and 1.61225 on stage2 public LB with weights





