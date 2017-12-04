import numpy as np
import pandas as pd
import sys

from sklearn.metrics import log_loss, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier

# load stage2 train and test data sets
df_train = np.load('../cache/train_stage2_fe2.npy')
df_test = np.load('../cache/test_stage2_fe2.npy')
df = pd.read_csv('../cache/stage2_labels.csv')
y = df['y'].values

# load stage2 train and test ids
df = pd.read_csv('../cache/stage2_test_id.csv')
pid = df.ID.values

wts_per_class = np.load('../cache/stage2_train_weights_per_class.npy')
wts_per_class = wts_per_class.tolist()
print(wts_per_class)

################################
# LGBM with grid CV
################################
w = np.array([wts_per_class[j] for j in y],)


params = {'num_leaves':[150,200], 'max_depth':[7,10],
          'learning_rate':[0.05,0.03],'max_bin':[100,200], 'n_estimators': [200]}


lgbm = LGBMClassifier(objective='multiclass', sample_weight=w)



clf = GridSearchCV(lgbm, params, n_jobs=4, 
                   cv=StratifiedKFold(y, 5, True),
                   scoring = 'neg_log_loss',
                   verbose=2, refit=True)

clf.fit(df_train, y)


print(clf.best_score_)
print(clf.best_params_)
#-0.871917791514
# {'learning_rate': 0.03, 'max_bin': 100, 'max_depth': 7, 'n_estimators': 200, 'num_leaves': 150}

test_preds = clf.predict_proba(df_test)
print(test_preds)
test_preds = test_preds.clip(min=0.05, max=0.95) # hack for kaggle since kaggle punishes severly if you predict 
# 1 and it is 0

submission = pd.DataFrame(test_preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('../submissions/subm_xgb_stage2_lgbm.csv', index=False)
# 2.62613 on stage2 private LB, 1.77046 on stage2 public LB

################################
# LGBM CV
################################

x1 = np.load('../cache/train_stage2_x1.npy')
x2 = np.load('../cache/train_stage2_x2.npy')
y1 = np.load('../cache/train_stage2_y1.npy')
y2 = np.load('../cache/train_stage2_y2.npy')

# w1 = np.array([wts_per_class[j] for j in y1], )
# w2 = np.array([wts_per_class[j] for j in y2], )

w1 = np.load('../cache/stage2_train_weights_per_class.npy').tolist()
w2 = np.load('../cache/stage2_train_weights_per_class.npy').tolist()

w1_p = np.array([w1[j] for j in y1], )
w2_p = np.array([w2[j] for j in y2], )

best_params_ = {'learning_rate': 0.03, 'max_bin': 100, 'max_depth': 7, 'n_estimators': 200, 'num_leaves': 150}
# clf1 = LGBMClassifier(** clf.best_params_)
clf1 = LGBMClassifier(** best_params_)
clf1.fit(x1, y1, sample_weight=w1_p, eval_metric='multi_logloss')


test_preds = clf1.predict_proba(x2)
print(test_preds)
test_preds = test_preds.clip(min=0.05, max=0.95)


score2 = log_loss(y2, test_preds, labels = list(range(1,10)))
print('CV multi_log_loss: {}'.format(score2))

fscore = f1_score(y2, test_preds.argmax(axis=1)+1, labels = list(range(1,10)), average='weighted')
print('CV f1_score: {}'.format(fscore))

acc = accuracy_score(y2, test_preds.argmax(axis=1)+1)
print('CV accuracy: {}'.format(acc))

print(confusion_matrix(y2, test_preds.argmax(axis=1)+1, labels = list(range(1,10))))
# CV multi_log_loss: 1.066500698876787
# CV f1_score: 0.6842818428184282, CV f1_score: 0.6117058344203261 (macro), 0.6855632734162391 (weighted)
# CV accuracy: 0.6842818428184282
# [[ 82   3   0  22  19   5   2   0   0]
#  [  2  65   0   1   1   0  29   2   0]
#  [  1   1  11   4   0   0   2   0   0]
#  [ 23   3   1 110   6   1   5   0   1]
#  [  9   2   1   4  27   3   7   0   0]
#  [  6   4   1   5   2  38   3   0   0]
#  [  2  22  11   6   4   0 164   1   1]
#  [  0   1   0   0   0   0   2   1   0]
#  [  0   0   0   1   0   0   1   0   7]]
