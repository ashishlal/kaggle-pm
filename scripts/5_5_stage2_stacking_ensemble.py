
# insert the path for xgboost if needed
import numpy as np
import pandas as pd
import xgboost as xgb

from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
  
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV

from sklearn.multioutput import MultiOutputClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from mlxtend.classifier import EnsembleVoteClassifier
import copy

from sklearn.cross_validation import StratifiedKFold
from scipy.optimize import minimize
from sklearn.metrics import log_loss, confusion_matrix, f1_score, accuracy_score

import pickle

# load the stage2 feature engineered train and test files
df_train = np.load('../cache/train_stage2_fe2.npy')
df_test = np.load('../cache/test_stage2_fe2.npy')

# load stage2 labels
df = pd.read_csv('../cache/stage2_labels.csv')
y = df['y'].values

# load stage2 test id
df = pd.read_csv('../cache/stage2_test_id.csv')
pid = df.ID.values

# load stage2 wts per class
wts_per_class = np.load('../cache/stage2_train_weights_per_class.npy')
wts_per_class = wts_per_class.tolist()
print(wts_per_class)

# 0 based wts
wts_per_class2 = []
for i in range(len(wts_per_class)):
    wts_per_class2.append(wts_per_class[i+1])

print(wts_per_class2)

#################################################################
# Level 2 - stacking with Logistic Regressions as meta classifier
# parameters of all level-1 classifiers are taken from gridcv best_params_ obtained earlier
#################################################################

# fix for 0 bound array
y = y-1

w = np.array([wts_per_class2[j] for j in y], )

wts_per_class3 = {}
for i in range(len(wts_per_class)):
    wts_per_class3[i]=wts_per_class[i+1]

print(wts_per_class3)

# parameters of all level-1 classifiers are taken from gridcv best_params_ obtained earlier
clf1 = xgb.XGBClassifier(objective='multi:softprob',
                         num_class= 9,
                         eval_metric= 'logloss',
                         colsample_bytree= 0.8,
                         learning_rate= 0.03,
                         max_depth= 6,
                         min_child_weight= 5,
                         missing= -999,
                         nthread= 4,
                         seed= 1337,
                         subsample= 0.8)

clf2 = RandomForestClassifier(random_state=0, max_features='auto', n_estimators=700, class_weight='balanced')
clf3 = LGBMClassifier(learning_rate=0.03, max_bin= 100, max_depth= 7, n_estimators= 200, num_leaves= 150, 
                      class_weight='balanced')
clf4 = MLPClassifier(alpha=0.001, learning_rate_init=0.1, max_iter=50, power_t=0.1, 
          random_state=1337, solver='adam')
clf5 = CalibratedClassifierCV(OneVsOneClassifier(svm.SVC(probability=True, C=0.01, class_weight='balanced')))
clf6 = OneVsRestClassifier(CalibratedClassifierCV(LinearSVC(class_weight='balanced')))


lr = LogisticRegression(class_weight = wts_per_class3)
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4, clf5, clf6], use_probas=True,
                          meta_classifier=lr)

print('5-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, sclf],  
                      ['xgb',
                       'RF',
                       'LGBM',
                       'mlp',
                       'ovo',
                       'ovr',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, df_train, y, cv=StratifiedKFold(y, 5, True), scoring='log_loss')
    print("log_loss: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

sclf.fit(df_train,y)

Z = df_test
X = df_train

proba = sclf.predict_proba(Z)
proba = proba.clip(min=0.05, max=0.95)

df = pd.DataFrame(proba, 
                  columns=['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9'])
df['ID'] = pid
df.to_csv('../submissions/sub_stage2_stacked_clf.csv', index=False)
# 2.59964 on stage2 private LB, 2.15842 on stage2 public LB

################################################
# Level 2 - stacking with XGB as meta classifier
#################################################

lr = xgb.XGBClassifier(booster='gbtree', objective='binary:logistic', eval_metric='error', eta=0.03,
                      gamma=0, max_depth=6, min_child_weight=1, max_delta_step=0, subsample=0.8,
                      colsample_by_tree=0.8, silent=1, seed = 0)
sclf2 = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4, clf5, clf6], use_probas=True,
                          meta_classifier=lr)
sclf2.fit(X,y)

proba2 = sclf2.predict_proba(Z)
proba2 = proba2.clip(min=0.05, max=0.95)

df = pd.DataFrame(proba2, 
                  columns=['class1', 'class2', 'class3', 'class4', 'class5', 
                           'class6', 'class7', 'class8', 'class9'])

df['ID'] = pid
df.to_csv('../submissions/sub_stage2_stacked_clf2.csv', index=False)
# 2.20186, stage2 private LB, 2.10209 stage2 public LB (with XGB, RF, LGBM, MLP, OVO, OVR)

################################################
# Level 2 - ensemble with weights
#################################################

# now first find ensemble weights
# load the split train and test sets
x1 = np.load('../cache/train_stage2_x1.npy')
x2 = np.load('../cache/train_stage2_x2.npy')
y1 = np.load('../cache/train_stage2_y1.npy')
y2 = np.load('../cache/train_stage2_y2.npy')

# w1 = np.array([wts_per_class[j] for j in y1], )
# w2 = np.array([wts_per_class[j] for j in y2], )

# load weights
w1 = np.load('../cache/stage2_train_weights_per_class.npy').tolist()
w2 = np.load('../cache/stage2_train_weights_per_class.npy').tolist()

w1_p = np.array([w1[j] for j in y1], )
w2_p = np.array([w2[j] for j in y2], )

clfs = []

# define the models
clf1 = xgb.XGBClassifier(objective='multi:softprob',
                         num_class= 9,
                         eval_metric= 'logloss',
                         colsample_bytree= 0.8,
                         learning_rate= 0.03,
                         max_depth= 6,
                         min_child_weight= 5,
                         missing= -999,
                         nthread= 4,
                         seed= 1337,
                         subsample= 0.8)

clf2 = RandomForestClassifier(random_state=0, max_features='auto', n_estimators=700, class_weight='balanced')
clf3 = LGBMClassifier(learning_rate=0.03, max_bin= 100, max_depth= 7, n_estimators= 200, num_leaves= 150, 
                      class_weight='balanced')
clf4 = MLPClassifier(alpha=0.001, learning_rate_init=0.1, max_iter=50, power_t=0.1, 
          random_state=1337, solver='adam')
clf5 = CalibratedClassifierCV(OneVsOneClassifier(svm.SVC(probability=True, C=0.01, class_weight='balanced')))
clf6 = OneVsRestClassifier(CalibratedClassifierCV(LinearSVC(class_weight='balanced')))


#train the models
clf1.fit(x1, y1)
clf2.fit(x1, y1)
clf3.fit(x1, y1)
clf4.fit(x1, y1)
clf5.fit(x1, y1)
clf6.fit(x1, y1)


clfs = [clf1, clf2, clf3, clf4, clf5, clf6]

# get predictions
predictions = []
for clf in clfs:
    predictions.append(clf.predict_proba(x2))


starting_values = [0.5]*len(predictions)

#adding constraints  and a different solver 
#https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
#our weights are bound between 0 and 1
bounds = [(0,1)]*len(predictions)


def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

    return log_loss(y2, final_prediction)

res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))

# Ensamble Score: 0.8806840766897692 (using xgb, rf, lgbm, mlp, ovo, ovr)
# Best Weights: [  8.19207423e-02   2.58625200e-01   6.59454057e-01   0.00000000e+00
#    1.94758402e-20   5.81090600e-17]

w = [ 8.19207423e-02,   2.58625200e-01,   6.59454057e-01,   0.00000000e+00, 1.94758402e-20,   5.81090600e-17]

eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3, clf4, clf5, clf6], 
                              weights=w, voting='soft',
                              refit=True)


eclf.fit(X, y)

print('accuracy:', np.mean(y == eclf.predict(X)))
# accuracy: 0.934128490106 


proba3 = eclf.predict_proba(Z)
proba3 = proba3.clip(min=0.05, max=0.95)


df = pd.DataFrame(proba3, 
                  columns=['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 
                           'class7', 'class8', 'class9'])
df['ID'] =pid
df.to_csv('../submissions/sub_stage2_ensemble_clf_xgb_RF_LGBM_MLP_OVO_OVR.csv', index=False)
# 2.51517 on stage2 private LB. 1.68053 on stage2 public LB (xgb+rf+lgbm+ovo+ovr)

################################################
# Level 2 - ensemble with equal weights
#################################################

# now submit with equal weights
eclf2 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3, clf4, clf5, clf6], weights=[1,1,1,1,1,1],
                              refit=True)
eclf2.fit(X, y)

proba4 = eclf2.predict_proba(Z)
proba4 = proba4.clip(min=0.05, max=0.95)

df = pd.DataFrame(proba4, 
                  columns=['class1', 'class2', 'class3', 'class4', 'class5', 
                           'class6', 'class7', 'class8', 'class9'])

df['ID'] =pid
df.to_csv('../submissions/sub_stage2_ensemble_clf_xgb_RF_LGBM_MLP_OVO_OVR_wt_111111.csv', index=False)
# 2.35864 on stage2 private LB, 1.68106 on stage1 public LB

################################################
# Level 3 - simple average of level 2 submissions
#################################################

# level 3 - simple average of level 2 files
pred = (proba + proba2 + proba3 + proba4)/4
pred = pred.clip(min=0.05, max=0.95)
df = pd.DataFrame(pred, 
                  columns=['class1', 'class2', 'class3', 'class4', 'class5', 
                           'class6', 'class7', 'class8', 'class9'])

df['ID'] =pid
df.to_csv('../submissions/sub_stage2_final_avg.csv', index=False)
# scores 2.29762 at stage2 private LB and 1.82645 stage2 public LB
                           


