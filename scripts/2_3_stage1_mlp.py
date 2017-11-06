import numpy as np
import pandas as pd

from sklearn.metrics import log_loss, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

# load the partial stage1 train and test data
df_train = np.load('../cache/train_p_stage1_fe2.npy')
df_test = np.load('../cache/test_p_stage1_fe2.npy')
df = pd.read_csv('../cache/stage1_labels.csv')
y = df['y'].values

df1 = pd.read_csv('../cache/stage1_p_test_labels.csv')
test_labels = df1['y'].values

wts_per_class = np.load('../cache/stage1_train_weights_per_class.npy')
wts_per_class = wts_per_class.tolist()
print(wts_per_class)

mlp_model = MLPClassifier()

parameters = {
    'solver': ['adam', 'sgd'], 
    'max_iter': [50,250,500],
    'learning_rate_init': [0.001, 0.0001, 0.01, 0.1]
    'alpha': [0.001],
    'power_t': [0.1],
    'random_state': [1337],
}

clf = GridSearchCV(mlp_model, parameters, n_jobs=4, 
                   cv=5,
                   scoring = 'neg_log_loss',
                   verbose=2, refit=True)

clf.fit(df_train, y)
print(clf.best_score_)
print(clf.best_params_)
# -2.19642899615
# {'alpha': 0.001, 'max_iter': 50, 'power_t': 0.1, 'random_state': 1337, 'solver': 'sgd'}


test_preds = clf.predict_proba(df_test)
print(test_preds)
print(test_preds.argmax(axis=1)+1)

score2 = log_loss(test_labels, test_preds, labels = list(range(1,10)))
print('partial stage1 MLP multi_log_loss: {}'.format(score2))

fscore = f1_score(test_labels, test_preds.argmax(axis=1)+1, labels = list(range(1,10)), average='macro')
print('partial stage1 MLP f1_score: {}'.format(fscore))

acc = accuracy_score(test_labels, test_preds.argmax(axis=1)+1)
print('partial stage1 MLP accuracy: {}'.format(acc))

print(confusion_matrix(test_labels, test_preds.argmax(axis=1)+1, labels = list(range(1,10))))

# partial stage1 MLP multi_log_loss: 2.197224577336219
# partial stage1 MLP f1_score: 0.024691358024691357
# partial stage1 MLP accuracy: 0.125
# [[  0  94   0   0   0   0   0   0   0]
# [  0  46   0   0   0   0   0   0   0]
# [  0   7   0   0   0   0   0   0   0]
# [  0  65   0   0   0   0   0   0   0]
# [  0  25   0   0   0   0   0   0   0]
# [  0  22   0   0   0   0   0   0   0]
# [  0 101   0   0   0   0   0   0   0]
# [  0   2   0   0   0   0   0   0   0]
# [  0   6   0   0   0   0   0   0   0]]
