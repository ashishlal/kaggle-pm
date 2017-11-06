import pandas as pd
import numpy as np
import feather
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing as pe
from tqdm import tqdm

# read stage 1 files
train_stage1 = pd.read_csv('../data/training_variants')
test_stage1 = pd.read_csv('../data/test_variants')
trainx_stage1 = pd.read_csv('../data/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx_stage1 = pd.read_csv('../data/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

# read stage1 partial solutions
test_stage1_soln = pd.read_csv('../data/stage1_solution_filtered.csv')

#get stage1 solution IDs
stage1_soln_ids = test_stage1_soln['ID']

# prepare a single class field in the stage1 solution
test_stage1_soln = test_stage1_soln.drop('ID',axis=1)
test_stage1_soln['class'] = test_stage1_soln.idxmax(axis=1)
test_stage1_soln['class'] = test_stage1_soln['class'].map(lambda x: int(x[-1]))


test_stage1_soln['ID'] = stage1_soln_ids

# get stage1 train and test files
train_stage1 = pd.merge(train_stage1, trainx_stage1, how='left', on='ID').fillna('')
test_stage1 = pd.merge(test_stage1, testx_stage1, how='left', on='ID').fillna('')

# prepare new stage2 train file from stage1 train file and stage1 partial solution
new_test = test_stage1[test_stage1.ID.isin(stage1_soln_ids)]
new_test = new_test.drop('ID',axis=1)
new_test_ids = [i for i in range(3321, 3321+new_test.shape[0])]
new_test['ID'] = new_test_ids
new_test['Class'] = test_stage1_soln['class'].values
train_stage2 = pd.concat([train_stage1, new_test], axis=0)

# save the stage2 train file
feather.write_dataframe(train_stage2, '../cache/train_stage2.feather')


# prepare stage2 test file
test_stage2 = pd.read_csv('../data/stage2_test_variants.csv')
testx_stage2 = pd.read_csv('../data/stage2_test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

test = pd.merge(test_stage2, testx_stage2, how='left', on='ID').fillna('')

# prepare stage2 test file
feather.write_dataframe(test, '../cache/test_stage2.feather')



