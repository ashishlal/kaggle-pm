import pandas as pd
import numpy as np
import feather
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing as pe
from tqdm import tqdm

import vars_file

TRAIN_VARIANTS_FILE = '../data/training_variants'
TEST_VARIANTS_FILE = '../data/test_variants'
TRAIN_TEXT_FILE = '../data/training_text'
TEST_TEXT_FILE = '../data/test_text'

STAGE1_TRAIN_FILE = '../cache/train_stage1.feather'
STAGE1_TEST_FILE = '../cache/test_stage1.feather'

# read the train and test data
train_stage1 = pd.read_csv(TRAIN_VARIANTS_FILE)
test_stage1 = pd.read_csv(TEST_VARIANTS_FILE)
trainx_stage1 = pd.read_csv(TRAIN_TEXT_FILE, sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx_stage1 = pd.read_csv(TEST_TEXT_FILE, sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

# merge train and test data, fill NAs with ''
train_stage1 = pd.merge(train_stage1, trainx_stage1, how='left', on='ID').fillna('')
test_stage1 = pd.merge(test_stage1, testx_stage1, how='left', on='ID').fillna('')

# now write into a binary data frame
feather.write_dataframe(train_stage1, STAGE1_TRAIN_FILE)
feather.write_dataframe(test_stage1, STAGE1_TEST_FILE)