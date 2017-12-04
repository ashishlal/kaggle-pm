import pandas as pd
import numpy as np
import feather
from sklearn import preprocessing as pe
from tqdm import tqdm

from utils import text_to_wordlist, get_fp

import vars_file

STAGE1_TRAIN_FE_FILE = '../cache/train_stage1_fe.feather'
STAGE1_TEST_FE_FILE = '../cache/test_stage1_fe.feather'

STAGE1_TEST_ID_CSV_FILE = '../cache/stage1_test_id.csv'
STAGE1_LABELS_CSV_FILE = '../cache/stage1_labels.csv'

STAGE1_WTS_CSV_FILE = '../cache/stage1_weights.csv'
STAGE1_WTS_NPY_FILE = '../cache/stage1_train_weights'

STAGE1_WTS_PER_CLASS_NPY_FILE = '../cache/stage1_train_weights_per_class'

STAGE1_TRAIN_FE2_NPY_FILE = '../cache/train_stage1_fe2'
STAGE1_TEST_FE2_NPY_FILE = '../cache/test_stage1_fe2'

STAGE_PARTIAL_SOLN_FILE = '../data/stage1_solution_filtered.csv'

STAGE1_PARTIAL_TEST_IDS_CSV_FILE = '../cache/stage1_p_test_id.csv'

STAGE1_PARTIAL_TRAIN_FE2_NPY_FILE = '../cache/train_p_stage1_fe2'
STAGE1_PARTIAL_TEST_FE2_NPY_FILE = '../cache/train_p_stage1_fe2'

STAGE1_PARTIAL_TEST_SOLN_CSV_FILE = '../cache/stage1_p_test_labels.csv'

def get_wt(cls):
    tot_neg_instances = df_train.shape[0] - p[cls]
    tot_pos_instances = p[cls]
    return float(tot_neg_instances/tot_pos_instances)

# do feature engg on the train and test sets
D = 2 ** 24
def do_feature_engg(df_all):
    
    # create a new col GeneVar
    df_all['GeneVar'] = df_all['Gene'] + ' ' + df_all['Variation']
    
    # get the share of the gene expression in the corresponding text in the row
    df_all['Gene_Share'] = df_all.apply(lambda row: sum([1 for w in row['Gene'].split(' ') 
                                                       if w in row['Text'].split(' ')]), axis=1)


    # get the share of the variation in the corresponding text in the row
    df_all['Variation_Share'] = df_all.apply(lambda row: sum([1 for w in row['Variation'].split(' ') 
                                                            if w in row['Text'].split(' ')]), axis=1)

    # get the share of the gene-variation in the corresponding text in the row
    df_all['Gene_Variation_Share'] = df_all.apply(lambda row: sum([1 for w in row['GeneVar'].split(' ') 
                                                            if w in row['Text'].split(' ')]), axis=1)

    # get the gene length
    df_all['GL'] = df_all['Gene'].apply(lambda x: len(x))

    # get the variation length
    df_all['VL'] = df_all['Variation'].apply(lambda x: len(x))

    max_len=55 # VL has max length 55 
    for i in range(max_len+1):
        df_all['Gene_'+str(i)] = df_all['Gene'].map(lambda x: str(x[i]) if len(x)>i else '')
        df_all['Variation_'+str(i)] = df_all['Variation'].map(lambda x: str(x[i]) if len(x)>i else '')
        df_all['GeneVar_'+str(i)] = df_all['GeneVar'].map(lambda x: str(x[i]) if len(x)>i else '')

    # from https://www.kaggle.com/the1owl/redefining-treatment-0-57456
    gene_var_lst = sorted(list(df_all.Gene.unique()) + list(df_all.Variation.unique()))
    gene_var_lst = [x for x in gene_var_lst if len(x.split(' '))==1]

    i_ = 0

    for el in tqdm(gene_var_lst):
        df_all['GV_'+str(el)] = df_all['Text'].map(lambda x: str(x).count(str(el)))
        i_ += 1
    
    # gene expression value counts
    s = df_all['Gene'].value_counts()
    df_all['G_VC'] = df_all['Gene'].apply(lambda gene: s[str(gene)])
    # print(df_all.G_VC[df_all.Gene == 'BRCA1'].count())

    # variation value counts
    s = df_all['Variation'].value_counts()
    df_all['V_VC'] = df_all['Variation'].apply(lambda var: s[str(var)])

    # gene variation value counts
    s = df_all['GeneVar'].value_counts()
    df_all['GV_VC'] = df_all['GeneVar'].apply(lambda var: s[str(var)])

    # hash gene variation and text
    df_all['hash'] = df_all['Gene'] + '_' + df_all['Variation'] + '_' + df_all['Text']
    df_all['hash'] = df_all['hash'].map(lambda x: hash(x) % D)

    # label encoding, length of column and number of words in the column
    for c in tqdm(df_all.columns):
        if df_all[c].dtype == 'object':
            if c in ['Gene','Variation','GeneVar']:
                lbl = pe.LabelEncoder()
                df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)  
                df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
                df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
            elif c != 'Text':
                lbl = pe.LabelEncoder()
                df_all[c] = lbl.fit_transform(df_all[c].values)
            if c=='Text': 
                df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
                df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' '))) 

    return df_all

##########
# first we do feature engineering on the stage1 files that we have prepared and store it
#################

# read the binary data frame
df_train = feather.read_dataframe(STAGE1_TRAIN_FILE)
df_test = feather.read_dataframe(STAGE1_TEST_FILE)

# get y values for use in classifiers later
y = df_train['Class'].values
df_train = df_train.drop(['Class'], axis=1)

# get test_ids too for later use (in classifers)
test_ids = df_test['ID'].values

# create a single train and test data frame
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

# do feature engineering on the combined data frame
df_all = do_feature_engg(df_all)

# get back train and test
df_train = df_all.iloc[:len(df_train)]
df_test = df_all.iloc[len(df_train):]

# restore Class col in df_train
df_train['Class'] = y
         
# store it
feather.write_dataframe(df_train, STAGE1_TRAIN_FE_FILE)
feather.write_dataframe(df_train, STAGE1_TEST_FE_FILE)

##########
# Second we do feature engineering on the stage1 train and partial solution test files
#################

# read the stage1 partial solutions file
df_stage1_partial_soln = pd.read_csv(STAGE_PARTIAL_SOLN_FILE)

cols = []
for cls in range(9):
    cols.append('class' + str(cls+1))

# append the class col
df_stage1_partial_soln['Class'] = df_stage1_partial_soln[cols].idxmax(axis=1).map(lambda x: int(x[-1]))

partial_id = df_stage1_partial_soln['ID'].values

# get the partial test data frame
df_partial_test = df_test[df_test.ID.isin(partial_id)].reset_index()

# concat the train and partial test frames
df_all_p = pd.concat((df_train, df_partial_test), axis=0, ignore_index=True)

# do feature engg
df_all_p = do_feature_engg(df_all_p)

# get back train and test data frames
df_train2 = df_all_p.iloc[:len(df_train)]
df_test2 = df_all_p.iloc[len(df_train):]

# drop the class field
df_train2 = df_train2.drop('Class',axis=1)
df_test2 = df_test2.drop('Class',axis=1)

##########
# Third we store stage1 partial test ids, stage1 full test ids, stage1 partial soln, sample weight, class weights
# for future use
#################

# store partial test ids
test_id2 = df_test2.ID.values
df2 = pd.DataFrame()
df2['ID'] = test_id2
df2.to_csv(STAGE_PARTIAL_TEST_IDS_CSV_FILE, index=False)

# stored test ids for later use
test_id = df_test.ID.values
df = pd.DataFrame()
df['ID'] = test_id
df.to_csv(STAGE_TEST_ID_CSV_FILE, index=False)

# store stage1 labels for future use
y = df_train.Class.values
df = pd.DataFrame()
df['y'] = y
df.to_csv(STAGE1_LABELS_CSV_FILE, index=False)

# also store stage1 partial solution for future use
y1 = df_stage1_partial_soln.Class.values
df2 = pd.DataFrame()
df2['y'] = y1
df2.to_csv(STAGE1_PARTIAL_TEST_SOLN_CSV_FILE, index=False)

# the classes are imbalanced. Wts for each class = tot_negative_instances/tot_positive_instances
# store the weights
df_train['wt'] =  df_train['Class'].map(lambda s: get_wt(s))
wt = df_train.wt.values
df = pd.DataFrame()
df['wt'] = wt
df.to_csv(STAGE1_WTS_CSV_FILE, index=False)
np.save(STAGE1_WTS_NPY_FILE, wt)

# store wts per class
my_wt = {}
n_class = 9
for cls in range(n_class):
    my_wt[cls+1] = get_wt(cls+1)
np.save(STAGE1_WTS_PER_CLASS_NPY_FILE, my_wt)

##########
# Fourth we do text mining using the Gene, GeneVar, Variations and Text columns
# and save the results for future use
# we do this for both stage1 train and test files as well as train and test files for partial solutions
#################

# remove stopwords from the text column, do stemming
df_train['Text'] = [text_to_wordlist(w) for w in df_train['Text'].values]
df_test['Text'] = [text_to_wordlist(w) for w in df_test['Text'].values]

# get the pipeline 
fp = 0
fp = get_fp(fp)

# drop wt and class from train
df_train = df_train.drop('Class',axis=1)
df_train = df_train.drop('wt',axis=1)

# transform the train data frame using the pipeline
df_train = fp.fit_transform(df_train)
print (df_train.shape)

# transform the test data frame using the pipeline
df_test = fp.fit_transform(df_test)
print (df_test.shape)

np.save(STAGE1_TRAIN_FE2_NPY_FILE, df_train)
np.save(STAGE1_TEST_FE2_NPY_FILE, df_test)

# Now start processing partial train and test file similarily
df_train2['Text'] = [text_to_wordlist(w) for w in df_train2['Text'].values]
df_test2['Text'] = [text_to_wordlist(w) for w in df_test2['Text'].values]

fp = 0
fp = get_fp(fp)

# transform the train and test data frames using the pipe line
df_train2 = fp.fit_transform(df_train2)
print (df_train2.shape)

df_test2 = fp.fit_transform(df_test2)
print (df_test2.shape)

np.save(STAGE1_PARTIAL_TRAIN_FE2_NPY_FILE, df_train2)
np.save(STAGE1_PARTIAL_TEST_FE2_NPY_FILE, df_test2)
