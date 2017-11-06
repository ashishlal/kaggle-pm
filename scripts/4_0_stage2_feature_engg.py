import pandas as pd
import numpy as np
import feather
from sklearn import preprocessing as pe
from tqdm import tqdm

from utils import text_to_wordlist, get_fp
from sklearn.model_selection import train_test_split

# get weight of each class since we have imbalanced classes
def get_wt(cls):
    tot_neg_instances = df_train.shape[0] - p[cls]
    tot_pos_instances = p[cls]
    return float(tot_neg_instances/tot_pos_instances)


# for hashing
D = 2 ** 24


# concat and do feature engineering on both train and test sets
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

    # print(sorted(df_all['VL'].values))

    max_len=55 # VL has max length 55 
    for i in range(max_len+1):
        df_all['Gene_'+str(i)] = df_all['Gene'].map(lambda x: str(x[i]) if len(x)>i else '')
        df_all['Variation_'+str(i)] = df_all['Variation'].map(lambda x: str(x[i]) if len(x)>i else '')
        df_all['GeneVar_'+str(i)] = df_all['GeneVar'].map(lambda x: str(x[i]) if len(x)>i else '')

    # from https://www.kaggle.com/the1owl/redefining-treatment-0-57456
#     gene_var_lst = sorted(list(df_train.Gene.unique()) + list(df_train.Variation.unique()))
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
    df_all['hash'] = df_all['hash'].apply(lambda x: hash(x) % D)

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


# read stage2 train data
df_train = feather.read_dataframe('../cache/train_stage2.feather')

# read stage2 test data
df_test = feather.read_dataframe('../cache/test_stage2.feather')

# read stage2 target values
y = df_train['Class'].values
df_train = df_train.drop(['Class'], axis=1)

# get stage2 test ids
test_ids = df_test['ID'].values

# concat train and test and do feature engg
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = do_feature_engg(df_all)


# get back train and test
df_train = df_all.iloc[:len(df_train)]
df_test = df_all.iloc[len(df_train):]


df_train['Class'] = y

# save the train and test data in binary form
feather.write_dataframe(df_train, '../cache/train_stage2_fe.feather')
feather.write_dataframe(df_test, '../cache/test_stage2_fe.feather')

# save the stage2 test ids for future use
test_id = df_test.ID.values
df = pd.DataFrame()
df['ID'] = test_id
df.to_csv('../cache/stage2_test_id.csv', index=False)

# save stage2 labels for future use
y = df_train.Class.values
df = pd.DataFrame()
df['y'] = y
df.to_csv('../cache/stage2_labels.csv', index=False)


# process the text column
df_train['Text'] = [text_to_wordlist(w) for w in df_train['Text'].values]
df_test['Text'] = [text_to_wordlist(w) for w in df_test['Text'].values]


df_train = df_train.drop('Class',axis=1)


fp = 0
fp = get_fp(fp)

# transform the use text based columns in train and test
df_train = fp.fit_transform(df_train)
print (df_train.shape)

df_test = fp.fit_transform(df_test)
print (df_test.shape)

# save the final feature engineered data 
np.save('../cache/train_stage2_fe2', df_train)
np.save('../cache/test_stage2_fe2', df_test)



# now claculate weights as classes are imbalanced
df_train = feather.read_dataframe('../cache/train_stage2_fe.feather')
df_train['wt'] =  df_train['Class'].map(lambda s: get_wt(s))
wt = df_train.wt.values
df = pd.DataFrame()
df['wt'] = wt
df.to_csv('../cache/stage2_weights.csv', index=False)
np.save('../cache/stage2_train_weights', wt)

# claculate wt per class
my_wt = {}
n_class = 9
for cls in range(n_class):
    my_wt[cls+1] = get_wt(cls+1)
np.save('../cache/stage2_train_weights_per_class', my_wt)


# now create validation data to be used by all stage2 classifers
df_train = np.load('../cache/train_stage2_fe2.npy')
df1 = pd.read_csv('../cache/stage2_labels.csv')
y = df1['y'].values

# make sure to strify y since classes are imbalanced
x1,x2,y1,y2 = train_test_split(df_train, y, test_size=0.2, random_state=42, startify=y)


np.save('../cache/train_stage2_x1', x1)
np.save('../cache/train_stage2_x2', x2)
np.save('../cache/train_stage2_y1', y1)
np.save('../cache/train_stage2_y2', y2)

# create the weights for the CV data sets
p =pd.value_counts(y1)
df_train = pd.DataFrame(x1)
my_wt = {}
n_class = 9

for cls in range(n_class):
    my_wt[cls+1] = get_wt(cls+1)
print(my_wt)

np.save('../cache/stage2_x1_weights_per_class', my_wt)

p =pd.value_counts(y2)
df_train = pd.DataFrame(x2)
my_wt = {}
n_class = 9

for cls in range(n_class):
    my_wt[cls+1] = get_wt(cls+1)
print(my_wt)

np.save('../cache/stage2_x2_weights_per_class', my_wt)


