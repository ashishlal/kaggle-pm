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


def create_train_txt_file(index):
    text = train.loc[index]['Text']
    if text == '':
        print('empty text: {%d}'.format(index))
    fname = '../cache/rsid/' + str(index) + '.txt'
    fdict = '{\'sourcedb\'=\'PubMed\',\'sourceid\'=\'11111111\',\'text\'=' + text + '}'
    with open(fname, 'w') as f:
        f.write(fdict)
        f.flush()
        f.close()

def create_test_txt_file(index):
    text = test.loc[index]['Text']
    if text == '':
        print('empty text: {%d}'.format(index))
    fname = '../cache/test_rsid/' + str(index) + '.txt'
    fdict = '{\'sourcedb\'=\'PubMed\',\'sourceid\'=\'11111111\',\'text\'=' + text + '}'
    with open(fname, 'w') as f:
        f.write(fdict)
        f.flush()
        f.close()

Parallel(n_jobs=20)(delayed(create_train_txt_file)(i) for i in range(train.shape[0]))

Parallel(n_jobs=20)(delayed(create_test_txt_file)(i) for i in range(1,test.shape[0]))

def bash_command(cmd):
    process = subprocess.Popen(cmd, shell=True, executable='/bin/bash')
    try:
        process.wait(300)
    except:
        print('timeout for {}'.format(cmd))

from os import walk

f = []
for (dirpath, dirnames, filenames) in walk('../cache/rsid'):
    f.extend(filenames)
    break

def create_train_tmvar_files(index):
    in_file = '../cache/rsid/' + str(index) + '.txt'
    out_file = '../cache/rsid/tmVar_op/' + str(index) + '.out.tmVar'
    cmd = 'python2 RESTful.client.post.py -i {} -t tmVar -e ashishlal@lazylearner.in > {}'.format(in_file, out_file)
    print('Excuting: {}'.format(cmd))
    bash_command(cmd)

Parallel(n_jobs=20)(delayed(create_train_tmvar_files)(i) for i in range(train.shape[0]))

from os import listdir
from os.path import isfile, join
mypath = '../cache/rsid/tmVar_op'
allfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


def create_test_tmvar_files(index):
    in_file = '../cache/test_rsid/' + str(index) + '.txt'
    out_file = '../cache/test_rsid/tmVar_op/' + str(index) + '.out.tmVar'
    cmd = 'python2 RESTful.client.post.py -i {} -t tmVar -e ashishlal@lazylearner.in > {}'.format(in_file, out_file)
    print('Excuting: {}'.format(cmd))
    bash_command(cmd)


for i in range(1,test.shape[0]):
    create_test_tmvar_files(i) 

train_rsid = {}
import re
train_path = '../cache/rsid/tmVar_op/'
for i in tqdm(range(train.shape[0])):
    fn = train_path + str(i) + '.out.tmVar'
    textfile = open(fn, 'r', encoding = "ISO-8859-1")
    
    filetext = textfile.read()
#     ft = filetext.decode("utf-8") 
    textfile.close()
    matches = re.findall("RS#:\d+", filetext)
    match = set(matches)
    if i == 0:
        print(matches)
        print('\n')
        print(match)
    if match != []:
        train_rsid[i] = match
    else:
        train_rsid[i] = 'None'

import pickle
with open('../cache/train_rsid.pkl', 'wb') as f:
    pickle.dump(train_rsid, f, protocol=pickle.HIGHEST_PROTOCOL)


test_rsid = {}
import re
test_path = '../cache/test_rsid/tmVar_op/'
for i in tqdm(range(test.shape[0])):
    fn = test_path + str(i) + '.out.tmVar'
    textfile = open(fn, 'r', encoding = "ISO-8859-1")
    
    filetext = textfile.read()
#     ft = filetext.decode("utf-8") 
    textfile.close()
    matches = re.findall("RS#:\d+", filetext)
    match = set(matches)
    if i == 0:
        print(matches)
        print('\n')
        print(match)
    if match != []:
        test_rsid[i] = match
    else:
        test_rsid[i] = 'None'


import pickle
with open('../cache/test_rsid.pkl', 'wb') as f:
    pickle.dump(test_rsid, f, protocol=pickle.HIGHEST_PROTOCOL)


id_list = []
rsid_list = []
for i in tqdm(range(len(train_rsid))):
    for rsid in train_rsid[i]:
        id_list.append(i)
        rsid_list.append(rsid)

df = pd.DataFrame() 
df['ID'] = id_list
df['rsid'] = rsid_list

train_stage2 = pd.merge(train_stage2, df, how='left', on='ID')
train_stage2.fillna('None', inplace=True)

feather.write_dataframe(train_stage2, '../cache/train_stage2.feather')

test_id_list = []
test_rsid_list = []
for i in tqdm(range(len(test_rsid))):
    for rsid in test_rsid[i]:
        test_id_list.append(i)
        test_rsid_list.append(rsid)

df = pd.DataFrame()
df['ID'] = test_id_list
df['rsid'] = test_rsid_list
df['ID'] = df['ID'].apply(lambda x: x+1)

df_compat = df[df.rsid.isin(rsid_list)]

test_id_list1 = []
test_rsid_list1 = []
for i in tqdm(range(len(test_rsid))):
    l = []
    test_id_list1.append(i+1)
    for rsid in test_rsid[i]:
        if rsid in rsid_list:
            l.append(rsid)
    if len(l) > 0:
        el = random.choice(l)
        test_rsid_list1.append(el)
    else:
        test_rsid_list1.append('None')    



df = pd.DataFrame()
df['ID'] = test_id_list1
df['rsid'] = test_rsid_list1

test_stage2['rsid'] = df['rsid'].values

feather.write_dataframe(test_stage2, '../cache/test_stage2.feather')
