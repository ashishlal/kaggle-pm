import csv
import os
import scipy as sp
import numpy as np

import re
from sklearn.base import BaseEstimator as be
from sklearn.base import TransformerMixin as tm
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import pickle

########################################
## process texts in datasets
########################################


# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=True, stem_words=True):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    my_stopwords = [
        "fig", "figure", "et", "al", "table",
        "data", "analysis", "analyze", "study",
        "method", "result", "conclusion", "author",
        "find", "found", "show", "perform",
        "demonstrate", "evaluate", "discuss"
    ]
    
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = [w for w in text if not w in my_stopwords]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

# from https://www.kaggle.com/the1owl/redefining-treatment-0-57456
class pm_numeric_cols(be, tm):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x.drop(['Gene', 'Variation', 'ID','Text'],axis=1).values
        return x

class pm_txt_col(be, tm):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.key].apply(str)
    
# from https://www.kaggle.com/the1owl/redefining-treatment-0-57456
def get_fp(fp):
    fp = Pipeline([
        ('union', FeatureUnion(
            n_jobs = 1,
            transformer_list = [
                ('standard', pm_numeric_cols()),
                ('pi1', Pipeline([('Gene', pm_txt_col('Gene')), 
                                           ('count_Gene', CountVectorizer(analyzer=u'char',ngram_range=(1, 8))), 
                                           ('tsvd1', TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
                ('pi2', Pipeline([('Variation', pm_txt_col('Variation')), 
                                           ('count_Variation', CountVectorizer(analyzer=u'char',ngram_range=(1, 8))), 
                                           ('tsvd2', TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),  
                ('pi4',Pipeline([('Text', pm_txt_col('Text')), 
                                       ('hv', HashingVectorizer(decode_error='ignore', n_features=2 ** 16,
                                                                non_negative=True, 
                                                                ngram_range=(1, 3))),
                                       ('tfidf_Text', TfidfTransformer()), 
                                       ('tsvd4', TruncatedSVD(n_components=300, n_iter=25, 
                                                                            random_state=12))]))

            ])
        )])
    return fp


def log_loss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def multi_log_loss(act, preds):
    scores = []
    for index in range(len(preds)):
        result = log_loss(act[index], preds[index])
        scores.append(result)

    mll = float(sum(scores) / len(scores))
    return mll

def f_score_and_accuracy(pred, labels):
    true_pos = sum(1 for i in range(len(pred)) if (pred[i]>=0.5) & (labels[i]==1))
    false_pos = sum(1 for i in range(len(pred)) if (pred[i]>=0.5) & (labels[i]==0))

    false_neg = act_pos-true_pos
    true_neg = act_neg-false_pos
    
    n1 = (true_pos+false_pos)
    n2 = (true_pos+false_neg)

    if n1 == 0:
        precision = 0.0
    else:
        precision = true_pos/(true_pos+false_pos)

    if n2 == 0:
        recall = 0.0
    else:
        recall = true_pos/(true_pos+false_neg)

    if (n1 == 0) or (n2 == 0):
        f_score = 0.0
    else:
        f_score = (2*precision*recall)/(precision+recall)  
      
    print('f-score: %f' % f_score)
    
    print('accuracy: %f' % accuracy)
    
    accuracy = (true_pos+true_neg)/(true_pos+false_pos+false_neg+true_neg)
    
    return f_score, accuracy

def print_confusion_matrix(tn, fp, fn, tp):
    print('\nconfusion matrix')
    print('----------------')
    print( 'tn:{:6d} fp:{:6d}'.format(tn,fp))
    print( 'fn:{:6d} tp:{:6d}'.format(fn,tp))

    
def save_classifier(fname, clf):
    # save the classifier
    with open(fname, 'wb') as fid:
        pickle.dump(clf, fid)

def load_classifier(fname):
    # load it again
    with open(fname, 'rb') as fid:
        clf = pickle.load(fid)
        return clf