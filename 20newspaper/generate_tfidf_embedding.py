#!/usr/bin/env python
# coding: utf-8

get_ipython().system('python -m spacy download en_core_web_lg')
import pandas as pd
import numpy as np

import json

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_lg

nlp = en_core_web_lg.load()


import re
import string

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import pandas as pd


# Just disable some annoying warning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



df = pd.read_csv('data/news_group20.csv')

df.count()



df.category.value_counts().plot.barh()

import re
import string

from sklearn.base import TransformerMixin

class TextPreprocessor(TransformerMixin):
    def __init__(self, text_attribute):
        self.text_attribute = text_attribute
        
    def transform(self, X, *_):
        X_copy = X.copy()
        X_copy[self.text_attribute] = X_copy[self.text_attribute].apply(self._preprocess_text)
        return X_copy
    
    def _preprocess_text(self, text):
        return self._lemmatize(self._leave_letters_only(self._clean(text)))
    
    def _clean(self, text):
        bad_symbols = '!"#%&\'*+,-<=>?[\\]^_`{|}~'
        text_without_symbols = text.translate(str.maketrans('', '', bad_symbols))

        text_without_bad_words = ''
        for line in text_without_symbols.split('\n'):
            if not line.lower().startswith('from:') and not line.lower().endswith('writes:'):
                text_without_bad_words += line + '\n'

        clean_text = text_without_bad_words
        email_regex = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
        regexes_to_remove = [email_regex, r'Subject:', r'Re:']
        for r in regexes_to_remove:
            clean_text = re.sub(r, '', clean_text)

        return clean_text
    
    def _leave_letters_only(self, text):
        text_without_punctuation = text.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(re.findall("[a-zA-Z]+", text_without_punctuation))
    
    def _lemmatize(self, text):
        doc = nlp(text)
        words = [x.lemma_ for x in [y for y in doc if not y.is_stop and y.pos_ != 'PUNCT' 
                                    and y.pos_ != 'PART' and y.pos_ != 'X']]
        return ' '.join(words)
    
    def fit(self, *_):
        return self

text_preprocessor = TextPreprocessor(text_attribute='text')
df_preprocessed = text_preprocessor.transform(df)


from sklearn.model_selection import train_test_split

train, test = train_test_split(df_preprocessed, test_size=0.3)


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(analyzer = "word", max_features=2000)

X_tfidf_train = tfidf_vectorizer.fit_transform(train['text'])
X_tfidf_test = tfidf_vectorizer.transform(test['text'])


y = train['category']
y_test = test['category']


from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score

kmeans = KMeans(n_clusters=20, random_state=0).fit(X_tfidf_train)

print("KMeans NMI score:", normalized_mutual_info_score(kmeans.labels_, y))


labels = list(set(y.to_numpy()))
dic = {k: v for v, k in enumerate(labels)}

name2digit = lambda t: dic[t]
vfunc = np.vectorize(name2digit)
y_labels = vfunc(y)
y_labels_test = vfunc(y_test)



import torch
saved_data = {"X": None, "y": None}
tfidf_embedding = torch.from_numpy(np.concatenate([X_tfidf_train.todense(),X_tfidf_test.todense()]))
target = torch.from_numpy(np.concatenate([y_labels, y_labels_test]))
saved_data["X"] = tfidf_embedding
saved_data["y"] = target
import pickle
with open("tfidf_embedding.pk", "wb") as f:
    pickle.dump(saved_data, f)

