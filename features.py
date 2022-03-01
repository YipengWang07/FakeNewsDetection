#!/usr/bin/env python
# coding: utf-8

# In[34]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

from collections import defaultdict

import nltk
import spacy

import numpy as np
import pandas as pd


pd.set_option('display.max_colwidth', None)

nlp = spacy.load('en_core_web_lg')
df = pd.read_csv('raw_data/fulltrain.csv', header=None)
df.rename({0: 'label', 1: 'text'}, axis=1, inplace=True)

def parse_pos_tags(document):
    tags = {}
    for tag in nlp.get_pipe("tagger").labels:
        tags[tag] = 0
    for token in document:
        try:
            tags[token.tag_] += 1
        except KeyError as e:
            pass
    return tags

def parse_ents(document):
    ents = {}
    for ent in nlp.get_pipe("ner").labels:
        ents[ent] = 0
    for ent in document.ents:
        try:
            ents[ent.label_] += 1
        except KeyError as e:
            pass
    return ents

def parse_dependencies(document):
    deps = {}
    for dep in nlp.get_pipe("parser").labels:
        deps[dep] = 0
    for token in document:
        try:
            deps[token.dep_] += 1
        except KeyError as e:
            pass
    return deps

def preprocess(doc, headers):
    row = np.zeros(len(headers))
    pos_tags = parse_pos_tags(doc)
    ents = parse_ents(doc)
    deps = parse_dependencies(doc)
    for i, header in enumerate(headers):
        if header.startswith('pos_'):
            row[i] = pos_tags[header[4::]]
            continue
        if header.startswith('ents_'):
            row[i] = ents[header[5::]]
            continue
        if header.startswith('deps_'):
            row[i] = deps[header[5::]]
            continue
        sys.exit(1)
    return row


# In[35]:


headers = (['_'.join(['pos', h]) for h in list(nlp.get_pipe("tagger").labels)]
           + ['_'.join(['ents', h]) for h in list(nlp.get_pipe("ner").labels)]
           + ['_'.join(['deps', h]) for h in list(nlp.get_pipe("parser").labels)])

texts = df['text'].to_list()

X = np.zeros((len(texts), len(headers)))

for i, doc in enumerate(nlp.pipe(texts, n_process=-1)):
    X[i] = preprocess(doc, headers)

# df_new = pd.concat([df, pd.DataFrame(X)], axis=1)
df = pd.concat([df, pd.DataFrame(X).astype('int16')], axis=1)
df.columns = ['label', 'text'] + headers


# In[36]:


df


# In[37]:


df.to_csv('fulltrain_spacy.csv', index=False)


# In[ ]:
import pickle
with open('dataframe.pickle', 'wb') as f:
    pickle.dump(df, f)

