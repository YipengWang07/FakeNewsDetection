import sys

import pickle

import nltk
import spacy

import util

import numpy as np
import pandas as pd


nlp = spacy.load('en_core_web_lg')
df = pd.read_csv('data/train.csv', header=None)
df_test = pd.read_csv('data/balancedtest.csv', header=None)
df = df.rename(columns={0: 'Verdict', 1: 'Text'})
df_test = df_test.rename(columns={0: 'Verdict', 1: 'Text'})

words, G = util.load_glove('data/glove.6B.300d.txt')
glove_len = G.shape[1]
mapping = dict([(word, i) for i, word in enumerate(words)])


def undersample(dataframe, col_name):
    nmin = df[col_name].value_counts().min()
    return (dataframe
            .groupby(col_name)
            .apply(lambda x: x.sample(nmin))
            .reset_index(drop=True)
            )

# pos tagger
def parse_pos_tags(document):
    tags = {}
    for tag in nlp.get_pipe("tagger").labels:
        tags[tag] = 0
    for token in document:
        try:
            tags[token.tag_] += 1
        except:
            pass
    return tags

# named entity recognition
def parse_ents(document):
    ents = {}
    for ent in nlp.get_pipe("ner").labels:
        ents[ent] = 0
    for ent in document.ents:
        try:
            ents[ent.label_] += 1
        except:
            pass
    return ents

# dependency parser
def parse_dependencies(document):
    deps = {}
    for dep in nlp.get_pipe("parser").labels:
        deps[dep] = 0
    for token in document:
        try:
            deps[token.dep_] += 1
        except:
            pass
    return deps

def preprocess(doc, headers, glove_len):
    row = np.zeros(len(headers) + glove_len)
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
    row[len(headers)::] = util.instance_to_vec(mapping, G, doc)
    return row

def to_structure(texts, headers, glove_len):
    X = np.zeros((len(texts), len(headers) + glove_len))
    for i, doc in enumerate(nlp.pipe(texts, n_process=-1)):
        X[i] = preprocess(doc, headers, glove_len)
    return X

def write_output(labels, path):
    labels = labels.astype(int)
    with open(path, 'w') as f:
        f.truncate(0)
        f.write('Sentence_id,Verdict\n')
        for i, label in enumerate(labels):
            f.write(str(i + 1) + ',' + str(label) + '\n')


headers = (['_'.join(['pos', h]) for h in list(nlp.get_pipe("tagger").labels)]
           + ['_'.join(['ents', h]) for h in list(nlp.get_pipe("ner").labels)]
           + ['_'.join(['deps', h]) for h in list(nlp.get_pipe("parser").labels)])
texts = df['Text'].to_list()
X_train = to_structure(texts, headers, glove_len)
y_train = df['Verdict']
texts_test = df_test['Text'].to_list()
X_test = to_structure(texts_test, headers, glove_len)

with open('X_train.pickle', 'wb') as f:
    pickle.dump(X_train, f)

with open('y_train.pickle', 'wb') as f:
    pickle.dump(y_train, f)

with open('X_test.pickle', 'wb') as f:
    pickle.dump(X_test, f)

