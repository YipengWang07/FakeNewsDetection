import sys

import pickle

import nltk
import spacy

import util

import numpy as np
import pandas as pd


nlp = spacy.load('en_core_web_lg')
df_train = pd.read_csv('raw_data/fulltrain.csv', header=None)
df_train = df_train.rename(columns={0: 'Verdict', 1: 'Text'})
df_test = pd.read_csv('raw_data/balancedtest.csv', header=None)
df_test = df_test.rename(columns={0: 'Verdict', 1: 'Text'})


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

def to_structure(texts, headers):
    X = np.zeros((len(texts), len(headers)))
    for i, doc in enumerate(nlp.pipe(texts, n_process=-1)):
        X[i] = preprocess(doc, headers)
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

texts = df_train['Text'].to_list()
X_train = to_structure(texts, headers)
df_train_spacy = pd.DataFrame(X_train)
rename_dict = dict(list(zip(list(df_train_spacy.columns), headers)))
df_train_spacy.rename(columns=rename_dict, inplace=True)
df_train_spacy = pd.concat([df_train, df_train_spacy], axis=1)

texts = df_test['Text'].to_list()
X_test = to_structure(texts, headers)
df_test_spacy = pd.DataFrame(X_test)
df_test_spacy.rename(columns=rename_dict, inplace=True)
df_test_spacy = pd.concat([df_test, df_test_spacy], axis=1)

df_train_spacy.to_csv('df_train_spacy.csv', index=False)
df_test_spacy.to_csv('df_test_spacy.csv', index=False)
