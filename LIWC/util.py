import sys

import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')

import pandas as pd

import numpy as np

def load_glove(path: str):
    words = []
    vectors = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tokens = line.split()
            words.append(tokens[0])
            vectors.append([float(v) for v in tokens[1::]])
        v = np.array(vectors)
        return words, v

def instance_to_vec(mapping, glove, doc):
    tokens = [token.text for token in doc if token.text not in STOPWORDS]
    indices = np.array([mapping[token] for token in tokens if token in mapping])
    try:
        embedded = np.add.reduce(glove[indices], axis=0)
    except IndexError as e:
        return np.zeros(glove.shape[1]).flatten()
    return embedded

def parse_data(mapping, glove, path: str, training=False):
    df = pd.read_csv(path)
    texts = df['Text'].tolist()
    transformed = []
    for text in texts:
        transformed.append(instance_to_vec(mapping, glove, text))
    transformed = np.array(transformed)
    if training:
        labels = df['Verdict'].to_numpy()
        return transformed, texts, labels
    else:
        return transformed, texts

def write_output(labels, path):
    labels = labels.astype(int)
    with open(path, 'w') as f:
        f.truncate(0)
        f.write('Sentence_id,Verdict\n')
        for i, label in enumerate(labels):
            f.write(''.join([str(i + 1), ',', str(label) + '\n']))
