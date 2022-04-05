#!/home/l/lue97/p/projectenv/bin/python3

print('running...')

from bs4 import BeautifulSoup
from keras.utils import np_utils
from scikeras.wrappers import KerasClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
import nltk
import numpy as np
import numpy as np
import pandas as pd
import pickle
import spacy
import sys
import tensorflow as tf
import util

from sklearn.utils import shuffle

import imblearn

import os

from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping


prefix_arr = ['mlp']
if '--no-dropout' in sys.argv:
    prefix_arr.append('no_dropout')
if '--no-reg' in sys.argv:
    prefix_arr.append('no_reg')
if '--no-pca' in sys.argv:
    prefix_arr.append('no_pca')
if '--noise' in sys.argv:
    prefix_arr.append('noise')
if '--undersample' in sys.argv:
    prefix_arr.append('undersample')
prefix = '_'.join(prefix_arr)

class MyCustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, val_Xs, val_ys):
        self.val_Xs = val_Xs
        self.val_ys = val_ys

    def on_epoch_end(self, epoch, logs=None):
        res_eval_1 = self.model.evaluate(self.val_Xs[0], self.val_ys[0], verbose = 2)
        res_eval_2 = self.model.evaluate(self.val_Xs[1], self.val_ys[1], verbose = 2)
        print(res_eval_1)
        print(res_eval_2)

if '--preprocess' in sys.argv:
    # PREPROCESSING + FEATURE ENGINEERING
    scaler = StandardScaler()

    nlp = spacy.load('en_core_web_lg')
    df = pd.read_csv('raw_data/fulltrain.csv', header=None)
    df_test = pd.read_csv('raw_data/balancedtest.csv', header=None)
    df = df.rename(columns={0: 'Verdict', 1: 'Text'})
    df_test = df_test.rename(columns={0: 'Verdict', 1: 'Text'})

    words, G = util.load_glove('raw_data/glove.6B.300d.txt')
    glove_len = G.shape[1]
    mapping = dict([(word, i) for i, word in enumerate(words)])

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
        row[len(headers)::] = util.instance_to_vec(mapping, G, doc)
        return row

    def to_structure(texts, headers, glove_len):
        texts = [strip_html(text) for text in texts]
        X = np.zeros((len(texts), len(headers) + glove_len))
        for i, doc in enumerate(nlp.pipe(texts, n_process=-1)):
            X[i] = preprocess(doc, headers, glove_len)
        return X

    # remove html tags
    def strip_html(text):
        soup = BeautifulSoup(text, 'lxml')
        return soup.get_text()

    headers = (['_'.join(['pos', h]) for h in list(nlp.get_pipe("tagger").labels)]
               + ['_'.join(['ents', h]) for h in list(nlp.get_pipe("ner").labels)]
               + ['_'.join(['deps', h]) for h in list(nlp.get_pipe("parser").labels)])

    texts = df['Text'].to_list()
    X_train = to_structure(texts, headers, glove_len)
    y_train = df['Verdict'].to_numpy().flatten().astype(int)
    texts_test = df_test['Text'].to_list()
    X_test = to_structure(texts_test, headers, glove_len)
    y_test = df_test['Verdict'].to_numpy()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    print(X_test.shape)

    # output data:
    # POS || ENT || DEPS || LIWC
    with open(prefix + '_X_train.pickle', 'wb') as f:
        pickle.dump(X_train, f)

    with open(prefix + '_y_train.pickle', 'wb') as f:
        pickle.dump(y_train, f)

    with open(prefix + '_X_test.pickle', 'wb') as f:
        pickle.dump(X_test, f)

    with open(prefix + '_y_test.pickle', 'wb') as f:
        pickle.dump(y_test, f)

# MODEL (Multilayer perceptron)

with open(prefix + '_X_train.pickle', 'rb') as f:
    X_train = normalize(pickle.load(f)[:,:], axis=1).astype(float)

with open(prefix + '_y_train.pickle', 'rb') as f:
    y_train = pickle.load(f)
    print('y_train', y_train.shape)
with open(prefix + '_X_test.pickle', 'rb') as f:
    X_test = normalize(pickle.load(f)[:,:], axis=1).astype(float)

with open(prefix + '_y_test.pickle', 'rb') as f:
    y_test = pickle.load(f)

if '--undersample' in sys.argv:
    from imblearn.under_sampling import RandomUnderSampler
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_train, y_train = undersample.fit_resample(X_train, y_train)

# add noise to train set
if '--noise' in sys.argv:
    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)
    for i, (m, s) in enumerate(zip(means, stds)):
        X_train[:, i] += np.random.normal(m, s / 1.0, (X_train[:, i].shape))

stratified_kfold = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
applied = False
for fold, (t_indices, v_indices) in enumerate(stratified_kfold.split(X_train, y_train)):
    X_train_, y_train_ = X_train[t_indices], y_train[t_indices]
    X_val_, y_val_ = X_train[v_indices], y_train[v_indices]

    # apply PCA
    if '--no-pca' not in sys.argv:
        pca = PCA(n_components='mle')
        X_train_ = pca.fit_transform(X_train_)
        X_val_ = pca.transform(X_val_)
        if applied is False:
            X_test = pca.transform(X_test)
            applied = True

    # X_train, y_train = shuffle(X_train, y_train)

    input_dim = X_train_.shape[1]

    output_dim = 4
    l = ((input_dim + output_dim) // 3) * 2
    # to one hot outputs
    encoder = LabelEncoder()
    encoder.fit(y_val_)
    reg_init = tf.keras.initializers.HeNormal()
    y_train_encoded = np_utils.to_categorical(encoder.transform(y_train_))
    y_val_encoded = np_utils.to_categorical(encoder.transform(y_val_))
    y_test_encoded = np_utils.to_categorical(encoder.transform(y_test))
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(l, input_dim=X_train_.shape[1], activation='relu', kernel_initializer=reg_init))
    layer_args = {
        'activation': 'relu',
        'kernel_initializer': reg_init
    }
    if '--no-reg' not in sys.argv:
        layer_args['activity_regularizer'] = tf.keras.regularizers.L2(0.01)
        layer_args['kernel_regularizer'] = tf.keras.regularizers.L2(0.01)

    if '--no-dropout' not in sys.argv:
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(l, **layer_args))
    if '--no-dropout' not in sys.argv:
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(l, **layer_args))
    if '--no-dropout' not in sys.argv:
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(l, **layer_args))
    if '--no-dropout' not in sys.argv:
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # estimator = KerasClassifier(model=model, epochs=256, batch_size=128, verbose=1)

    print(model.summary())

    model.fit(
        X_train_,
        np_utils.to_categorical(encoder.transform(y_train_)),
        validation_data=(X_val_, y_val_encoded),
        epochs=128,
        batch_size=256,
        verbose=2,
        callbacks=[MyCustomCallback((X_val_, X_test), (y_val_encoded, y_test_encoded))]
    )
    model.save('./' + prefix + '_fold_' + str(fold) + '_mlp')

    predicted = encoder.inverse_transform(np.argmax(model.predict(X_test), axis=1))
    print(classification_report(y_test, predicted))

if '--no-pca' not in sys.argv:
    pca = PCA(n_components='mle')
    X_train = pca.fit_transform(X_train)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(l, input_dim=X_train_.shape[1], activation='relu', kernel_initializer=reg_init))
layer_args = {
    'activation': 'relu',
    'kernel_initializer': reg_init
}
if '--no-reg' not in sys.argv:
    layer_args['activity_regularizer'] = tf.keras.regularizers.L2(0.01)
    layer_args['kernel_regularizer'] = tf.keras.regularizers.L2(0.01)

if '--no-dropout' not in sys.argv:
    model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(l, **layer_args))
if '--no-dropout' not in sys.argv:
    model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(l, **layer_args))
if '--no-dropout' not in sys.argv:
    model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(l, **layer_args))
if '--no-dropout' not in sys.argv:
    model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(
    X_train,
    np_utils.to_categorical(encoder.transform(y_train)),
    epochs=128,
    batch_size=256,
    verbose=2
)

model.save('./' + prefix + '_mlp')
predicted = encoder.inverse_transform(np.argmax(model.predict(X_test), axis=1))
print(classification_report(y_test, predicted))
