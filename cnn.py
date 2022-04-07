#!/usr/bin/env python
# coding: utf-8

# In[13]:

from bs4 import BeautifulSoup
from keras.utils import np_utils
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPool1D, Embedding, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
import numpy as np
import pandas as pd
import pickle
import re
import spacy
import sys
import tensorflow

STOPWORDS = set(stopwords.words('english'))
LINK_RE = re.compile(r'(https|http|ftp)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\-)*\b')
PUNCT = ':()[]{},?!"\';.:-/\\*'

max_len = 1000
if '--max-len' in sys.argv:
    max_len = int(sys.argv[sys.argv.index('--max-len') + 1])

embed_size = 100
if '--embed-size' in sys.argv:
    embed_size = int(sys.argv[sys.argv.index('--embed-size') + 1])


prefix_arr = ['cnn']
prefix_arr.append(str(max_len))
prefix_arr.append(str(embed_size))

if '--chunked' in sys.argv:
    prefix_arr.append('chunked')
prefix = '_'.join(prefix_arr)

print(prefix)



def only_punct(text):
    for c in text:
        if c not in PUNCT:
            return False
    return True

def preprocess(texts, labels=None):
    V = set()
    tmp = [BeautifulSoup(t, 'html.parser').get_text().lower() for t in texts]
    tmp = [LINK_RE.sub('', t) for t in tmp]
    tmp = [wordpunct_tokenize(t) for t in tmp]
    tmp = [list(filter(lambda x: x not in STOPWORDS, ts)) for ts in tmp]
    tmp = [list(filter(lambda x: not only_punct(x), ts)) for ts in tmp]

    label_ = []
    if labels is not None:
        out = []
        for ts, l in zip(tmp, labels):
            if len(ts) <= max_len:
                out.append(ts)
                label_.append(l)
            else:
                chunks = [ts[i:i + max_len] for i in range(0, len(ts), max_len // 2)]
                out.extend(chunks)
                label_.extend([l] * len(chunks))
        tmp = out

    V = set.union(*[set(ts) for ts in tmp])
    L = [len(ts) for ts in tmp]
    cleaned = [' '.join(ts) for ts in tmp]
    if labels is not None:
        return V, L, cleaned, np.array(label_)
    else:
        return V, L, cleaned, None


df = pd.read_csv('raw_data/fulltrain.csv', header=None)
df = df.rename(columns={0: 'verdict', 1: 'text'})
df_test = pd.read_csv('raw_data/balancedtest.csv', header=None)
df_test = df_test.rename(columns={0: 'verdict', 1: 'text'})
if '--chunk' not in sys.argv:
    V, L, cleaned, _ = preprocess(df['text'].to_list(), None)
    df['cleaned'] = cleaned
    df['length'] = L
else:
    V, L, cleaned, _ = preprocess(df['text'].to_list(), df['verdict'].to_list())
    df_tmp = pd.DataFrame(list(zip(cleaned, _)), columns=['cleaned', 'verdict'])
    df_tmp['length'] = L
    df = df_tmp

V_test, L_test, cleaned_test, _ = preprocess(df_test['text'].to_list())
df_test['cleaned'] = cleaned_test
df_test['length'] = L_test


# kfold goes here

X = df['text'].to_list()
y = df['verdict'].to_numpy().astype(int).flatten()

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_encoded = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train_encoded, maxlen=max_len, padding='post')

X_val_encoded = tokenizer.texts_to_sequences(X_val)
X_val = pad_sequences(X_val_encoded, maxlen=max_len, padding='post')

X_test_encoded = tokenizer.texts_to_sequences(cleaned_test)
X_test = pad_sequences(X_test_encoded, maxlen=max_len, padding='post')

y_test = df_test['verdict'].to_numpy().flatten()

with open(prefix + '_V.pickle', 'wb') as f:
    pickle.dump(V, f)

with open(prefix + '_X_val.pickle', 'wb') as f:
    pickle.dump(X_val, f)

with open(prefix + '_y_val.pickle', 'wb') as f:
    pickle.dump(y_val, f)

with open(prefix + '_X_train.pickle', 'wb') as f:
    pickle.dump(X_train, f)

with open(prefix + '_X_test.pickle', 'wb') as f:
    pickle.dump(X_test, f)

with open(prefix + '_y_train.pickle', 'wb') as f:
    pickle.dump(y_train, f)

with open(prefix + '_y_test.pickle', 'wb') as f:
    pickle.dump(y_test, f)

with open(prefix + '_tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)

with open(prefix + '_V.pickle', 'rb') as f:
    V = pickle.load(f)

with open(prefix + '_X_train.pickle', 'rb') as f:
    X_train = pickle.load(f)

with open(prefix + '_X_test.pickle', 'rb') as f:
    X_test = pickle.load(f)

with open(prefix + '_y_train.pickle', 'rb') as f:
    y_train = pickle.load(f)

with open(prefix + '_y_test.pickle', 'rb') as f:
    y_test = pickle.load(f)

encoder = LabelEncoder()
encoder.fit(y_train)
X_train, y_train = shuffle(X_train, y_train)
y_train_encoded = np_utils.to_categorical(encoder.transform(y_train))
y_test_encoded = np_utils.to_categorical(encoder.transform(y_test))
y_val_encoded = np_utils.to_categorical(encoder.transform(y_val))

class MyCustomCallback(tensorflow.keras.callbacks.Callback):

    def __init__(self, val_Xs, val_ys):
        self.val_Xs = val_Xs
        self.val_ys = val_ys

    def on_epoch_end(self, epoch, logs=None):
        p1 = encoder.inverse_transform(np.argmax(model.predict(self.val_Xs[0]), axis=1))
        p2 = encoder.inverse_transform(np.argmax(model.predict(self.val_Xs[1]), axis=1))

        print('1a,', accuracy_score(self.val_ys[0], p1))
        print('1fma', f1_score(self.val_ys[0], p1,  average='macro'))
        print('1fmi', f1_score(self.val_ys[0], p1,  average='micro'))
        print('1pma', precision_score(self.val_ys[0], p1,  average='macro'))
        print('1pmi', precision_score(self.val_ys[0], p1,  average='micro'))
        print('1rma', recall_score(self.val_ys[0], p1,  average='macro'))
        print('1rmi', recall_score(self.val_ys[0], p1,  average='micro'))

        print('2a', accuracy_score(self.val_ys[1], p2))
        print('2fma', f1_score(self.val_ys[1], p2,  average='macro'))
        print('2fmi', f1_score(self.val_ys[1], p2,  average='micro'))
        print('2pma', precision_score(self.val_ys[1], p2,  average='macro'))
        print('2pmi', precision_score(self.val_ys[1], p2,  average='micro'))
        print('2rma', recall_score(self.val_ys[1], p2,  average='macro'))
        print('2rmi', recall_score(self.val_ys[1], p2,  average='micro'))

# Model
inp = Input(shape=(max_len,))
embedding = Embedding(30000, embed_size, input_length=max_len)(inp)
conv1d_1 = Conv1D(filters=128, kernel_size=3, activation='relu')(embedding)
conv1d_2 = Conv1D(filters=128, kernel_size=4, activation='relu')(embedding)
conv1d_3 = Conv1D(filters=128, kernel_size=5, activation='relu')(embedding)
maxpool1d_1 = MaxPool1D(pool_size=5)(conv1d_1)
maxpool1d_2 = MaxPool1D(pool_size=5)(conv1d_2)
maxpool1d_3 = MaxPool1D(pool_size=5)(conv1d_3)
concat = concatenate([maxpool1d_1, maxpool1d_2, maxpool1d_3], axis=1)
conv1d_4 = Conv1D(filters=128, kernel_size=5, activation='relu')(concat)
maxpool1d_4 = MaxPool1D(pool_size=5)(conv1d_4)
conv1d_5 = Conv1D(filters=128, kernel_size=5, activation='relu')(maxpool1d_4)
maxpool1d_5 = MaxPool1D(pool_size=32)(conv1d_5)
flatten = Flatten()(maxpool1d_5)
dropout1 = Dropout(0.2)(flatten)
dense1 = Dense(128, activation='relu')(dropout1)
dropout2 = Dropout(0.2)(dense1)
output = Dense(4, activation='softmax')(dropout2)
model = Model(inputs=inp, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])

model.summary()
model.fit(
    X_train,
    y_train_encoded,
    batch_size=64,
    epochs=512,
    verbose=2,
    callbacks=[MyCustomCallback((X_val, X_test), (y_val, y_test))]
)

# fit model on everything
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)


X_all_encoded = tokenizer.texts_to_sequences(X)
X_all = pad_sequences(X_all_encoded, maxlen=max_len, padding='post')
y_all_encoded = np_utils.to_categorical(encoder.transform(y))

with open(prefix + '_X_all.pickle', 'wb') as f:
    pickle.dump(X_all, f)

with open(prefix + '_y_all.pickle', 'wb') as f:
    pickle.dump(y, f)

with open(prefix + '_tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)

inp = Input(shape=(max_len,))
embedding = Embedding(30000, embed_size, input_length=max_len)(inp)
conv1d_1 = Conv1D(filters=128, kernel_size=3, activation='relu')(embedding)
conv1d_2 = Conv1D(filters=128, kernel_size=4, activation='relu')(embedding)
conv1d_3 = Conv1D(filters=128, kernel_size=5, activation='relu')(embedding)
maxpool1d_1 = MaxPool1D(pool_size=5)(conv1d_1)
maxpool1d_2 = MaxPool1D(pool_size=5)(conv1d_2)
maxpool1d_3 = MaxPool1D(pool_size=5)(conv1d_3)
concat = concatenate([maxpool1d_1, maxpool1d_2, maxpool1d_3], axis=1)
conv1d_4 = Conv1D(filters=128, kernel_size=5, activation='relu')(concat)
maxpool1d_4 = MaxPool1D(pool_size=5)(conv1d_4)
conv1d_5 = Conv1D(filters=128, kernel_size=5, activation='relu')(maxpool1d_4)
maxpool1d_5 = MaxPool1D(pool_size=32)(conv1d_5)
flatten = Flatten()(maxpool1d_5)
dropout1 = Dropout(0.2)(flatten)
dense1 = Dense(128, activation='relu')(dropout1)
dropout2 = Dropout(0.2)(dense1)
output = Dense(4, activation='softmax')(dropout2)
model = Model(inputs=inp, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
model.fit(
    X_all,
    y_all_encoded,
    batch_size=64,
    epochs=512,
    verbose=2,
    validation_data=(X_test, y_test_encoded),
    validation_freq=1,
)

model.save('./' + prefix + '_cnn')

predicted = encoder.inverse_transform(np.argmax(model.predict(X_test), axis=1))
print(classification_report(y_test, predicted))

