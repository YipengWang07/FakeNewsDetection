import sys

import pickle

import nltk
import spacy

import numpy as np
import pandas as pd


nlp = spacy.load('en_core_web_lg')
headers = (list(nlp.get_pipe("tagger").labels)
           + list(nlp.get_pipe("ner").labels)
           + list(nlp.get_pipe("parser").labels))


for tag in headers:
    print("|{} | {}|".format(tag, spacy.explain(tag)))
