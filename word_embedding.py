#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 16:02:30 2018

@author: s1700808
"""

import sys
import time
import numpy as np
from utils import *
from rnnmath import *
from sys import stdout
import json
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import nltk
from keras.engine import Input
from keras.layers import Embedding, merge
from keras.models import Model
import numpy as np
from nltk.corpus import stopwords
import multiprocessing
import glob
import re
from string import punctuation
from keras.preprocessing.text import text_to_word_sequence
from keras.datasets import imdb
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout,GlobalAveragePooling1D,SimpleRNN


data_folder = '/afs/inf.ed.ac.uk/user/s17/s1700808/nlu-coursework/data'
np.random.seed(2018)

vocab_size = 2000
train_size = 1000
dev_size = 1000


vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0,
                              names=['count', 'freq'], )

#vocab_new = vocab.sort_values('count',ascending=False)
num_to_word = dict(enumerate(vocab.index[:vocab_size]))
word_to_num = invert_dict(num_to_word)

sents = load_lm_dataset(data_folder + '/wiki-train.txt')
S_train = docs_to_indices(sents, word_to_num, 1, 1)
X_train, D_train = seqs_to_lmXY(S_train)

#docs = load_lm_dataset(data_folder + '/wiki-dev.txt')
#S_dev = docs_to_indices(docs, word_to_num, 1, 1)
#X_dev, D_dev = seqs_to_lmXY(S_dev)
#
#X_train = X_train[:train_size]
#D_train = D_train[:train_size]
#X_dev = X_dev[:dev_size]
#D_dev = D_dev[:dev_size]



#X_train = sequence.pad_sequences(X_train, maxlen=100)
#D_train = sequence.pad_sequences(D_train, maxlen=100)
#X_dev = sequence.pad_sequences(X_dev, maxlen=100)
#D_dev = sequence.pad_sequences(D_dev, maxlen=100)
#
#X_train_one = X_train.copy()
#D_train_one = D_train.copy()
#X_dev_one = X_dev.copy()
#D_dev_one = D_dev.copy()

#X_train_one = to_categorical(X_train, vocab_size)
#D_train_one = to_categorical(D_train, vocab_size)
#X_dev_one = to_categorical(X_dev, vocab_size)
#D_dev_one = to_categorical(D_dev, vocab_size)
    

#model = Sequential()
#embedding_layer = Embedding(vocab_size,
#                            100
#                            )
#model.add(embedding_layer)
#print ('after embedding layer the shape is:',model.output_shape)
#model.add(SimpleRNN(64,activation='sigmoid'))
#print ('after RNN layer the shape is:',model.output_shape)
#model.add(Dense(5,activation='softmax'))

print ('training begins')
skipgram_vec_model = Word2Vec(sents,size=100, window=5, min_count=5, workers=multiprocessing.cpu_count()*2, sg=1, iter=40,compute_loss=True)

print ('vector model training process is done') 
print ('vocabulary size is :', len(skipgram_vec_model.wv.index2word))
print ('the latese loss is :', skipgram_vec_model.get_latest_training_loss())

skipgram_vec_model.wv.save_word2vec_format('100d_SKIP.txt',binary=False)


#f = open('100d_CBOW.txt', encoding='utf-8')
#embeddings_index = {}
#
#for line in f:
#    values = line.split()
#    word = values[0]
#    coefs = np.asarray(values[1:], dtype='float32')
#    embeddings_index[word] = coefs
#f.close()
#print('Found %s word vectors.' % len(embeddings_index))


#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#history=model.fit(X_train_one,D_train_one , validation_data=(X_dev_one,D_dev_one), epochs=15, batch_size=32)
#scores = model.evaluate(X_dev_one, D_dev_one, verbose=0)
#print ('=====================the result for val set==============================')
#print("Loss: %.2f,  Accuracy: %.2f%%" % (scores[0],scores[1]*100))
#
#print (history.history.keys())
#
#(xx,yy),(xxx,yyy)= imdb.load_data()







