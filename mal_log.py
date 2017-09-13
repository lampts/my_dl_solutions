"""
Access log can be transformed as sequence of bytes, e.g. 1024.
RNN to model the sequence and map to n_class of problems. 
"""

import sys
import os
import json
import pandas
import numpy
import optparse
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from collections import OrderedDict

max_log_length = 1024

model = Sequential()
model.add(Embedding(num_words, 32, input_length=max_log_length))
# Prevent overfitting using dropout method of regularization
model.add(Dropout(0.5))
model.add(LSTM(64, recurrent_dropout=0.5))
model.add(Dropout(0.5))
# Condense to single binary output value
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Training set automatically split 75/25 to check validation loss/accuracy at each epoch
model.fit(X_train, Y_train, validation_split=0.25, epochs=3, batch_size=128, callbacks=[tb_callback])
# Evaluation of separate test dataset performed after training
score, acc = model.evaluate(X_test, Y_test, verbose=1, batch_size=128)
