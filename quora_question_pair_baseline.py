import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed, Convolution1D, Lambda, Activation, RepeatVector, Flatten, Permute, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rc
%matplotlib inline
plt.style.use("ggplot")

from IPython.display import SVG, display
from keras.utils.visualize_util import model_to_dot, plot
from sklearn import metrics

import numpy as np

embedding_matrix = np.load(open("YOUR_EMBEDDING_PATH"))

np.random.seed(75)

VOCAB = len(word2idx)
EMBED_HIDDEN_SIZE = 300
MAX_LEN = 40
SENT_HIDDEN_SIZE = 300
ACTIVATION = 'relu'
RNN_HIDDEN_SIZE = 32
NB_FILTER = 16
NB_FILTER2 = 8
LAYERS = 2

DP = 0.2
L2 = 4e-6 #4e-6
OPTIMIZER = 'rmsprop'

# helper
def ew_maL1(x):
    return K.abs(x[0] - x[1])

# structure
embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False)
embed_pos = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False)
translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))
encode1 = Bidirectional(recurrent.LSTM(RNN_HIDDEN_SIZE, return_sequences=True, init='glorot_uniform', dropout_W=0.2, dropout_U=0.2))
encode2 = Bidirectional(recurrent.LSTM(RNN_HIDDEN_SIZE, return_sequences=False, init='glorot_uniform', dropout_W=0.2, dropout_U=0.2))
cnns = [Convolution1D(filter_length=filt, nb_filter=NB_FILTER, border_mode='same') for filt in [2, 3, 5]]
cnns2 = [Convolution1D(filter_length=filt, nb_filter=NB_FILTER, border_mode='same') for filt in [2, 3, 5]]
maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
minpool = Lambda(lambda x: K.min(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))

# input defined
message_pre = Input(shape=(MAX_LEN,), dtype='int32')
message_target = Input(shape=(MAX_LEN,), dtype='int32')

# embedding
pre = embed(message_pre)
target = embed(message_target)

# dropout
pre = Dropout(DP)(pre)
target = Dropout(DP)(target)

# keep local chunk info with cnn
local_pre = merge([maxpool(cnn(pre)) for cnn in cnns], mode='concat')
local_target = merge([maxpool(cnn(target)) for cnn in cnns], mode='concat')
local_pre_min = merge([minpool(cnn(pre)) for cnn in cnns], mode='concat')
local_target_min = merge([minpool(cnn(target)) for cnn in cnns], mode='concat')

# GLOBAL
global_pre = encode2(encode1(pre))
global_target = encode2(encode1(target))

# aggregation
joint = merge([local_pre, local_target, local_pre_min, local_target_min, global_pre, global_target], mode='concat')


# Fully connected
joint = Dense(8 * RNN_HIDDEN_SIZE, activation=ACTIVATION, W_regularizer=l2(L2) if L2 else None, init='he_normal')(joint)
joint = Dropout(DP)(joint)
joint = Dense(6 * RNN_HIDDEN_SIZE, activation=ACTIVATION, W_regularizer=l2(L2) if L2 else None, init='he_normal')(joint)
joint = Dropout(DP)(joint)

# final score
score = Dense(1, activation='sigmoid')(joint)

# plug all in one
model = Model(input=[message_pre, message_target], output=score)
model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
