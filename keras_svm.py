# SVM using hinge loss + l2 regularizer on weight
from keras.regularizers import l2

model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes), W_regularizer=l2(0.01))
model.add(Activation('linear'))

model.compile(loss='hinge',
              optimizer='adadelta',
              metrics=['accuracy'])
