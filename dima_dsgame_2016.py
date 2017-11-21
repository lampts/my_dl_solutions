def hard_normalizing(X):
    return (X - 0.5) / 0.5
  
    
def init_model():
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='valid', input_shape=(3, 64, 64)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(32, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(16, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dropout(0.3))
    model.add(Dense(32))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.3))
    model.add(Dense(4))
    model.add(Activation('softmax'))

    sgd = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    return model
