from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense, AveragePooling2D,Dropout
from keras.models import Model
from keras.optimizers import RMSprop, SGD

def load_model():
  print('Loading ResNet50 Weights ...')
  resnet_50_notop = ResNet50(include_top=False, weights='imagenet',
  input_tensor=None, input_shape=(img_width, img_height, 3))
  # resnet_50_notop.summary()
  learning_rate = 0.0001
  output = resnet_50_notop.get_layer(index = -1).output # Shape: (8, 8, 2048)
  output = Flatten(name='flatten')(output)
  output = Dense(8, activation='softmax', name='predictions')(output)
  # for layer in model.layers[:25]:
  # layer.trainable = False
  model = Model(resnet_50_notop.input, output)
  # resnet_50_notop.summary()
  optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
  model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
return model
