from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation

def swish(x):
    return (K.sigmoid(x) * x)

def add_swish():
    get_custom_objects().update({'swish': Activation(swish)})
