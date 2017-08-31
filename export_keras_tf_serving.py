from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.contrib.session_bundle import exporter
import keras.backend as k
import tensorflow as tf
from keras.models import load_model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


k.clear_session()
sess = tf.Session()
k.set_session(sess)

# disable loading of learning nodes
k.set_learning_phase(0)
    
export_path = './export'
builder = saved_model_builder.SavedModelBuilder(export_path)

signature = predict_signature_def(inputs={'message': new_model.layers[0].input, 'position': new_model.layers[4].input},outputs={'scores': new_model.output})

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],\
                                         signature_def_map={signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
    builder.save()
