# encoding utf-8
__author__ = 'lamp'
__version__ = '0.1'

import json, argparse, time

import tensorflow as tf
from flask import Flask, request
from flask_cors import CORS
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
import numpy as np
import cPickle as pickle

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

with open("../emb/word2idx.p", 'rb') as fp:
    word2idx = pickle.load(fp)

vocab = set(word2idx.keys())

from nlp_preprocessor import text_replace_url, text_punc_seperation, text_replace_num
from keras.preprocessing.sequence import pad_sequences
MAX_LEN = 40
TARGET = '__target__'
UNKID = word2idx['__unk__']
TARGETID = word2idx['__target__']
postprocessing = lambda x: text_replace_num(text_punc_seperation(text_replace_url(x.lower())))
mapping = lambda x, keywords: x.replace(keywords, TARGET)
tokenising = lambda x: mapping(postprocessing(x))
text2seq = lambda xs: map(lambda w: word2idx[w.replace('@','')] if w.replace('@', '') in vocab else UNKID, xs.split())
to_seq = lambda X: pad_sequences([text2seq(postprocessing(x)) for x in X], maxlen=MAX_LEN)

distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
minDistance = -30
maxDistance = 30
for dis in xrange(minDistance,maxDistance+1):
    distanceMapping[dis] = len(distanceMapping)
    
def position_feature_extract(target_seq_idx):
    try:
        target_pos = np.where(target_seq_idx.ravel()==TARGETID)[0][0]
        pos_seq_idx = np.zeros(MAX_LEN)
        for ii in xrange(0,MAX_LEN):
            if target_seq_idx[ii] == 0:
                pos_seq_idx[ii] = distanceMapping['PADDING']
            else:
                d = ii - target_pos
                if d in distanceMapping:
                    pos_seq_idx[ii] = distanceMapping[d]
                elif d <= minDistance:
                    pos_seq_idx[ii] = distanceMapping['LowerMin']
                else:
                    pos_seq_idx[ii] = distanceMapping['GreaterMax']
        return pos_seq_idx
    except Exception, e:
        return np.zeros(MAX_LEN)
    
map_to_postion_sequence = lambda X: np.asarray([position_feature_extract(s) for s in X])

app = Flask(__name__)
cors = CORS(app)
@app.route("/api/predict", methods=['POST'])
def predict():
    start = time.time()
    
    data = request.data.decode("utf-8")
    if data == "":
        params = request.form
        x_in = json.loads(params['x'])
    else:
        params = json.loads(data)
        x_in = params['x']
    
    v_target = to_seq([x_in])
    v_pos = map_to_postion_sequence(v_target)
    x_in = np.hstack([v_target,v_pos])
    r.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(x_in.astype(np.int32)))
    result = stub.Predict(r, 60.0)
    if result:
        y_out = np.asarray(result.outputs['outputs'].float_val)
    else:
        y_out = np.zeros((1,))
    json_data = json.dumps({'y': y_out.tolist()})
    print("Time spent handling the request: %f" % (time.time() - start)) 
    return json_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host_port", default="0.0.0.0:9000", type=str, help="host:port")
    args = parser.parse_args()
    print('Set host:port %s' % args.host_port)
    host, port = args.host_port.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    print('Initiated request')
    r = predict_pb2.PredictRequest()
    r.model_spec.name = 'disam'
    r.model_spec.signature_name = 'predict'
    print('Starting the API')
    app.run(host='0.0.0.0')
