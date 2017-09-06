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

    y_out = persistent_sess.run('outputs/Sigmoid:0', feed_dict={'inputs:0': np.asarray(x_in).reshape((1,-1))})
    json_data = json.dumps({'y': y_out.tolist()})
    print("Time spent handling the request: %f" % (time.time() - start)) 
    return json_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="../export/disam/1", type=str, help="saved model folder")
    parser.add_argument("--gpu_memory", default=.2, type=float, help="GPU memory per process")
    args = parser.parse_args()
    print('Initiating Session, setting the GPU memory usage to %f' % args.gpu_memory)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    persistent_sess = tf.Session(config=sess_config)
    print('Loading the model')
    loader.load(persistent_sess, [tag_constants.SERVING], args.model_path)
    print('Starting the API')
    app.run(host='0.0.0.0')
