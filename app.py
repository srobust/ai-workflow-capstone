import argparse
from flask import Flask, jsonify, request
from flask import render_template, send_from_directory
import os
import re
import joblib
import socket
import json
import numpy as np
import pandas as pd


## import model specific functions and variables
from model import model_train, model_load, model_predict
from model import MODEL_VERSION, MODEL_VERSION_NOTE

from logger import update_predict_log, update_train_log
from cslib import fetch_ts, engineer_features

def string_to_bool(string):
    false = ['false', 'f', '0']
    if string.strip().lower() in false:
        return False
    return True

app = Flask(__name__)

@app.route("/")
def landing():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/running', methods=['POST'])
def running():
    return render_template('running.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    """
    basic predict function for the API
    """
    
    ## input checking
    if not request.json:
        print("ERROR: API (predict): did not receive request data")
        return jsonify([])

    if 'query' not in request.json:
        print("ERROR API (predict): received request, but no 'query' found within")
        return jsonify([])

    if 'type' not in request.json:
        print("WARNING API (predict): received request, but no 'type' was found assuming 'numpy'")
        query_type = 'numpy'

    ## set the test flag
    test = False
    if 'mode' in request.json and request.json['mode'] == 'test':
        test = True

    ## extract the query
    query = request.json['query']
        
    if request.json['type'] == 'dict':
        pass
    else:
        print("ERROR API (predict): only dict data types have been implemented")
        return jsonify([])

    _result = model_predict(query['country'], str(query['year']), str(query['month']), str(query['day']), test=test)
    result = {}
    ## convert numpy objects to ensure they are serializable
    for key,item in _result.items():
        if isinstance(item,np.ndarray):
            result[key] = item.tolist()
        else:
            result[key] = item

    return(jsonify(result))

@app.route('/train', methods=['GET','POST'])
def train():
    """
    basic predict function for the API

    the 'mode' flag provides the ability to toggle between a test version and a 
    production version of training
    """
    
    ## check for request data
    if not request.json:
        print("ERROR: API (train): did not receive request data")
        return jsonify(False)

    ## set the test flag
    test = False
    if 'mode' in request.json and request.json['mode'] == 'test':
        test = True
    ## set the clean flag
    clean = False
    if 'clean' in request.json and string_to_bool(request.json['clean']):
        clean = True

    print("... training model")
    model = model_train(test=test, clean=clean)
    print("... training complete")

    return(jsonify(True))
        
@app.route('/logs/<filename>', methods=['GET'])
def logs(filename):
    """
    API endpoint to get logs
    """

    if not re.search(".log",filename):
        print("ERROR: API (log): file requested was not a log file: {}".format(filename))
        return jsonify([])

    log_dir = os.path.join(".","logs")
    if not os.path.isdir(log_dir):
        print("ERROR: API (log): cannot find log dir")
        return jsonify([])

    file_path = os.path.join(log_dir, filename)
    if not os.path.exists(file_path):
        print("ERROR: API (log): file requested could not be found: {}".format(filename))
        return jsonify([])
    
    return send_from_directory(log_dir, filename, as_attachment=True)

if __name__ == '__main__':

    ## parse arguments for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = vars(ap.parse_args())

    if args["debug"]:
        app.run(debug=True, port=8080)
    else:
        app.run(host='0.0.0.0', threaded=True, port=8080)

