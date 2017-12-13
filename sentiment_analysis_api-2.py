#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from flask_api import FlaskAPI
import codecs
from itertools import chain
import requests
import base64
import numpy as np
import csv

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import load_csv
from tensorflow.python.framework.ops import reset_default_graph

from flask import Flask, jsonify, json, Response
from flask_restful import reqparse, Api, Resource
from json import encoder
from decimal import Decimal
from tflearn.data_utils import shuffle


app = Flask(__name__)
#app.config['JSON_AS_ASCII'] = False

api = Api(app)
parser = reqparse.RequestParser()
parser.add_argument('sentence')

class Thai_segment(Resource):
    
    def get(self, sentence):
        args = parser.parse_args()
        evaluation_result = self.sentiment_analysis(sentence)
        print(evaluation_result)
        data = {}
        data['positiveResult'] = str(evaluation_result[0][1])
        data['negativeResult'] = str(evaluation_result[0][0])

        if (evaluation_result[0][1]>=evaluation_result[0][0]):
            data['Label'] = "Positive"
        else:
            data['Label'] = "Negative"

        data['Sentence'] = sentence
        json_result = json.dumps(data, ensure_ascii=False)
        response = Response(json_result, content_type="application/json; charset=utf-8")
        #evaluation_result = []
        return response

    def preprocess_server_2(self, data):
        rlist = []
        preprocessdata = []
        x = requests.get('http://174.138.26.245:5000/preprocess/'+data)
        resu = x.json()
        preprocessdata.append(resu['result'])
        for i in range(len(preprocessdata)):
            r = requests.get('http://174.138.26.245:5000/tokenize/'+preprocessdata[i])
            result = r.json()
            rlist.append(result['result'])
        return rlist

    def uniqueword_csvload(self):
        uniquewords = []
        f = open('./uniqueword/Combined_inhousedata_UTF8-1_uniquewords.csv', 'r')
        for word in f:
            uniquewords.append(word.strip())
        return uniquewords   

    def sentiment_analysis(self, sentencedata):
        
        unique_words = self.uniqueword_csvload()

        neurons = len(unique_words)

        # shuffle the dataset
        # data, labels = shuffle(data, labels)

        reset_default_graph()
        network = input_data(shape=[None, neurons])
        network = conv_1d(network, 16, 3, activation='relu')
        network = max_pool_1d(network, 3)

        network = conv_1d(network, 32, 3, activation='relu')
        network = max_pool_1d(network, 3)
        
        network = fully_connected(network, 16, activation='relu')
        network = dropout(network, 0.5)

        network = fully_connected(network, 2, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy')

        model = tflearn.DNN(network)
        model.load("./model/thaitext-classifier-combined_inhousedata-UTF8-1.tfl")

        input_sentencedata = self.preprocess_server_2(sentencedata)[0]
        #input_uniquewords = self.get_uniquewords(input_sentencedata)
        sentences = []
        #f = open(file_path, 'r')
        for word in input_sentencedata:
            sentences.append(word)
        vector_one = []
        for word in unique_words:
            if word in sentences:
                vector_one.append(1)
            else:
                vector_one.append(0)
        vector_one = np.array(vector_one, dtype=np.float32)

        label = model.predict_label([vector_one])
        pred = model.predict([vector_one])

        return pred

api.add_resource(Thai_segment, '/Thai_segment/<sentence>')

if __name__ == "__main__":
    app.run(host='0.0.0.0')