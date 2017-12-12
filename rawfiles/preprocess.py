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
from tflearn.layers.conv import conv_2d, max_pool_2d
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
        #print (evaluation_result)
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
        del evaluation_result
        #evaluation_result = []
        return response

    def sentiment_analysis(self, sentencedata):
        
        file_path = './corpus/BurgerKing_UTF8.csv'
        data, labels = load_csv(file_path, target_column=0, categorical_labels=True, n_classes=2)

        pdata =self.preprocess_server(data)
        unique_words = self.get_uniquewords(pdata)
        data = self.preprocess_vector(pdata, unique_words)

        

        neurons = len(data[0])

        # shuffle the dataset
        data, labels = shuffle(data, labels)

    def preprocess_server(self, data):
            rlist = []
            preprocessdata = []
            for i in range(len(data)):
                x = requests.get('http://174.138.26.245:5000/preprocess/'+data[i][0])
                resu = x.json()
                preprocessdata.append(resu['result'])
            
            for i in range(len(preprocessdata)):
                r = requests.get('http://174.138.26.245:5000/tokenize/'+preprocessdata[i])
                result = r.json()
                rlist.append(result['result'])
            return rlist

    def get_uniquewords(self, listdata):
        uniquewords = []
        for line in range(len(listdata)):
            words = listdata[line]
            inner_data = []
            for word in words:
                if word not in uniquewords:
                    #w = repr(word.encode('utf-8'))
                    uniquewords.append(word)
        return uniquewords

    def preprocess_vector(self, listdata, uniquewords):
            sentences = []
            vectors = []
            #f = open(file_path, 'r')
            for line in range(len(listdata)):
                words = listdata[line]
                inner_data = []
                for word in words:
                    inner_data.append(word)
                sentences.append(inner_data)
            
            for sentence in sentences:
                inner_vector = []
                for word in uniquewords:
                    if word in sentence:
                        inner_vector.append(1)
                    else:
                        inner_vector.append(0)
                vectors.append(inner_vector)
            return np.array(vectors, dtype=np.float32)

api.add_resource(Thai_segment, '/Thai_segment/<sentence>')

if __name__ == "__main__":
    app.run(host='0.0.0.0')