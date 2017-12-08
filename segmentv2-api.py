#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from flask_api import FlaskAPI
from nltk import NaiveBayesClassifier as nbc
from pythainlp.tokenize import word_tokenize
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

from flask import Flask, jsonify, json, Response
from flask_restful import reqparse, Api, Resource
from json import encoder
from decimal import Decimal

app = Flask(__name__)
#app.config['JSON_AS_ASCII'] = False

api = Api(app)
parser = reqparse.RequestParser()
parser.add_argument('sentence')

class Thai_segment(Resource):
    def get(self, sentence):
        args = parser.parse_args()

        evaluation_result = self.segment_analysis(sentence)
        data = {}
        data['positive result'] = str(evaluation_result[0][1])
        data['negative result'] = str(evaluation_result[0][0])

        if (evaluation_result[0][1]>=evaluation_result[0][0]):
            data['Label'] = "Positive"
        else:
            data['Label'] = "Negative"

        data['Sentence'] = sentence
        json_result = json.dumps(data, ensure_ascii=False)
        response = Response(json_result, content_type="application/json; charset=utf-8")
        return response

    def segment_analysis(self, sentencedata):
        
        file_path = 'Cleaned-Masita corpus 2.csv'
        data, labels = load_csv(file_path, target_column=0, categorical_labels=True, n_classes=2)

        pdata =self.preprocess_server(data)
        unique_words = self.get_uniquewords(pdata)
        data = self.preprocess(pdata, unique_words)

        neurons = len(data[0])

        # shuffle the dataset
        from tflearn.data_utils import shuffle
        data, labels = shuffle(data, labels)

        network = input_data(shape=[None, neurons])
        network = fully_connected(network, 8, activation='relu')
        network = fully_connected(network, 8*2, activation='relu')
        network = fully_connected(network, 8, activation='relu')
        network = dropout(network, 0.5)

        network = fully_connected(network, 2, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy')

        model = tflearn.DNN(network)
        #model.fit(data, labels, n_epoch=40, shuffle=True, validation_set=None , show_metric=True, batch_size=None, snapshot_epoch=True, run_id='task-classifier')
        #model.save("./model/thaitext-classifier-mashita.tfl")
        #print("Network trained and saved as thaitext-classifier-mashita.tfl")

        model.load("./model/thaitext-classifier-mashita.tfl")
        #file_path3 = 'Cleaned-Masita-traindataset-2.csv'
        input_sentencedata = self.preprocess_server(sentencedata)

        vector_one = []
        for word in unique_words:
            if word in input_sentencedata:
                vector_one.append(1)
            else:
                vector_one.append(0)

        vector_one = np.array(vector_one, dtype=np.float32)

        label = model.predict_label([vector_one])
        #print (label)

        pred = model.predict([vector_one])
        #print(pred)
        return pred

        #testdata, testlabels = load_csv(file_path3, target_column=0, categorical_labels=True, n_classes=2)
        #resultdata = self.preprocess_server(testdata)
        #resultdata = self.preprocess(resultdata, unique_words)

        #model.fit(data, labels, n_epoch=40, shuffle=True, validation_set=(resultdata, testlabels) , show_metric=True, batch_size=None, snapshot_epoch=True, run_id='task-classifier')
        #model.save("thaitext-classifier-mashita.tfl")
        #print("Network trained and saved as Pthaitext-classifier-mashita.tfl")


        #result = model.evaluate(resultdata, testlabels)

        #return label

        #predict = model.predict(resultdata)
        #print("Evaluation result: %s" %result)

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

    def preprocess(self, listdata, uniquewords):
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