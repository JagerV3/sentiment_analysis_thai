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


from json import encoder
from decimal import Decimal


class uniqueword():

    def preprocess_server(data):
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

    def get_uniquewords(listdata):
        uniquewords = []
        for line in range(len(listdata)):
            words = listdata[line]
            inner_data = []
            for word in words:
                if word not in uniquewords:
                    #w = repr(word.encode('utf-8'))
                    uniquewords.append(word)
        return uniquewords

    def csv_writer(write_data):
        with open('./uniqueword/Combined_inhousedata_UTF8-2.csv', 'wb') as write_file:
            writer = csv.writer(write_file)
            for line in write_data:
                print(line)
                writer.writerow(line)

    file_path = './corpus/Combined_inhousedata_UTF8-2.csv'
    data, labels = load_csv(file_path, target_column=0, categorical_labels=True, n_classes=2)

    pdata =preprocess_server(data)
    unique_words = get_uniquewords(pdata)
    csv_writer(unique_words)
