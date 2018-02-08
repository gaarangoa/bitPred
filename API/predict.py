
from keras.models import load_model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
import json
from pymongo import MongoClient
import pymongo
from train import model
import utilslib
import re
from train import model
from train import config
from sklearn.preprocessing import StandardScaler


class Predict():
    def __init__(self, model_name="./API/regression.hdf5"):
        self.par = model.Train()
        self.model = load_model(model_name)
        self.client = MongoClient("mongodb://localhost", 11914)
        self.symbol = 'BTC.X'
        self.timeframe = "history_15m"
        self.data_window = 15

    def pred(self, max_timestamp=0, data_window="history_15"):
        self.data_window = data_window
        # assuming query has the data in the right format
        print("Loading input file")
        data, signal = utilslib.get_data(max_timestamp=max_timestamp, client=self.client, symbol=self.symbol, timeframe=self.timeframe, window=self.data_window)

        docs = data['text']
        encoded_docs = [one_hot(docs, config.vocab_size)] #uses a hash function to represent words, if words are similar they will have collisions
        padded_docs = pad_sequences(encoded_docs, maxlen=config.max_length, padding='post')
        
        stocks = np.array(data['close'])
        volume = np.array(data['volume'])
        bearish = np.array(data['bearish'])
        sentiment = np.array(data['bullish'])
        
        pred = self.model.predict([padded_docs, stocks, sentiment, volume, bearish])

        return [pred, data, signal]


def test():
    p = Predict()
    time = 1517515200
    for _ in range(500):
        x = p.pred(max_timestamp=time, data_window=15)
        vis(data=x, p=p)
        time+=900


