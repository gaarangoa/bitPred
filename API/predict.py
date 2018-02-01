
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
import config

class Predict():
    def __init__(self, model_name="./API/model.hdf5"):
        self.par = model.Train()
        self.model = load_model(model_name)
        self.client = MongoClient("mongodb://localhost", 11914)
        self.symbol = 'BTC.X'
        self.timeframe = "history_5m"
        self.data_window = 20

    def pred(self, max_timestamp):
        # assuming query has the data in the right format
        print("Loading input file")
        data = utilslib.get_data(max_timestamp=max_timestamp, client=self.client, symbol=self.symbol, timeframe=self.timeframe, window=self.data_window)

        docs = data['text']
        encoded_docs = [one_hot(d, config.vocab_size) for d in docs] #uses a hash function to represent words, if words are similar they will have collisions
        padded_docs = pad_sequences(encoded_docs, maxlen=config.max_length, padding='post')
        
        stocks = data['close']
        volume = data['volume']
        bearish = data['bearish']
        sentiment = data['bullish']

        pred = self.model.predict([padded_docs, stocks, sentiment, volume, bearish])

        return {
            "bear": str(pred[0][0]),
            "bull": str(pred[0][1]),
            "stay": str(pred[0][2])
        }

def test():
    data = json.load(open('data.json'))
    predictor = Predict(model_name="model.hdf5");
    predictor.pred(query = data[0])


