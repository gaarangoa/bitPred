
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
class Predict():
    def __init__(self, model_name="./API/model.hdf5"):
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
        data = utilslib.get_data(max_timestamp=max_timestamp, client=self.client, symbol=self.symbol, timeframe=self.timeframe, window=self.data_window)

        docs = data['text']
        encoded_docs = [one_hot(d, config.vocab_size) for d in docs] #uses a hash function to represent words, if words are similar they will have collisions
        padded_docs = pad_sequences(encoded_docs, maxlen=config.max_length, padding='post')
        
        stocks = np.array(data['close'])
        volume = np.array(data['volume'])
        bearish = np.array(data['bearish'])
        sentiment = np.array(data['bullish'])
        
        pred = self.model.predict([padded_docs, stocks, sentiment, volume, bearish])

        return [{
            "bear": pred[0][0],
            "bull": pred[0][1],
            "stay": pred[0][2]
        }, data]


import matplotlib.pyplot as plt
import datetime
import pymongo
import numpy as np

def vis(data=[], p=[]):
    pred = data[0]
    data = data[1]
    first_timestamp = data['timestamp'][0][0]
    gdax = p.client.gdax[p.timeframe]
    stocks = [i for i in gdax.find({"time":{"$gte": first_timestamp}}).sort([("time", pymongo.ASCENDING)]).limit(25)]
    price = np.array([i['close'] for i in stocks])
    # 
    fmt = "%Y-%m-%d %H:%M:%S"
    t1 = datetime.datetime.fromtimestamp(float(stocks[18]['time']))
    t2 = datetime.datetime.fromtimestamp(float(stocks[22]['time']))
    #
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 
    ax.plot(price)
    ax.scatter(14, price[14])
    ax.scatter(15, price[15])
    tx = 'Bullish: '+str(pred['bull']) + '\nBearish: '+str(pred['bear']) + '\nStay: '+ str(pred['stay'])
    ax.text(18, price[15], tx)
    ax.text(14, price[11], "Now")
    ax.text(1, max(price), t1.strftime(fmt))
    ax.text(10, max(price), t2.strftime(fmt))
    plt.show()



def test():
    p = Predict()
    time = 1517515200
    for _ in range(500):
        x = p.pred(max_timestamp=time, data_window=15)
        vis(data=x, p=p)
        time+=900


