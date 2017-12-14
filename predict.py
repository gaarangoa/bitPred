
from keras.models import load_model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
import json

class Predict():
    def __init__(self, model_name="model.hdf5"):
        self.model = load_model(model_name)
        self.vocab_size = 5000
        self.text_max_len = 50
        self.max_length_price_series = 12
        self.max_length_stock_series = 12
        self.stock_embedding_size = 6

    def pred(self, query={}):
        print("Loading input file and feature extraction")
        
        encoded_doc = one_hot(query['text'], self.vocab_size)
        padded_doc = pad_sequences(encoded_doc, maxlen=self.max_length, padding='post')

        padded_price = np.array(pad_sequences(query['price'], maxlen=self.max_length_price_series, padding='pre'))
        padded_price = np.expand_dims(padded_price, axis=3)

        stock = []
        for j in query['stocks']:
            stock.append([10*j['volume'], j['high'], j['low'], j['open'], j['close'], j['high']-j['low']])

        padded_stock = np.array(pad_sequences([stock], maxlen=self.max_length_stock_series, padding='pre'))

        pred = self.model.predict([padded_doc, padded_price, padded_stock])

        print("query\t%Low\t%High")
        for ix,i in enumerate(pred):
            print("\t".join([
                    self.L[ix],
                    str(100*i[0]),
                    str(100*i[1])
                ]))

def main(args):
    data = json.load(open('data.json'))
    predictor = Predict();
    predictor.pred(query = data[0])


