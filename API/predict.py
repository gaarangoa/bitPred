
from keras.models import load_model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
import json
from train import model
class Predict():
    def __init__(self, model_name="model.hdf5"):
        self.par = model.Train()
        self.model = load_model(model_name)
        self.vocab_size = self.par.vocab_size
        self.text_max_len = self.par.text_max_len
        self.max_length_price_series = self.par.max_length_price_series
        self.max_length_stock_series = self.par.max_length_stock_series
        self.stock_embedding_size = self.par.stock_embedding_size

    def pred(self, query={}):
        print("Loading input file and feature extraction")

        stock = []
        price = []
        for j in query['stocks']:
            stock.append( [10*j['volume'], j['high']-j['low']] )
            price.append( [j['high'], j['low']] )

        padded_stock = np.array(pad_sequences([stock], maxlen=self.max_length_stock_series, padding='pre'))
        padded_price = np.array(pad_sequences([price], maxlen=self.max_length_price_series, padding='pre'))
        encoded_doc = one_hot(query['text'], self.vocab_size)
        padded_doc = pad_sequences([encoded_doc], maxlen=self.text_max_len, padding='post')

        pred = self.model.predict([padded_doc, padded_price, padded_stock])

        return {
            "bear": str(pred[0][0]),
            "bull": str(pred[0][1])
        }

def test():
    data = json.load(open('data.json'))
    predictor = Predict(model_name="model.hdf5");
    predictor.pred(query = data[0])


