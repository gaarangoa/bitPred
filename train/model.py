import json
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, Dropout, LSTM, Conv1D, MaxPooling1D, GlobalAveragePooling1D, concatenate, Input, Reshape, Bidirectional
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.models import Model

from sklearn import preprocessing
import numpy as np
import sys

# this model is contains three main stages: 
# 1) An LSTM to characterize the sequence from posts
# 2) An LSTM to characterize the previous (series) price taking all parameters (volume, price, open, close, maximum, minimum)
# 3) A 1D CNN to characterize the temporal price variation taking 15 time points. 
# 
# The time interval consists of windows of 15 minutes. 
# Data is taken from gdax (bitcoin price) and stocktwitts for post/text data. 
# TODO: include twitter data
# 
# The data contains only information from users who gave a right prediction, bullish/bearish. Other data is not takint into account for this version.
# 
# Developed by Gustavo Arango

class Train():
    def __init__(self):
        self.info = ''
        self.dataset = 'data.json'
        self.vocab_size = 8000
        self.max_length = 150
        self.embedding_size = 500
        self.max_length_stock_series = 15
        self.stock_embedding_size = 2
        self.max_length_price_series = 15
        self.price_embedding_size = 2

    def run(self):
        # load the data
        data = json.load(open(self.dataset))

        # ---------------------------- #
        #  topology for the text LSTM  #
        # ---------------------------- #

        # dataset and class labels
        docs = [i['text'] for i in data]
        raw_labels = [i['label'] for i in data]
        labels_encoder = preprocessing.LabelEncoder()
        labels_encoder.fit(raw_labels)
        encoded_labels = labels_encoder.transform(raw_labels)
        categorical_labels = np_utils.to_categorical( encoded_labels )

        # encode the documents
        encoded_docs = [one_hot(d, self.vocab_size) for d in docs] #uses a hash function to represent words, if words are similar they will have collisions
        padded_docs = pad_sequences(encoded_docs, maxlen=self.max_length, padding='post')

        # Build the model
        text_model_input = Input(shape = (self.max_length,), dtype="int32", name = 'text_model_input')
        text_model = Embedding(input_dim = self.vocab_size, mask_zero=True, output_dim = self.embedding_size, input_length = self.max_length, name="text-embedding" )(text_model_input)
        text_model = LSTM(128, name = "text-lstm-2", return_sequences=True)(text_model)
        text_model_output = LSTM(256, name = 'text-lstm-3')(text_model)

        # ---------------------------- #
        #         Stocks Model         #
        # ---------------------------- #

        stocks = []
        prices = []
        for i in data:
            stock = []
            price = []
            for j in i['stocks']:
                stock.append([10*j['volume'], j['high']-j['low']])
                price.append([j['high'], j['low']])
            stocks.append(stock)
            prices.append(price)

        padded_stocks = np.array(pad_sequences(stocks, maxlen=self.max_length_stock_series, padding='pre'))
        stock_model_input = Input(shape = (self.max_length_stock_series, self.stock_embedding_size), dtype="float32", name = 'stock_model_input')
        stock_model = LSTM(264, return_sequences=True, name = 'stock_lstm', input_shape = (self.max_length_stock_series, self.stock_embedding_size) )(stock_model_input)
        stock_model_output = LSTM(264)(stock_model)


        # ---------------------------- #
        #         Price Model          #
        # ---------------------------- #

        padded_prices = np.array(pad_sequences(prices, maxlen=self.max_length_price_series, padding='pre'))
        # padded_prices = np.array(prices)

        price_model_input = Input(shape = (self.max_length_price_series, self.price_embedding_size), dtype="float32", name = 'price_model_input')
        price_model = LSTM(264, return_sequences=True, name = 'price_lstm', input_shape = (self.max_length_price_series, self.price_embedding_size) )(price_model_input)
        # stock_model = LSTM(64)(stock_model, return_sequences=True)
        price_model_output = LSTM(264)(price_model)

        # **************************** #
        #        MERGE MODELS          #
        # **************************** #

        merged_model = concatenate([text_model_output, price_model_output, stock_model_output], axis=1)
        merged_model = Dense(1200, activation="relu")(merged_model)
        merged_model = Dropout(0.5)(merged_model)
        merged_model = Dense(640, activation="relu")(merged_model)
        merged_model = Dropout(0.5)(merged_model)
        merged_model = Dense(420, activation="relu")(merged_model)
        merged_model_output = Dense(2, activation = "softmax", name = 'merged_model_output')(merged_model)

        model = Model(inputs = [text_model_input, price_model_input, stock_model_input], outputs = [merged_model_output])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        print(model.summary())

        try:
            from keras.utils import plot_model
            plot_model(model, to_file='model.png')
        except:
            pass

        # Train the model
        model.fit([padded_docs, padded_prices, padded_stocks], [categorical_labels], batch_size=128, epochs=100)

        # save model
        model.save('model.hdf5')
