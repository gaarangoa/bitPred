import json
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, Dropout, LSTM, Conv1D, MaxPooling1D, GlobalAveragePooling1D, concatenate, Input, Reshape, Bidirectional
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.models import Model
from keras.callbacks import ModelCheckpoint

from sklearn import preprocessing
import numpy as np
import sys

import re

from sklearn.preprocessing import StandardScaler

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

import config 
class Train():
    def __init__(self):
        self.info = ''
        self.dataset = 'data.json'
        self.vocab_size = config.vocab_size
        self.max_length = config.max_length
        self.embedding_size = config.embedding_size
        self.max_length_stock_series = config.max_length_stock_series
        self.stock_embedding_size = config.stock_embedding_size
        self.max_length_sentiment_series = config.max_length_sentiment_series
        self.sentiment_embedding_size = config.sentiment_embedding_size

    def run(self):
        # load the data
        data = json.load(open(self.dataset))

        # ---------------------------- #
        #  topology for the text LSTM  #
        # ---------------------------- #

        # dataset and class labels
        docs = data['text']
        raw_labels = [ i[0] for i in data['labels'] ]

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
        text_model_output = LSTM(512, name = "text-lstm-2", return_sequences=False)(text_model)
        # text_model_output = LSTM(512, name = 'text-lstm-3')(text_model)

        # ---------------------------- #
        #         Close  Model         #
        # ---------------------------- #

        stocks = np.array(data['close'])

        # padded_stocks = np.array(pad_sequences(stocks, maxlen=self.max_length_stock_series, padding='pre'))
        stock_model_input = Input(shape = (self.max_length_stock_series, self.stock_embedding_size), dtype="float32", name = 'stock_model_input')
        stock_model_output = LSTM(256, return_sequences=False, name = 'stock_lstm', input_shape = (self.max_length_stock_series, self.stock_embedding_size) )(stock_model_input)
        # stock_model_output = LSTM(256)(stock_model)


        # ---------------------------- #
        #         Volumen  Model       #
        # ---------------------------- #

        volume = np.array(data['volume'])

        # padded_volume = np.array(pad_sequences(volume, maxlen=self.max_length_stock_series, padding='pre'))
        volume_model_input = Input(shape = (self.max_length_stock_series, self.stock_embedding_size), dtype="float32", name = 'volume_model_input')
        volume_model_output = LSTM(256, return_sequences=False, name = 'volume_lstm', input_shape = (self.max_length_stock_series, self.stock_embedding_size) )(volume_model_input)

        # ---------------------------- #
        #         Bearish  Model       #
        # ---------------------------- #

        bearish = np.array(data['bearish'])

        # padded_bearish = np.array(pad_sequences(bearish, maxlen=self.max_length_stock_series, padding='pre'))
        bearish_model_input = Input(shape = (self.max_length_stock_series, self.stock_embedding_size), dtype="float32", name = 'bearish_model_input')
        bearish_model = LSTM(256, return_sequences=False, name = 'bearish_lstm', input_shape = (self.max_length_stock_series, self.stock_embedding_size) )(bearish_model_input)
        bearish_model_output = LSTM(256)(bearish_model)

        # ---------------------------- #
        #       Sentiment Model        #
        # ---------------------------- #

        sentiment = np.array(data['bullish'])

        # padded_sentiment = np.array(pad_sequences(sentiment, maxlen=self.max_length_sentiment_series, padding='pre'))
        sentiment_model_input = Input(shape = (self.max_length_sentiment_series, self.sentiment_embedding_size), dtype="float32", name = 'bullish_model_input')
        sentiment_model = LSTM(256, return_sequences=True, name = 'bullish_lstm', input_shape = (self.max_length_sentiment_series, self.sentiment_embedding_size) )(sentiment_model_input)
        sentiment_model_output = LSTM(256)(sentiment_model)

        # **************************** #
        #        MERGE MODELS          #
        # **************************** #

        merged_model = concatenate([text_model_output, stock_model_output, sentiment_model_output, volume_model_output, bearish_model_output], axis=1)
        merged_model = Dense(1500, activation="relu")(merged_model)
        merged_model = Dropout(0.5)(merged_model)
        merged_model = Dense(1000, activation="relu")(merged_model)
        merged_model = Dropout(0.5)(merged_model)
        merged_model = Dense(600, activation="relu")(merged_model)
        merged_model_output = Dense(3, activation = "softmax", name = 'merged_model_output')(merged_model)

        model = Model(inputs = [text_model_input, stock_model_input, sentiment_model_input, volume_model_input, bearish_model_input], outputs = [merged_model_output])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        print(model.summary())

        try:
            from keras.utils import plot_model
            plot_model(model, to_file='model.png')
        except:
            pass

        # Train the model
        checkpointer = ModelCheckpoint(filepath='./epoch/model.hdf5', verbose=1, save_weights_only=False)
        # tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=254, write_graph=True, write_grads=True, write_images=True, embeddings_freq=1)

        model.fit([padded_docs, stocks, sentiment, volume, bearish], [categorical_labels], batch_size=254, epochs=200, callbacks=[checkpointer])

        # save model
        model.save('model.hdf5')

