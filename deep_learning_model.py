# this model is contains three main stages: 
# 1) An LSTM to characterize the sequence from posts
# 2) An LSTM to characterize the previous (series) price taking all parameters (volume, price, open, close, maximum, minimum)
# 3) A 1D CNN to characterize the temporal price variation taking 10 time points. 
# 
# The time interval consists of windows of 15 minutes. 
# Data is taken from gdax (bitcoin price) and stocktwitts for post/text data. 
# TODO: include twitter data
# 
# The data contains only information from users who gave a right prediction, bullish/bearish. Other data is not takint into account for this version.
# 
# Developed by Gustavo Arango

import json
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, Dropout, LSTM, Conv1D, MaxPooling1D, GlobalAveragePooling1D, concatenate, Input, Reshape
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.models import Model

from sklearn import preprocessing
import numpy as np

# load the data
data = json.load(open('data.json'))

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
vocab_size = 5000
encoded_docs = [one_hot(d, vocab_size) for d in docs] #uses a hash function to represent words, if words are similar they will have collisions
# encoded_docs = np.expand_dims(encoded_docs, axis=2)
# pad documents to a max length of 50 words
max_length = 50
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
embedding_size = 500 # output of the embedded vector

# Build the model
text_model_input = Input(shape = (max_length,), dtype="int32", name = 'text_model_input')
text_model = Embedding(input_dim = vocab_size, output_dim = embedding_size, input_length = max_length, name="text-embedding" )(text_model_input)
text_model_output = LSTM(128, name = 'text_lstm' )(text_model)
# text_model_output = Dense(100, activation="relu", name="pred-text" )(text_model)

# text_model_out = Dense(2, activation="relu", name="text-out" )(text_model_output)
# model = Model(text_model_input, text_model_out)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.fit([padded_docs], [categorical_labels], batch_size=128, epochs=1)


# ---------------------------- #
#     CNN prices topology      #
# ---------------------------- #
prices = [i['price'] for i in data]

# pad sequences to fixed length
max_length_price_series = 12
padded_prices = np.array(pad_sequences(prices, maxlen=max_length_price_series, padding='pre'))
padded_prices = np.expand_dims(padded_prices, axis=3)

price_vector_len = 1

# the model
price_model_input = Input( shape = (max_length_price_series, price_vector_len), name = "price_model_input" )
price_model = Conv1D( 128, 3, activation = 'softmax', input_shape = (max_length_price_series, price_vector_len) )(price_model_input) 
price_model = MaxPooling1D(3)(price_model)
price_model = Dropout(0.5)(price_model)
price_model = Conv1D(128, 2, activation='softmax')(price_model)
price_model = MaxPooling1D()(price_model)
price_model_output = Flatten()(price_model)
# price_model_output = Dense(100, activation="relu", name="pred-price")(price_model)

# price_model_out = Dense(2, activation="softmax", name="price-out")(price_model_output)
# model = Model(price_model_input, price_model_out)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.fit([padded_prices], [categorical_labels], batch_size=128, epochs=1)

# ---------------------------- #
#         Stocks Model         #
# ---------------------------- #

stocks = []
for i in data:
    stock = []
    for j in i['stocks']:
        stock.append([10*j['volume'], j['high'], j['low'], j['open'], j['close'], j['high']-j['low']])
    stocks.append(stock)

max_length_stock_series = 12
stock_embedding_size = 6
padded_stocks = np.array(pad_sequences(stocks, maxlen=max_length_stock_series, padding='pre'))

stock_model_input = Input(shape = (max_length_stock_series, stock_embedding_size), dtype="float32", name = 'stock_model_input')
stock_model_output = LSTM(128, name = 'stock_lstm', input_shape = (max_length_stock_series, stock_embedding_size) )(stock_model_input)

# **************************** #
#        MERGE MODELS          #
# **************************** #

merged_model = concatenate([text_model_output, price_model_output, stock_model_output], axis=1)
merged_model = Dense(240, activation="relu")(merged_model)
merged_model = Dropout(0.5)(merged_model)
merged_model = Dense(240, activation="relu")(merged_model)
merged_model = Dropout(0.5)(merged_model)
merged_model_output = Dense(2, activation = "softmax", name = 'merged_model_output')(merged_model)


model = Model(inputs = [text_model_input, price_model_input, stock_model_input], outputs = [merged_model_output])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

# Train the model

model.fit([padded_docs, padded_prices, padded_stocks], [categorical_labels], batch_size=128, epochs=100)

# model.fit([padded_docs, padded_prices], [encoded_labels, encoded_labels, encoded_labels], batch_size=128, epochs=1)

# score = model.evaluate(padded_docs, encoded_labels, batch_size=128)

# save model
model.save('model.hdf5')

try:
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
except:
    pass