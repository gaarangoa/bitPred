from API.predict import Predict
import matplotlib.pyplot as plt
import datetime
import pymongo
import numpy as np
import json
import time

p = Predict()

max_time = int(time.time())
timestamp = max_time - 100*900
t = datetime.datetime.fromtimestamp(timestamp)
x, xr, signal = p.pred(max_timestamp=timestamp, data_window=11)

timestamp+=900
# real = [xr['close'][0][0][0]]
pred = [x[0][0][0]]
sigl = []
# pred = []
# real = [xr['close'][0][-1]]
real = []
index = []
ix = 0
plt.figure(1)

while not timestamp == max_time:
    index.append(ix)
    x, xr, signal = p.pred(max_timestamp=timestamp, data_window=11)
    real.append(xr['close'][0][-1])
    sigl.append(signal[-1])
    # pred.append(x[0][0][0][0])
    plt.subplot(311)
    py, = plt.plot(index,real, label="real signal", color='blue')
    px, = plt.plot(index,pred, label="n+1", color='red')
    plt.legend(handles=[px])
    plt.draw()
    plt.pause(0.2)
    pred.append(x[0][0][0])
    plt.subplot(312)
    pz, = plt.plot(index, sigl, label="real signal", color='blue')
    plt.legend(handles = [py,pz])
    print(t)
    # real.append(xr['close'][0][-1])
    timestamp+=900
    ix+=1


while 1:
    max_time = int(time.time())
    x = p.pred(max_timestamp=timestamp, data_window=15)
    real.append(x[1]['close'][0][0][0])
    py, = plt.plot(index,real, label="real signal", color='blue')
    px, = plt.plot(index,pred, label="prediction", color='red')
    plt.legend(handles=[px, py])
    plt.draw()
    plt.pause(120)




def vis(step=0):
    max_time = int(time.time())
    timestamp = max_time - step*900
    t = datetime.datetime.fromtimestamp(timestamp)
    x, xr, signal = p.pred(max_timestamp=timestamp, data_window=11)
    # 
    label = 'bullish'
    if x[0][0][0] < xr['close'][0][-1][0]: 
        label = 'bearish'
    # 
    print(t)
    print('the price is', signal[-1], 'and it will tend to ', label )
    print(x)
    return (x,xr,signal,t)



import API.utilslib as utilslib
client = pymongo.MongoClient("mongodb://localhost", 11914)
data = utilslib.get_data(max_timestamp=timestamp, client=client, window=15, symbol='BTC.X', timeframe='history_15m')

t1 = datetime.datetime.fromtimestamp(timestamp)
plt.ion()
# x = p.pred(max_timestamp=timestamp, data_window=15)
# timestamp+=900


data = json.load(open('./train/data.json'))
stocks = np.array([ data['close'][0] ])
volume = np.array([ data['volume'][0] ])
bearish = np.array([ data['bearish'][0] ])
sentiment =np.array([ data['bullish'][0] ])
docs = [data['text'][-1]]

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

encoded_docs = [one_hot(d, 1000) for d in docs] #uses a hash function to represent words, if words are similar they will have collisions
padded_docs = pad_sequences(encoded_docs, maxlen=500, padding='post')

p.model.predict([padded_docs, stocks, sentiment, volume, bearish])





from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))



x = json.load(open('./train/data.json'))

plt.figure(1)


from pymongo import MongoClient
gdax = pymongo.MongoClient("mongodb://localhost", 11914).gdax["history_15m"]
timestamp = x['timestamp'][index]
stocks = [i['close'] for i in gdax.find({"time":{"$gte": timestamp}}).sort([("time", pymongo.ASCENDING)]).limit(20)]

s = scaler.fit_transform(np.array(stocks).reshape(-1,1))

plt.subplot(212)
plt.plot([i[0] for i in s])



plt.subplot(211)
index = -1
y = [i[0] for i in x['close'][index]] + [i[0] for i in x['regression'][index]][1:]
plt.plot( stocks )

plt.show()



def create_dataset(dataset, look_back, sentiment, sent=False):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        if i >= look_back:
            a = dataset[i-look_back:i+1, 0]
            a = a.tolist()
            if(sent==True):
                a.append(sentiment[i].tolist()[0])
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
    #print(len(dataY))
    return np.array(dataX), np.array(dataY)