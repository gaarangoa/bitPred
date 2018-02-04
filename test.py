from API.predict import Predict
import matplotlib.pyplot as plt
import datetime
import pymongo
import numpy as np

import time

p = Predict()

max_time = int(time.time())
timestamp = max_time - 10*900

x = p.pred(max_timestamp=timestamp, data_window=15)

import API.utilslib as utilslib
client = pymongo.MongoClient("mongodb://localhost", 11914)
data = utilslib.get_data(max_timestamp=timestamp, client=client, window=15, symbol='BTC.X', timeframe='history_15m')






t1 = datetime.datetime.fromtimestamp(timestamp)
plt.ion()
# x = p.pred(max_timestamp=timestamp, data_window=15)
# timestamp+=900

real = []
predx = []
index = []
ix = 0
while not timestamp == max_time:
    index.append(ix)
    
    x2 = p.pred(max_timestamp=timestamp+900, data_window=15)
    # 
    # real.append(x[1]['close'][0][-1][0])
    # 
    py, = plt.plot(x[1]['close'][0].tolist()+x2[1]['close'][0].tolist(), label="real signal", color='blue')
    px, = plt.plot([15, 16, 17], [i[0][0] for i in x[0]], label="prediction", color='red')
    plt.legend(handles=[px, py])
    plt.draw()
    plt.pause(5)
    plt.cla()
    # predx.append(x[0][2][0][0])
    timestamp+=900
    ix+=1

while 1:
    max_time = int(time.time())
    x = p.pred(max_timestamp=timestamp, data_window=15)






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