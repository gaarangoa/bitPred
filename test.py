from API.predict import Predict
from API.predict import vis
import matplotlib.pyplot as plt
import datetime
import pymongo
import numpy as np

import time

p = Predict()

max_time = int(time.time())
timestamp = max_time - 50*900

t1 = datetime.datetime.fromtimestamp(timestamp)
plt.ion()

x = p.pred(max_timestamp=timestamp, data_window=15)
timestamp+=900
real = []
predx = [x[0][0][0]]
index = []
ix = 0



while not timestamp == max_time:
    index.append(ix)
    x = p.pred(max_timestamp=timestamp, data_window=15)
    # 
    real.append(x[1]['close'][0][-1][0])
    # 
    px, = plt.plot(index, predx, label="prediction", color='red')
    py, = plt.plot(index, real, label="real signal", color='blue')
    plt.legend(handles=[px, py])
    plt.draw()
    plt.pause(1)
    predx.append(x[0][0][0])
    timestamp+=900
    ix+=1

while 1:
    max_time = int(time.time())
    x = p.pred(max_timestamp=timestamp, data_window=15)







