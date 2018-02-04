from bisect import bisect_left
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re
import numpy as np
import pymongo

look_back = 5;

def take_closest(my_list, my_number):
    """
    Assumes my_list is sorted. Returns closest value to my_number.
    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(my_list, my_number)
    if pos == 0:
        return my_list[0]
    if pos == len(my_list):
        return my_list[-1]
    before = my_list[pos - 1]
    after = my_list[pos]
    if after - my_number < my_number - before:
       return after
    else:
       return before

def get_sentiment_stw(tweets=[]):
    sentiment = {'Bearish':0, 'Bullish':0}
    for tweet in tweets:
        try:
            sentiment[tweet['entities']['sentiment']['basic']]+=1
        except Exception as e:
            pass
        try:
            sentiment[tweet['sentiment']['name']]+=1
        except Exception as e:
            pass
    return sentiment

def window_lookup(data=[]):
    x = []
    for ix in range(len(data)):
        if ix<=look_back: continue
        x.append( [i[0] for i in data[ix-look_back:ix] ])
    return x

def format_input(data):
    data = sorted(data, key=lambda k: k['time'])
    table = []
    for i in data:
        if i['close'] > i['open']:
            bear = 1
            bull = 0
        if i['close'] < i['open']:
            bull = 1
            bear = 0
        else:
            bull = 0
            bear = 0
        # 
        table.append([
            i['time'],
            i['volume'],
            i['open'],
            i['close'],
            i['high'],
            i['low'],
            i['sentiment']['Bearish']+bear,
            i['sentiment']['Bullish']+bull,
            i['text']
        ])

    # scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Normalize data
    timestamp = [i[0] for i in table]
    volume = window_lookup( scaler.fit_transform(np.array([i[1] for i in table]).reshape(-1, 1)) )
    sopen = window_lookup( scaler.fit_transform(np.array([i[2] for i in table]).reshape(-1, 1)) )
    sclose = window_lookup( scaler.fit_transform(np.array([i[3] for i in table]).reshape(-1, 1)) )
    high = window_lookup( scaler.fit_transform(np.array([i[4] for i in table]).reshape(-1, 1)) )
    low = window_lookup( scaler.fit_transform(np.array([i[5] for i in table]).reshape(-1, 1)) )
    bearish = window_lookup( scaler.fit_transform(np.array([i[6] for i in table]).reshape(-1, 1)) )
    bullish = window_lookup( scaler.fit_transform(np.array([i[7] for i in table]).reshape(-1, 1)) )
    comments = [" ".join([i[8] for i in table])]
    


    return {
        "timestamp":[timestamp],
        "text": comments,
        "open": [np.array(sopen)],
        "close": [np.array(sclose)],
        "high": [np.array(high)],
        "low": [np.array(low)],
        "volume": [np.array(volume)],
        "bearish": [np.array(bearish)],
        "bullish": [np.array(bullish)]
    }


def get_data(max_timestamp=0, client=[], symbol='BTC.X', timeframe='history_15m', window=15):
    stw = client.stocktwits[symbol]
    gdax = client.gdax[timeframe]

    # find the data between the range. To do so, first get the max-timestamp and go back in time up to the max length series (15 default). First for gdax, so, we can get the more accurate data.

    stocks = [i for i in gdax.find({"time":{"$lte": max_timestamp}}).sort([("time", pymongo.DESCENDING)]).limit(window+6)]
    # print stocks
    stocks.reverse()
    min_t = stocks[0]['time']
    max_t = stocks[-1]['time']

    texts = [i for i in stw.find({ 
        "$and":
            [{"timestamp": {"$lte": max_t}},
            {"timestamp": {"$gte": min_t}}]
    }).sort([("timestamp", pymongo.DESCENDING)])]
    texts.reverse()

    stocks_ts = [i['time'] for i in stocks]
    texts_ts = [i['timestamp'] for i in texts]
    
    stocks_ob = {i['time']:i for i in stocks}
    texts_ob = {i['timestamp']:i for i in texts}

    for tix in texts_ts:
        try:
            stocks_ob[take_closest( stocks_ts, tix)]['stweets'].append(texts_ob[tix])
        except:
            stocks_ob[take_closest( stocks_ts, tix)].update({'stweets': [texts_ob[tix]]})
    
    # get sentiment
    for i in stocks_ob:
        try:
            sentiment = get_sentiment_stw(stocks_ob[i]['stweets'])
        except:
            sentiment = {"Bearish": 0, "Bullish":0}
        stocks_ob[i].update({"sentiment": sentiment})
        # 
        try:
            text = [tweet['body'] for tweet in stocks_ob[i]['stweets']]
            text = re.sub(r'http\S+', '', " ".join(text), flags=re.MULTILINE).replace('\n','')
            stocks_ob[i].update({'text': text})
        except Exception as e:
            stocks_ob[i].update({'text': ""})

    data = format_input(stocks_ob.values())

    return data