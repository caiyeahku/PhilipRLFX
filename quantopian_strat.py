import numpy as np
import pandas as pd
import talib

def linreg(X, Y):
    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x*x
        Syy = Syy + y*y
        Sxy = Sxy + x*y
    det = Sxx * N - Sx * Sx
    return (Sxy * N - Sy * Sx)/det
    
def initialize(context):
    schedule_function(record_vars, date_rules.every_day(), time_rules.market_close(minutes=60))
    context.market = symbol('FSLR')
    context.holding = False
    
def record_vars(context, data):
    market = context.market
    current_price = data.current(market, 'price')
    ema_window = 40
    trend_window = 5
    episode = 120
    high = data.history(market, "high", bar_count=ema_window+episode, frequency='1m')
    low = data.history(market, "low", bar_count=ema_window+episode, frequency='1m')
    
    high_ema = talib.EMA(high, timeperiod=ema_window)
    low_ema = talib.EMA(low, timeperiod=ema_window)
    hlmean = (np.array(high_ema) + np.array(low_ema))/2

    trends = []
    for i in range(ema_window-2, len(hlmean)-trend_window-1):
    	trends.append(linreg(range(trend_window), hlmean[i+1:i+trend_window+1]))
    trend = trends[-1]
    
    buy_thresh = np.percentile(trends, 95)
    sell_thresh = np.percentile(trends, 5)
    
    if trend>=buy_thresh and buy_thresh>=0 and not context.holding:
        order(context.market, 10000)
        context.holding = True
    elif trend<=sell_thresh and context.holding:
        order_target_percent(context.market, 0)
        context.holding = False
    
    record(HLmean=hlmean[-1], TREND=trend, PRICE=current_price,
          buy_thresh=buy_thresh, sell_thresh=sell_thresh)
    
    
    
    
