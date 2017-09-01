"""
This is a basic pairs trading algorithm that uses the optimize 
WARNING: THIS IS A LEARNING EXAMPLE ONLY. DO NOT TRY TO TRADE SOMETHING THIS SIMPLE.
https://www.quantopian.com/workshops
https://www.quantopian.com/lectures

For any questions, email max@quantopian.com
"""
import numpy as np
import pandas as pd
import quantopian.experimental.optimize as opt
import quantopian.algorithm as algo
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
    """
    Called once at the start of the algorithm.
    """   
    schedule_function(record_vars, date_rules.every_day(), time_rules.market_close(hours=1))
    
    context.market = symbol('FSLR')
    
    context.holding = False
    

def record_vars(context, data):
    
    # Get pricing history
    market = context.market
    isholding = context.holding
    current_price = data.current(market, 'price')
    ema_window = 20
    t_window = 5
    high = data.history(market, "high", bar_count=ema_window+t_window, frequency='1d')
    low = data.history(market, "low", bar_count=ema_window+t_window, frequency='1d')
    
    high_ema = talib.EMA(high, timeperiod=ema_window)
    low_ema = talib.EMA(low, timeperiod=ema_window)
    hlmean = (np.array(high_ema) + np.array(low_ema))/2

    trend = linreg(range(t_window), hlmean[-t_window:])
    
    if trend>0 and not isholding:
        order(context.market, 10000)
        context.holding = True
    elif trend<=0 and isholding:
        order_target_percent(context.market, 0)
        context.holding = False
    
    
    record(high_ema=high_ema[-1],low_ema=low_ema[-1], HLmean=hlmean[-1], 
            trend=trend, curr_price=current_price)
    
    
    
    