import numpy as np
import talib


def initialize(context):
    schedule_function(record_vars, date_rules.every_day(), time_rules.market_close(minutes=60))
    context.market = symbol('FSLR')
    context.myHist = []
    context.all_buy_price = []
    context.all_mfe = []
    context.all_mae = []
    context.mfe = 0.0
    context.mae = 0.0
    context.virtual_trading = True

def virtual_trading(context, data):
    market = context.market
    
    all_hist = np.array(data.history(market, "price", bar_count=3000, frequency='1d'))
    all_ma5 = talib.SMA(all_hist, timeperiod=5)
    all_ma10 = talib.SMA(all_hist, timeperiod=10)
    
    for p in range(len(all_hist)):
        
        current_price = all_hist[p]
        ma5 = all_ma5[p]
        ma10 = all_ma10[p]
        
        if ma5 is np.nan or ma10 is np.nan or current_price is np.nan:
            continue
            
        if ma5 > ma10:
        
            for i in range(len(context.myHist)):
                context.myHist[i].append(current_price)
            context.myHist.append([current_price])
        
            context.all_buy_price.append(current_price)
        
            sell_list = []
            for i in range(len(context.all_buy_price)):
                if (context.mfe > 0 and current_price >= context.all_buy_price[i]+context.mfe) or \
                    (context.mae < 0 and current_price <= context.all_buy_price[i]+context.mae):
                    sell_list.append(i)
            context.all_buy_price = [ context.all_buy_price[x] for x in range(len(context.all_buy_price)) if x not in sell_list ]
        
        elif ma5 < ma10:
        
            update_mfae(context)
        
            context.all_buy_price = []
        
        else:
        
            for i in range(len(context.myHist)):
                context.myHist[i].append(current_price)
        
            sell_list = []
            for i in range(len(context.all_buy_price)):
                if (context.mfe > 0 and current_price >= context.all_buy_price[i]+context.mfe) or \
                    (context.mae < 0 and current_price <= context.all_buy_price[i]+context.mae):
                    sell_list.append(i)
            context.all_buy_price = [ context.all_buy_price[x] for x in range(len(context.all_buy_price)) if x not in sell_list ]
            
    

def update_mfae(context):
    
    if len(context.myHist) == 0:
        return
    
    for i in range(len(context.myHist)):
        this_mfe = 0.0
        this_mae = 0.0
        
        buy_price = context.myHist[i][0]
        
        k = 0
        
        for m in range(1,len(context.myHist[i])):
            e = context.myHist[i][m] - buy_price    
            if e > this_mfe:
                this_mfe = e
                k = m
        for n in range(k, 0, -1):
            e = context.myHist[i][n] - buy_price    
            if e < this_mae:
                this_mae = e
        
        context.all_mfe.append(this_mfe)
        context.all_mae.append(this_mae)
        
    context.mfe = np.percentile(context.all_mfe, 50)
    context.mae = np.percentile(context.all_mae, 50)
    
    context.myHist = []
        
def check_stops(current_price, context):
    sell_list = []
    for i in range(len(context.all_buy_price)):
        if (context.mfe > 0 and current_price >= context.all_buy_price[i]+context.mfe) or \
            (context.mae < 0 and current_price <= context.all_buy_price[i]+context.mae):
            order_target(context.market, -100)
            sell_list.append(i)
    context.all_buy_price = [ context.all_buy_price[x] for x in range(len(context.all_buy_price)) if x not in sell_list ]
            

def record_vars(context, data):
    
    if context.virtual_trading:
        virtual_trading(context, data)
        context.virtual_trading = False
    
    market = context.market
    current_price = data.current(market, 'price')
    
    hist = np.array(data.history(market, "price", bar_count=10, frequency='1d'))
    
    ma5 = talib.SMA(hist, timeperiod=5)[-1]
    ma10 = talib.SMA(hist, timeperiod=10)[-1]
    
    if ma5 > ma10:
        
        order(context.market, 100)
        
        for i in range(len(context.myHist)):
            context.myHist[i].append(current_price)
        context.myHist.append([current_price])
        
        context.all_buy_price.append(current_price)
        
        check_stops(current_price, context)
        
    elif ma5 < ma10:
        
        order_target_percent(context.market, 0)
        
        update_mfae(context)
        
        context.all_buy_price = []
        
    else:
        
        for i in range(len(context.myHist)):
            context.myHist[i].append(current_price)
        
        check_stops(current_price, context)
    
    record(MFE=context.mfe, MAE=context.mae, PRICE=current_price,
              MA5=ma5, MA10=ma10)
    
    
    
    
