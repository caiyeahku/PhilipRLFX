#   寫一個函數來得到 MTF 
#   輸入 : 
#       - all_ts            < 1 x N double > 時間序列資料
#       - window_size       < int > 資料窗格大小（類似技術指標的參數）
#       - rolling_length    < int > 每次滾動移動大小（可以抓 window_size 的 1/8 ~ 1/10 做觀察）
#       - method            [optional] < int > 使用 summation(GASF) 或者 difference(GADF) (預設: summation)
#       - scale             [optional] < int > 正規化為 [0,1] 或 [-1,1] (預設: [0,1])
#---------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import scipy.misc as misc
import os, errno
from sys import argv
from tqdm import trange

# 初始化Placeholder陣列函式
def placeholderMatrix(n,m):
    matrix = []
    for i in range(n):
        matrix.append(np.zeros((q,q), float))
    return np.array(matrix)

# 繪製並輸出Miscellaneous圖
def outputMiscellaneous(features):
    
    # 確認misc資料夾存在
    try:
        os.makedirs("gaf_misc")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    # 開始繪製Miscellaneous圖片
    for index in trange(features.shape[0], desc="Drawing..."):
        img = misc.toimage(features[index])
        img.save('gaf_misc/%04d.png'%(index))

# 主函式
def GramianAngularField(all_ts, window_size, rolling_length, method='summation', scale='[0,1]'):

    # 取得時間序列長度
    n = len(all_ts) 
    
    # 我們要避免微觀被過度放大, 所以移動的 window_size 是原本的 2 倍
    moving_window_size = window_size * 2
    
    # 根據我們的滾動大小，將資料切成一組一組
    n_rolling_data = int(np.floor((n - moving_window_size)/rolling_length))
    
    # 最終的 GAF
    gramian_field = []
    
    # 紀錄價格，用來畫圖
    Prices = []

    # 開始從第一筆資料前進
    for i_rolling_data in trange(n_rolling_data, desc="Extracting..."):

        # 起始位置
        start_flag = i_rolling_data*rolling_length
        
        # 整個窗格的資料先從輸入時間序列中取出來
        full_window_data =  list(all_ts[start_flag : start_flag+moving_window_size])

        # 紀錄窗格的資料，用來畫圖
        Prices.append(full_window_data[-window_size:])
        
        # 因為等等要做cos/sin運算, 所以先正規化時間序列
        rescaled_ts = np.zeros((moving_window_size, moving_window_size), float)
        min_ts, max_ts = np.min(full_window_data), np.max(full_window_data)
        if scale == '[0,1]':
            rescaled_ts = (full_window_data - min_ts) / (max_ts - min_ts)
        if scale == '[-1,1]':
            rescaled_ts = (2 * full_window_data - max_ts - min_ts) / (max_ts - min_ts)

        # 留下原始 window_size 長度的資料
        rescaled_ts = rescaled_ts[-window_size:]
        
        # 計算 Gramian Angular Matrix
        this_gam = np.zeros((window_size, window_size), float)
        sin_ts = np.sqrt(np.clip(1 - rescaled_ts**2, 0, 1))
        if method == 'summation':
            # cos(x1+x2) = cos(x1)cos(x2) - sin(x1)sin(x2)
            this_gam = np.outer(rescaled_ts, rescaled_ts) - np.outer(sin_ts, sin_ts)
        if method == 'difference':
            # sin(x1-x2) = sin(x1)cos(x2) - cos(x1)sin(x2)
            this_gam = np.outer(sin_ts, rescaled_ts) - np.outer(rescaled_ts, sin_ts)
            
        gramian_field.append(this_gam)
    
    return np.array(gramian_field), np.array(Prices)
    
    
def main():
    # 讀取檔案
    data = pd.read_csv(argv[1])
    data.dropna(inplace=True)
    
    # 製作 Gramian Angular Field
    Window_Size = 100
    Rolling_Length = 10
    Features, Prices = GramianAngularField(all_ts=data['CLOSE'], 
                                window_size=Window_Size, 
                                rolling_length=Rolling_Length)

    # 輸出 Gramian Angular Field
    Features.dump('gaf_Features.pkl')
    Prices.dump('gaf_Prices.pkl')
    
    # 輸出所有Miscellaneous圖片
    outputMiscellaneous(Features)
        
        
if __name__ == "__main__":
    main()
        