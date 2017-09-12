import pandas as pd
import numpy as np
import time
import scipy.misc as misc
import os, errno
from sys import argv
from tqdm import trange

def GramianAngularField(all_ts, window_size, rolling_length, method='summation', scale=0):

    # 取得時間序列長度
    n = len(all_ts) 
    
    # 我們要避免微觀被過度放大, 所以真正的 window_size 是原本的 1.25倍
    real_window_size = window_size + 1 
    
    # 根據我們的滾動大小，將資料切成一組一組
    n_rolling_data = int(np.floor((n - real_window_size)/rolling_length))
    
    

    # Rescaling aggregated time series
    min_ts, max_ts = np.min(aggregated_ts), np.max(aggregated_ts)
    if scale == 0:
        rescaled_ts = (aggregated_ts - min_ts) / (max_ts - min_ts)
    if scale == -1:
        rescaled_ts = (2 * aggregated_ts - max_ts - min_ts) / (max_ts - min_ts)

    # Compute GAF
    sin_ts = np.sqrt(np.clip(1 - rescaled_ts**2, 0, 1))
    if method == 'summation':
        return np.outer(rescaled_ts, rescaled_ts) - np.outer(sin_ts, sin_ts)
    if method == 'difference':
        return np.outer(sin_ts, rescaled_ts) - np.outer(rescaled_ts, sin_ts)