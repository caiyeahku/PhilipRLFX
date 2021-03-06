#   寫一個函數來得到 MTF 
#   輸入 : 
#       - all_ts            < 1 x N double > 時間序列資料
#       - window_size       < int > 資料窗格大小（類似技術指標的參數）
#       - rolling_length    < int > 每次滾動移動大小（可以抓 window_size 的 1/8 ~ 1/10 做觀察）
#       - quantile_size     < int > 分位數，就是可以快速地做 K 分位數（當 quantile_size = 4 就會有四分位數）
#       - label_size        < int > 看多久之後的趨勢線斜率
#---------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import scipy.misc as misc
import os, errno
from sys import argv
from tqdm import trange

# 初始化Placeholder陣列函式
def placeholderMatrix(n,m,q):
    matrix = []
    for i in range(n):
        tmp = []
        for j in range(m):
            tmp.append(np.zeros((q,q), float))
        matrix.append(tmp)
    return np.array(matrix)

# 用基本線性迴歸找到斜率, 求得趨勢
def findTrend(src,slope_thresh=None,residual_thresh=None):
    
    # 取得 Y 的數量
    n = len(src)
    
    # 令 X 為 0~n 的參數矩陣
    x , y = np.array([i for i in range(n)]) , np.array(src)
    x = np.vstack([x, np.ones(n)]).T
    
    # 執行線性回歸
    LinReg = np.linalg.lstsq(x, y)
    
    # 取得斜率
    slope = LinReg[0][0]
    
    # 取得所有點到回歸線的距離累加
    residual = 9999
    try:
        residual = LinReg[1][0]
    except:
        pass
    
    # 如果沒有給予「閥值區間」, 則回傳斜率和距離總和
    if slope_thresh==None or residual_thresh==None:
        return slope, residual
    # 如果距離總和夠小, 代表回歸線的準確度越高（變異數越小）
    elif residual < residual_thresh:
        # 若斜率夠大, 代表趨勢向上
        if slope >= slope_thresh[0] and slope > 0.0:
            return 1
        # 若斜率夠小, 代表趨勢向下
        elif slope <= slope_thresh[1] and slope < 0.0:
            return -1
        else:
            return 0
    else:
        return 0

# 繪製並輸出Miscellaneous圖
def outputMiscellaneous(features):

    # 先整合所有圖片
    N = features.shape[0]
    Q = int(np.sqrt(features.shape[1]))
    W = features.shape[2]
    new_features = np.zeros( (N, W*Q , W*Q), float )
    for n in trange(N, desc="Combining..."):
        for i in range(Q):
            for j in range(Q):
                for k in range(W):
                    for l in range(W):
                        new_features[n, i*W+k, j*W+l] = features[n, i*Q+j, k, l]
    
    # 輸出圖片的pickle檔(.pkl)
    new_features.dump('mtf_Features4plot.pkl')
    
    # 確認misc資料夾存在
    try:
        os.makedirs("mtf_misc")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    # 開始繪製Miscellaneous圖片
    for index in trange(new_features.shape[0], desc="Drawing..."):
        img = misc.toimage(new_features[index])
        img.save('mtf_misc/%04d.png'%(index))                   
    
# 主函式
def MarkovTransitionField(all_ts, window_size, rolling_length, quantile_size, label_size):

    # 取得時間序列長度
    n = len(all_ts) 
    
    # 由於我們要比較前後來算有沒有狀態轉換, 所以真正的 window_size 會是原本的 +1
    real_window_size = window_size + 1 
    
    # 根據我們的滾動大小，將資料切成一組一組
    n_rolling_data = int(np.floor((n - real_window_size)/rolling_length)) 
    
    # 真正的 feature 會有 Q x Q　個
    feature_size = quantile_size * quantile_size 
    
    # 最終的 MTF
    markov_field = placeholderMatrix(n_rolling_data, feature_size, window_size) 
    
    # 宣告一個紀錄後n天趨勢的陣列（作為Label）
    labels = []
    
    # 紀錄價格，用來畫圖
    Prices = []
    
    # 開始從第一筆資料前進
    for i_rolling_data in trange(n_rolling_data, desc="Extracting..."):
    
        # 先宣告一個「矩陣的矩陣」
        this_markov_field =  placeholderMatrix(window_size, window_size, quantile_size)
        
        # 起始位置
        start_flag = i_rolling_data*rolling_length
        
        # 整個窗格的資料先從輸入時間序列中取出來
        full_window_data =  list(all_ts[start_flag : start_flag+real_window_size])
        
        # 紀錄窗格的資料，用來畫圖
        Prices.append(full_window_data)
        
        # 從資料斜率的分佈來取得漲跌閥值 (藉由斜率分佈的一個標準差，來確認漲跌)
        history_slope_data = []
        history_residual_data = []
        for d in range(real_window_size-label_size):
            this_slope, this_residual = findTrend(full_window_data[d:d+label_size])
            history_slope_data.append(this_slope)
            history_residual_data.append(this_residual)
        slope_up_thresh = np.percentile(history_slope_data, 63)
        slope_down_thresh = np.percentile(history_slope_data, 37)
        Slope_Thresh = [slope_up_thresh, slope_down_thresh]
        Residual_Thresh = np.percentile(history_residual_data, 50)
        
        # 製作Label
        label_source = list(all_ts[start_flag+real_window_size : start_flag+real_window_size+label_size])
        new_label = findTrend(label_source, slope_thresh=Slope_Thresh, residual_thresh=Residual_Thresh)
        labels.append(new_label)
        
        # 從第 i 筆資料開始
        for i_window_size in range(window_size):
            # 到第 j 筆資料，我們要算的是投影片裡面的 W_i,j
            for j_window_size in range(window_size):
            
                # 因為如果我們要算 5 分位數，至少要有 5 筆資料，如果小於我們就放 0
                if np.abs(i_window_size - j_window_size) > quantile_size - 1:
                    
                    # 如果 j > i 就是時間序列要反過來
                    this_window_data = []
                    if i_window_size > j_window_size:
                        # 正常情況(窗格內我們要使用的資料，在本次迴圈中大小就取出來)
                        this_window_data = full_window_data[j_window_size:i_window_size+1]
                    else:
                        # 反過來情況
                        flips = full_window_data[i_window_size:j_window_size+1]
                        this_window_data =  flips[::-1]
                    
                    # 取得本次要算馬可夫矩陣的資料長度大小
                    n_this_window_data = len(this_window_data)
                    
                    # 根據本次要算矩陣的資料，取得他的 K 分位數
                    quantiles = [(100/quantile_size)*i for i in range(1, quantile_size)]
                    this_quantile = []
                    for q in quantiles:
                        this_quantile.append(np.percentile( this_window_data, q, interpolation='midpoint'))
                    
                    # 加入 -inf 與 inf 方便我們判斷資料是介在哪個分位數之間
                    this_quantile = [ -np.inf ] + this_quantile + [ np.inf ]
                    
                    # 取得分位數總長（為了跑迴圈用）
                    n_quantile = len( this_quantile )
                    
                    # 先宣告一個矩陣待會要放馬可夫矩陣
                    this_markov_matrix = np.zeros((quantile_size, quantile_size), float);
                    
                    # 從第一筆資料開始算是介在哪個狀態（哪兩個 K 分位數之間）
                    for i_this_window_data in range(n_this_window_data-1):
                        
                        # 從兩個分位數開始跑迴圈
                        for i_quantile in range(1, n_quantile):
                            for j_quantile in range(1, n_quantile):
                                
                                # 如果資料介於 i 與 j 之間，矩陣在 i, j 就要 +1
                                if this_window_data[i_this_window_data] < this_quantile[i_quantile] and \
                                    this_window_data[i_this_window_data] >= this_quantile[i_quantile-1] and \
                                    this_window_data[i_this_window_data+1] < this_quantile[j_quantile] and \
                                    this_window_data[i_this_window_data+1] >= this_quantile[j_quantile-1]:
                                        this_markov_matrix[ i_quantile-1 , j_quantile-1 ] += 1
                                    
                    
                    # 由於剛剛算的是個數，最後每一行都要除以行總數，來得到轉移機率
                    this_markov_matrix_count = [ sum(x) for x in this_markov_matrix ]
                    n_this_markov_matrix_count = len(this_markov_matrix_count)
                    for i_this_markov_matrix_count in range(n_this_markov_matrix_count):
                        # 如果那個狀態轉換有發生至少 1 次
                        if this_markov_matrix_count[i_this_markov_matrix_count] > 0:
                            this_markov_matrix[i_this_markov_matrix_count,:] /= this_markov_matrix_count[i_this_markov_matrix_count]
                        else:
                            # 如果狀態轉換根本沒發生，就不要除，否則會有除零誤
                            this_markov_matrix[i_this_markov_matrix_count,:] = 0
                        
                    # 最後把矩陣放到矩陣的矩陣裡面的  W_i,j 位置
                    this_markov_field[i_window_size,j_window_size] = this_markov_matrix
                
        # 當矩陣的矩陣都弄完了，我們就要把矩陣的矩陣切成各別 N 個矩陣
        feature_count = 0
        
        # 切法是依照狀態轉換, 例如 1->1, 1->2 ... 2->1 , ... 所以兩個 for loop
        for i_quantile in range(quantile_size):
            for j_quantile in range(quantile_size):
            
                # 先建立一個要蒐集所有被拆開出來的相同元素要放的矩陣
                seperated_markov_matrix = np.zeros( (window_size, window_size), float )
                
                # 從本次的「矩陣的矩陣」中依序 1...n 和 1...n 去取出來，放到前面宣告的矩陣
                for i_window_size in range(window_size):
                    for j_window_size in range(window_size):
                        
                        # 先從矩陣的矩陣取出特定的 W_i,j
                        this_markov_matrix = this_markov_field[i_window_size, j_window_size];
                        
                        # 由於剛剛有提到，當 i j 太近的時候，資料會不夠沒有 K 分位數，所以矩陣會有空的（等於沒有）
                        if sum(sum(this_markov_matrix)) is not 0:
                            # 如果有矩陣，就把對應的狀態轉換機率放到拆分後的矩陣中
                            seperated_markov_matrix[i_window_size, j_window_size] = this_markov_matrix[i_quantile, j_quantile];
                        else:
                            # 如果 i j 太近沒矩陣，就放 0 
                            seperated_markov_matrix[ i_window_size, j_window_size ] = 0.0
                
                # 再把拆分出來的矩陣，放到整個滾動資料的對應位置
                markov_field[i_rolling_data,feature_count] = seperated_markov_matrix
                feature_count += 1
            
        
    return np.array(markov_field), np.array(labels), np.array(Prices)

        
def main():
    # 讀取檔案
    data = pd.read_csv(argv[1])
    data.dropna(inplace=True)
    
    # 取得「馬可夫轉移場矩陣（Features）」和「標記資料（Labels）」
    Window_Size = 20
    Rolling_Length = 2
    Quantile_Size = 4
    Label_Size = 4
    Features , Labels , Prices = MarkovTransitionField(all_ts=data['CLOSE'], 
                                                        window_size=Window_Size, 
                                                        rolling_length=Rolling_Length, 
                                                        quantile_size=Quantile_Size,
                                                        label_size=Label_Size)
    
    # 輸出numpy矩陣為pickle檔(.pkl)
    Features.dump('mtf_Features.pkl')
    Labels.dump('mtf_Labels.pkl')
    Prices.dump('mtf_Prices.pkl')
    
    # 檢查Label分布（balance or imbalance?）
    unique, counts = np.unique(Labels, return_counts=True)
    print('Labels distribution:', dict(zip(unique, counts)))
    
    # 檢查Feature, Prices, Labels的shape
    print('Features.shape: {}'.format(np.array(Features).shape))
    print('Labels.shape: {}'.format(np.array(Labels).shape))
    print('Prices.shape: {}'.format(np.array(Prices).shape))
    
    # 輸出所有Miscellaneous圖片
    outputMiscellaneous(Features)
    
    
if __name__ == "__main__":
    main()

    
    
    
