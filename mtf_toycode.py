# 寫一個函數來得到 MTF 
# 輸入 : 
#        - all_ts      < 1 x N double > 時間序列資料
#        - window_size < int > 資料窗格大小(類似技術指標的參數)
#        - rolling_length < int > 每次滾動移動大小
#            - 可以抓 window_size 的 1/8 ~ 1/10 做觀察, 若和 window_size 一樣，資料就不會重疊
#        - quantile_size < int > 分位數，就是可以快速地做 K 分位數 （當 quantile_size = 4 就會有四分位數 )
#---------------------------------------------------------------------------------------------------------

import numpy as np
import time

# 初始化空陣列函式
def nullMatrix(n,m,q):
	matrix = []
	for i in range(n):
		tmp = []
		for j in range(m):
			tmp.append(np.zeros((q,q), float))
		matrix.append(tmp)
	return np.array(matrix)


def MTF(all_ts, window_size, rolling_length, quantile_size):

	# 取得時間序列長度
	n = len(all_ts) 
	
	# 由於我們要比較前後來算有沒有狀態轉換, 所以真正的 window_size 會是原本的 +1
	real_window_size = window_size + 1 
	
	# 根據我們的滾動大小，將資料切成一組一組
	n_rolling_data = int(np.floor((n - real_window_size)/rolling_length)) 
	
	# 真正的 feature 會有 Q x Q　個
	feature_size = quantile_size * quantile_size 
	
	# 最終的 MTF
	markov_field = nullMatrix(n_rolling_data, feature_size, window_size) 
	
	# 開始從第一筆資料前進
	for i_rolling_data in range(n_rolling_data):
	
		# 先宣告一個「矩陣的矩陣」
		this_markov_field =  nullMatrix(window_size, window_size, quantile_size)
		
		# 整個窗格的資料先從輸入時間序列中取出來
		full_window_data =  all_ts[i_rolling_data*rolling_length : i_rolling_data*rolling_length+real_window_size]
		
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
								if this_window_data[i_this_window_data] < this_quantile[i_quantile] and this_window_data[i_this_window_data] >= this_quantile[i_quantile-1]:
									if this_window_data[i_this_window_data+1] < this_quantile[j_quantile] and this_window_data[i_this_window_data+1] >= this_quantile[j_quantile-1]:
										this_markov_matrix[ i_quantile-1 , j_quantile-1 ] = this_markov_matrix[ i_quantile-1 , j_quantile-1 ] + 1
									
					
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
				seperated_markov_matrix = np.zeros( (window_size, window_size),float )
				
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
							seperated_markov_matrix[ i_window_size, j_window_size ] = 0
				
				# 再把拆分出來的矩陣，放到整個滾動資料的對應位置
				markov_field[i_rolling_data,feature_count] = seperated_markov_matrix
				feature_count += 1
			
		# 完成!
		
		##############################
		#.....想要畫圖來確認對錯.....#
		##############################

		
def main():
	source = [1,2,3,4,5,6,7,8,98,0,7,5,45,6,65,645,6,6,6,5,5,5,5,6,4,3,7,9,6,54,7,7,0,0,0,4,5,7,9,1,0,745,4,5,45,0]	
	MTF(source, 10, 1, 4)
		
		
if __name__ == "__main__":
	main()
	
	
	
	