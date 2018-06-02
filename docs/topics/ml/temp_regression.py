'''
Here we are estimating temperature values through KNN regressor (or other regressors)

This is used to help in Wolfe Senior Project, the analysis of rising temperatures upon snowpack at different elevations

22 May 2018

We'll need to one hot encode 3 categorical variabels, then normalize the datasets

EXAMPLE: http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example
'''
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score





###################### Data Wrangling / Preparation ###########################

def get_data():
	#####
	# Read in CSV from to_knn.R
	#####
	
	
	# Read in csv
	data_32 = pd.read_csv("/Users/Wolfe/Desktop/spring_2018/Senior_Project/data/to_python/data32.csv")
	high_max = pd.read_csv("/Users/Wolfe/Desktop/spring_2018/Senior_Project/data/to_python/high_max_selected.csv")
	max_mid = pd.read_csv("/Users/Wolfe/Desktop/spring_2018/Senior_Project/data/to_python/max_mid_selected.csv")
	high_min = pd.read_csv("/Users/Wolfe/Desktop/spring_2018/Senior_Project/data/to_python/high_min_selected.csv")
	low_max = pd.read_csv("/Users/Wolfe/Desktop/spring_2018/Senior_Project/data/to_python/low_max_selected.csv")
	min_mid = pd.read_csv("/Users/Wolfe/Desktop/spring_2018/Senior_Project/data/to_python/min_mid_selected.csv")
	low_min = pd.read_csv("/Users/Wolfe/Desktop/spring_2018/Senior_Project/data/to_python/low_min_selected.csv")
	
	print()
	
	# modify returns a list of changed/wrangled datasets
	return(modify(data_32, high_max, max_mid, high_min, low_max, min_mid, low_min))



def modify(d1, d2, d3, d4, d5, d6, d7):
	#####
	# Makes dummy variable for 'mon' column in all datasets
	#####
	
	
	# List of DFs to use in a loop
	df_list = [d1, d2, d3, d4, d5, d6, d7]
	new_list = []
	
	for df in df_list:
		# One-hot encode month column
		df = pd.concat([df,pd.get_dummies(df['mon'], prefix='mon')], axis=1)


		# Drop regular month column: we don't need it anymore
		df.drop(['mon'],axis=1, inplace=True)
		
		# append modified df
		new_list.append(df)
		
	# Print shape to check column amounts
	#shapes = [df.shape for df in new_list]
	#print(shapes)
	
	return new_list


def correlation(pred, targets):
	######
	# Return correlation
	# preds = predictions
	# targets = given temperature from middle 99.6% of data
	######
	
	return round(r2_score(targets, pred), 3)
	
	

	
def train_split(data, target_column):
	
	# Targets are 1st column
	train_data, test_data, train_targets, test_targets = train_test_split(data.iloc[:,1:], data[target_column], test_size=.30)
	
	return train_data, test_data, train_targets, test_targets



def export(predictions, data_type):
	######
	# prepares and dispatches data
	######
	
	# makes dataframe0
	pred = {'predictions_' + str(data_type): predictions}
	pred = pd.DataFrame(pred)

	# Export data
	pred.to_csv('/Users/Wolfe/Desktop/spring_2018/Senior_Project/data/knn_to_r/' + data_type + '.csv', sep=',')





###################### Checking Regression Strength ###########################	
	
def check_high(data):
	#######
	# Checks accuracy of regressor on high temps
	# High 10 - .852
	#######
	
	# split data
	train_data, test_data, train_targets, test_targets = train_split(data, 'max_temp_mean')
	
	
	
	n = int(input("How many neighbors for high temp? "))

	print()
	
	neigh = KNeighborsRegressor(n_neighbors = n)
	
	predictions = neigh.fit(train_data, train_targets).predict(test_data)
	
	print("High temp correlation is {}".format(correlation(predictions, test_targets)))
	print()
	
	
	
def check_low(data):
	#######
	# Checks accuracy of regressor on low temps
	# Low 15 - .84"
	#######
	
	# split data
	train_data, test_data, train_targets, test_targets = train_split(data, 'min_temp_mean')
	
	
	
	n = int(input("How many neighbors for low temp? "))

	print()
	
	neigh = KNeighborsRegressor(n_neighbors = n)
	
	predictions = neigh.fit(train_data, train_targets).predict(test_data)
	
	print("Low temp correlation is {}".format(correlation(predictions, test_targets)))
	





###################### Prediction Functions ###########################

def make_predictions(datasets):
	######
	# Distribute datasets to each function
	######

	predict(datasets[0], datasets[2], "high_32_preds")
	predict(datasets[0], datasets[5], "low_32_preds")
	predict(datasets[1], datasets[2], "high_max_preds")
	predict(datasets[3], datasets[5], "high_min_preds")
	predict(datasets[4], datasets[2], "low_max_preds")
	predict(datasets[6], datasets[5], "low_min_preds")
	
	
def predict(extreme, mid, data_type):

	n = int(input("How many neighbors? "))
	
	print()
	
	neigh = KNeighborsRegressor(n_neighbors = n)
	
	predictions = neigh.fit(mid.iloc[:,1:], mid.iloc[:,0]).predict(extreme)
	
	# Send data to be prepared and sent off
	export(predictions, data_type)
	

	
	





def main():
	# Get list of modified datasets
	datasets = get_data()
		
	# check accuracy of high and low temps with regressor
	check_high(datasets[2])
	check_low(datasets[5])
	
	# Make predictions! Correlation of 85% seems reliable enough
	#make_predictions(datasets)
	
	
	
if __name__ == "__main__":
	main()









