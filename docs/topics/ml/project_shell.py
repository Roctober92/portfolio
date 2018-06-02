'''
Script for Machine Learning Project
Andrew Wolfe
Ryan
Jeremy Chandler

1. Get predictions for 2 sizes of data
2. Store accuracy in dataset in array for comparison

mortality_small_data = categorical, 945 rows, 3 predictors, 15 targets, 7% guess
older_data = categorical, 2724 rows, 4 predictors, 59 targets, 2% guess
shelter = categorical, 5000 rows, 5 predictors, 4 targets, 25% guess
pima = numerical, 767 rows, 8 predictors, 2 targets, 50% guess
chess = categorical, 3195 rows, 36 predictors, 2 targets, 50% guess
car = categorical, 1728 rows, 6 predictors, 4 targets, 25% guess
walmart = categorical, 5000 rows, 5 predictors, 10 targets, 10% guess
abalone = 7/8 numeric, 4176 rows, 8 predictors, 6 targets, 17% guess
'''

##########################################################################
##### LIBRARIES
##########################################################################

import pandas as pd
import numpy as np 
from operator import itemgetter                         ### For sorting accuracy
from sklearn.model_selection import train_test_split    ### Split the Data
from sklearn import preprocessing                       ### For normalizing and encoding
from sklearn.neural_network import MLPClassifier        # Neural Network
from sklearn.naive_bayes import GaussianNB              # Naive Bayes
from sklearn.ensemble import RandomForestClassifier     # Random Forest
from sklearn import tree                                # Decision Tree
from sklearn.model_selection import cross_val_score     ### Cross Validation
from sklearn.neighbors import KNeighborsClassifier      # Knn 
from sklearn import svm                                 # Support Vector Machine
from sklearn import datasets                            # For Iris


##########################################################################
##### ENSEMBLE
##########################################################################
def knn(data, splits, k):
	#train_data, test_data, train_targets, test_targets = split(data)
	#predictions = clf.fit(train_data, train_targets).predict(test_data)
	#return calculate_accuracy(predictions, test_targets)
	
	clf = KNeighborsClassifier(n_neighbors = k)

	scores = cross_val_score(clf, data[:,0:len(data[0])-2], data[:,-1], cv = splits)
	
	return round(scores.mean(), 2)
	
	
	
def neural(data, splits, init_rate, epochs):
	#train_data, test_data, train_targets, test_targets = split(data)
	#predictions = clf.fit(train_data, train_targets).predict(test_data)
	#return calculate_accuracy(predictions, test_targets)
	
	clf = MLPClassifier(hidden_layer_sizes=(3, 3, 3, 3), solver = 'sgd', learning_rate_init = init_rate, max_iter = epochs)

	scores = cross_val_score(clf, data[:,0:len(data[0])-2], data[:,-1], cv = splits)
	
	return round(scores.mean(), 2)
	
	
	
def dtree(data, splits):
	#train_data, test_data, train_targets, test_targets = split(data)
	#	predictions = clf.fit(train_data, train_targets).predict(test_data)
	#	return calculate_accuracy(predictions, test_targets)
	
	clf = tree.DecisionTreeClassifier()

	scores = cross_val_score(clf, data[:,0:len(data[0])-2], data[:,-1], cv = splits)
	
	return round(scores.mean(), 2)
	
	
	
def naive(data, splits):
	#train_data, test_data, train_targets, test_targets = split(data)
	#	predictions = clf.fit(train_data, train_targets).predict(test_data)
	#	return calculate_accuracy(predictions, test_targets)
	
	clf = GaussianNB()
	scores = cross_val_score(clf, data[:,0:len(data[0])-2], data[:,-1], cv = splits)
	
	return round(scores.mean(), 2)
	
	
	
	
def vector(data, splits):
	#train_data, test_data, train_targets, test_targets = split(data)
 	#return calculate_accuracy(predictions, test_targets)

	clf = svm.SVC(kernel= 'rbf' , gamma = 2, C = 1)
	
	scores = cross_val_score(clf, data[:,0:len(data[0])-2], data[:,-1], cv = splits)
	
	return round(scores.mean(), 2)
	
	
	
def forest(data, splits, num_trees):
	#train_data, test_data, train_targets, test_targets = split(data)
	#	predictions = clf.fit(train_data, train_targets).predict(test_data)
	#	return calculate_accuracy(predictions, test_targets)
	
	
	clf = RandomForestClassifier(n_estimators = num_trees)
	scores = cross_val_score(clf, data[:,0:len(data[0])-2], data[:,-1], cv = splits)
	
	return round(scores.mean(), 2)
	
	
##########################################################################
##### ACCURACY AND SPLITTING (only used without kfold cross validation)
##########################################################################
	
#def split(data):
#	
#	# this is assuming the targets are on the last column
#	# data[0] gives us the column amount, so our data is all the columns - 1, so positions 0 to col_num - 2
#	return train_data, test_data, train_targets, test_targets = train_test_split(data[:,0:len(data[0])-2], data[:,-1], test_size=.30)
#	
#	
#def calculate_accuracy(predictions, targets):
#	# Expected output to be a number 0-1, rounded 2 decimals
#	
#	correct = 0
#	for position in range(len(targets)):
#		if test_targets[position] == predictions[position]:
#			correct += 1
#		
#	return round(correct/len(targets), 2)
	
	
	
	
##########################################################################
##### ORGANIZATION AND OPERATIONS OF PROGRAM
##########################################################################	


def data_one():
	
	# Read in small data
	data = pd.read_csv("/Users/Wolfe/Desktop/Winter 2018/Machine Learning/mortality_small.csv")
	#data = pd.read_csv("/Users/Wolfe/Desktop/Python/car.data.txt")
	#data = pd.read_csv("/Users/Wolfe/Desktop/Winter 2018/Machine Learning/shelter_sample.csv")
#	data = pd.read_csv("/Users/Wolfe/Desktop/Winter 2018/Machine Learning/abalone.csv")
#	data.iloc[:,0:7] = preprocessing.normalize(data.iloc[:,0:7])
	
	le = preprocessing.LabelEncoder()
	data = data.apply(le.fit_transform)
	
	data = np.array(data)

	return data
	
def data_two():
	# Read in larger data
	#data = datasets.load_iris()
	data = pd.read_csv("/Users/Wolfe/Desktop/Winter 2018/Machine Learning/pima.txt")
	data.iloc[:,1:8] = data.iloc[:,1:8].replace(0, np.NaN)
	data.dropna(inplace=True)
	data.iloc[:,1:8] = preprocessing.normalize(data.iloc[:,1:8])
	
	#data = pd.read_csv("/Users/Wolfe/Desktop/Winter 2018/Machine Learning/chess.data.txt")
	#data = pd.read_csv("/Users/Wolfe/Desktop/Winter 2018/Machine Learning/older_data.csv")
	#data = pd.read_csv("/Users/Wolfe/Desktop/Winter 2018/Machine Learning/walmart_data.csv")
	
	le = preprocessing.LabelEncoder()
	data = data.apply(le.fit_transform)
	
	data = np.array(data)
	
	return data

def get_parameters():
	
	# Ask user for number of kfold splits
	splits = int(input("How many cross-validation splits do we want? "))
	
	# Get nearest-neighbor number
	k1 = int(input("How many nearest neighbors do we want for first KNN? "))
	k2 = int(input("How many nearest neighbors do we want for second KNN? "))
	
	# Get neural network parameters
	init_rate1 = float(input("What init rate do we want for 1st NN? "))
	epochs1 = int(input("How many epochs do we want for 1st NN? "))
	
	init_rate2 = float(input("What init rate do we want for 2nd NN? "))
	epochs2 = int(input("How many epochs do we want for 2nd NN? "))
	
	# Get amount of trees for the forest
	num_trees1 = int(input("How many trees for 1st Forest? "))
	num_trees2 = int(input("How many trees for 2nd Forest? "))
	
	return splits, k1, k2, init_rate1, epochs1, init_rate2, epochs2, num_trees1, num_trees2



############################################
# DATASET --> ALGORITHMS
############################################
def ensemble(data1, data2):
	# We'll iterrate every combination of these
	# text_list contains "large" or "small"
	# data_list the actual large or small data
	text_list = [0, 1]
	data_list = [data1, data2]
	algorithms = [0, 1, 2, 3, 4, 5, 6, 7, 8]
	
	# Number of algorithms
	algorithm_num = len(algorithms)
	
	# Number of rows, which should be all combinations
	row_num = algorithm_num*len(data_list)
	
	# Contains accuracy, size, and type information of each algorithm
	accuracy_array = np.zeros([row_num, 3], dtype=np.float)
	
	# Get parameters for the different algorithms
	splits, k1, k2, init_rate1, epochs1, init_rate2, epochs2, num_trees1, num_trees2 = get_parameters()
	
	print()
	print("There will be 4 hidden layers, 3 nodes each")
	print()
	
	# Pass data into algorithms
	# Extract accuracies from each
	# set them to be displayed
	
	# row of accuracy array. Each combination of algorithm and size gets a row
	row = 0
	text_list_iteration = 0
	
	for dataset in data_list:
			
		accuracy_array[row][0] = knn(dataset, splits, k1)
		accuracy_array[row][1] = text_list[text_list_iteration]
		accuracy_array[row][2] = algorithms[0]
			 
		accuracy_array[row+1][0] = neural(dataset, splits, init_rate1, epochs1)
		accuracy_array[row+1][1] = text_list[text_list_iteration]
		accuracy_array[row+1][2] = algorithms[1]
			
		accuracy_array[row+2][0] = dtree(dataset, splits)
		accuracy_array[row+2][1] = text_list[text_list_iteration]
		accuracy_array[row+2][2] = algorithms[2]
			
		accuracy_array[row+3][0] = naive(dataset, splits)
		accuracy_array[row+3][1] = text_list[text_list_iteration]
		accuracy_array[row+3][2] = algorithms[3]
			
		accuracy_array[row+4][0] = vector(dataset, splits)
		accuracy_array[row+4][1] = text_list[text_list_iteration]
		accuracy_array[row+4][2] = algorithms[4]
			
		accuracy_array[row+5][0] = forest(dataset, splits, num_trees1)
		accuracy_array[row+5][1] = text_list[text_list_iteration]
		accuracy_array[row+5][2] = algorithms[5]
		
		accuracy_array[row+6][0] = knn(dataset, splits, k2)
		accuracy_array[row+6][1] = text_list[text_list_iteration]
		accuracy_array[row+6][2] = algorithms[6]
		
		accuracy_array[row+7][0] = neural(dataset, splits, init_rate2, epochs2)
		accuracy_array[row+7][1] = text_list[text_list_iteration]
		accuracy_array[row+7][2] = algorithms[7]
		
		accuracy_array[row+8][0] = forest(dataset, splits, num_trees2)
		accuracy_array[row+8][1] = text_list[text_list_iteration]
		accuracy_array[row+8][2] = algorithms[8]
			
		# If is started at row 0, next round it starts at 0 + algorithm_num
		# Then next iterritive combination will occupy rows algorithm_num:algorithm_num*2 - 1
		row += algorithm_num
		text_list_iteration += 1
			
					
	# Put parameters in a list more easily pass through functions
	parameters = [k1, k2, init_rate1, epochs1, init_rate2, epochs2, num_trees1, num_trees2]
	
	# Send it to rank_display to be sorted and displayed
	sort_array(accuracy_array, row_num, parameters)
				
	
	
	
def sort_array(accuracy_array, row_num, parameters):
	
	# Declare array where we'll put the sorted accuracy array
	sorted_array = np.zeros([row_num, 3], dtype=np.str)
	
	# Sort accuracy array
	# 0 means sort on column 0
	# reverse = TRUE means descending order
	sorted_array = sorted(accuracy_array, key=itemgetter(0), reverse = True)
	
	export_results(sorted_array, parameters)
	
	
	

############################################
# ARRAY --> 3 LISTS
# LISTS --> EXCEL
############################################	
def export_results(sorted_array, parameters):
	
	# Declare lists to be exported
#	accuracy = []
#	size = []
#	algorithm = []
	
	# array --> Lists
#	for pos in range(len(sorted_array)):
#		accuracy.append(sorted_array[pos][0])
#		size.append(sorted_array[pos][1])
#		algorithm.append(sorted_array[pos][2])
	
	# Do what is done above, but in list comprehension form
	accuracy = [sorted_array[pos][0] for pos in range(len(sorted_array))]
	size = [sorted_array[pos][1] for pos in range(len(sorted_array))]
	algorithm = [sorted_array[pos][2] for pos in range(len(sorted_array))]
		
	# Get the meanings for each number
	size_text, algorithm_text = interpret_lists(size, algorithm, parameters)
		
	# no scientific notation
	np.set_printoptions(suppress=True)
		
	print()
	print("Here is our ensemble: It is sorted by highest accuracy descending order")
	print()
		
	print(accuracy)
	print(size_text)
	print(algorithm_text)
	
	# Export results to excel spreadsheet
	d = {'Accuracy' : accuracy, 'Dataset_Size' : size_text, 'Algorithm_Type' : algorithm_text}
	ensemble_learning = pd.DataFrame(d)

	#ensemble_learning.to_csv('/Users/Wolfe/Desktop/Winter 2018/Machine Learning/ensemble_learning.csv', sep=',')
	
	
	
############################################
# NUMBERS --> MEANINGS
############################################	
def interpret_lists(size, algorithm, parameters):
	
	# Declare lists
	size_text = []
	algorithm_text = []
	
	# Get descriptions of each data from user
	data_one_description = input("Description of 1st dataset: ")
	data_two_description = input("Description of 2nd dataset: ")

	# Switch the meanings for the text size lists
#	for pos in range(len(size)):
#		if size[pos] == 0:
#			size_text.append(data_one_description)
#		else:
#			size_text.append(data_two_description)
	
	size_text = [data_one_description if size[pos] == 0 else data_two_description for pos in range(len(size))]
			

	# Switch the meanings for the algorithm name lists
	for pos in range(len(algorithm)):
		if algorithm[pos] == 0:
			algorithm_text.append('KNN with ' + str(parameters[0]) + ' neighbors')
		elif algorithm[pos] == 1:
			algorithm_text.append('Neural Network with ' + str(parameters[2]) + ' learning rate and ' + str(parameters[3]) + ' epochs')
		elif algorithm[pos] == 2:
			algorithm_text.append('Decision Tree')
		elif algorithm[pos] == 3:
			algorithm_text.append('Naive Bayes')
		elif algorithm[pos] == 4:
			algorithm_text.append('SVM')
		elif algorithm[pos] == 5:
			algorithm_text.append('Random Forest: ' + str(parameters[6]) + ' trees')
		elif algorithm[pos] == 6:
			algorithm_text.append('KNN with ' + str(parameters[1]) + ' neighbors')
		elif algorithm[pos] == 7:
			algorithm_text.append('Neural Network with ' + str(parameters[4]) + ' learning rate and ' + str(parameters[5]) + ' epochs')
		else:
			algorithm_text.append('Random Forest: ' + str(parameters[7]) + ' trees')
			
	return size_text, algorithm_text


def main():
	
	# Get data
	data1 = data_one()
	data2 = data_two()
	
	# Send them off to the ensemble
	ensemble(data1, data2)
	

if __name__ == "__main__":
	main()
