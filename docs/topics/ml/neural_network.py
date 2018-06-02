import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import random
from sklearn.neural_network import MLPClassifier
from sklearn import datasets



#---------------------------------------------------------------------------------------
# Create some type of data structure to hold a node (a.k.a. neuron).
# Store the set of input weights for each node.
# Provide a way to create a layer of nodes of any number (this should be easily specified via a parameter).
# Account for a bias input.
# Produce 1 or 0 according to weight
# Load in Pima, Iris
# Normalize data maybe (x - mean / SD)
# np.insert(input_row, 0, -1)
#---------------------------------------------------------------------------------------


class Layer():
	def __init__(self, node_num = None):
		self.nodes = []
		self.node_num = node_num
		self.row_outputs = []
		
	def distribute(self, inputs):
		# number of inputs for a row, to create a weight
		# creates node
		# inputs = train_data
		self.init_node_weights(len(inputs), inputs)
		
		# This inserts the bias, or y-intercept
		row = np.insert(inputs, 0, -1)

		# giving a row to each node created 
		for node in self.nodes:
			node.array(row)
			
			# this appends the node activation amounts to each row
			# in feed fowards, these will become the new input values (activation ==> inputs)
			self.row_outputs.append(node.calculate_output(row))
				
	def init_node_weights(self, input_num, inputs):
		
		# for every node, get the weights. Weights determined by input values + 1
		for num in range(self.node_num):
			# +1 is for the bias
			node = Node(input_num + 1)
			
			# saves the node in a list to use in distribute() function
			self.nodes.append(node)
		

class Node():
	def __init__(self, input_num = None):
		self.weights = None
		self.h = 0
		self.a = 0
		self.input_num = input_num
		self.init_weights()
		self.inputs = []
		
	def array(self, inputs = None):
		for i in inputs:
			self.inputs.append(i)
			
	def init_weights(self):
		# creates random weights at first
		self.weights = np.random.uniform(-.5, .5
		, self.input_num)
		
	def calculate_h(self, inputs):		
		# X0W0 + X1W1 + ....
		for num in range(len(self.weights)):
			self.h += inputs[num] * self.weights[num]
		
	def calculate_a(self):
		# calculating activation
		self.a = 1/(1 + np.exp(-self.h))
		
	def calculate_output(self, inputs = None):
		self.calculate_h(inputs)
		self.calculate_a()
		return self.a
		
	def clear_node(self):
		# this way nodes don't remember activation between rows coming through
		self.h = 0
		self.a = 0


def read_data():
	pima = pd.read_csv("/Users/Wolfe/Desktop/Winter 2018/Machine Learning/pima.txt")

	# change 0 --> NA for columns 2 through 7
	pima.iloc[:,1:8] = pima.iloc[:,1:8].replace(0, np.NaN)

	# Drop all values with NaN
	pima.dropna(inplace=True)

	# Pandas --> Numpy Array
	pima = np.array(pima)
	
	data = pima

	# no scientific notation
	np.set_printoptions(suppress=True)

	#data = datasets.load_iris()
	
	return data
	
def normalize(data):
	data_normalized = preprocessing.normalize(data)
	return data_normalized
	
def split(data):

#	iris_data = data.data
#	iris_targets = data.target
#	
#	train_data, test_data, train_targets, test_targets = train_test_split(iris_data, iris_targets, test_size=.30)
	
	# It's -2 because it's the column_num - 1, plus you start at position 0, so it's -2 then
	train_data, test_data, train_targets, test_targets = train_test_split(data[:,0:len(data[0])-2], data[:,-1], test_size=.30)

	return train_data, test_data, train_targets, test_targets

def calculate_target_number(data):
	uniques, counts = np.unique(data, return_counts=True)
	number = len(uniques)
	return number
	
def calculate_accuracy(outputs, targets):
	# the activation output number will determine how many targets we need
	# if we have 3 rows of output, we need 3 targets
	row_amount = len(outputs)
	
	correct = 0
	
	# through every row in inputs, targets
	for pos in range(row_amount):
		# this stores the position of the highest activiation value among the outputs	
		max_position = outputs[pos].index(max(outputs[pos]))
		# if the position of the highest activation = the actual target
		if max_position == targets[pos]:
			correct += 1
	# return accuracy
	accuracy = (correct/row_amount)
	return accuracy

def get_amounts():
	layer_amount = int(input("Layer amount:  "))
	node_amount = int(input("Node amount per layer:  "))
	learning_rate = float(input("Learning Rate:  "))
	epoch_amount = int(input("Number of Epochs:  "))
	return layer_amount, node_amount, learning_rate, epoch_amount
	
def feed_forward(layer_amount, train_data, test_data, target_number, train_targets, test_targets, node_amount, learning_rate, epoch_amount):
	
	# Used for accessing weights in back propogation, all forward propogation after epoch and row 1
	# has layer_amount positions, each position having node_amount nodes
	# Accessed like node_array[which.layer][which_node]
	node_array = []
	
	# create an error list
	error_list = []
	epoch_list = []
	
	for epoch in range(epoch_amount):
		# this drop and creates the array for every epoch
		all_outputs = []
		
		# for every row in the train data
		for row in range(len(train_data)):
			
			# node and layer creation happens on the first row
			if row == 0 and epoch == 0:
				
				# feed forward is done through the number 'layer_amount' amount of layers
				# for every layer
				for num in range(layer_amount):
					# we want data to determine input amounts
					if num == 0:
						new_layer = Layer(node_amount)
						new_layer.distribute(train_data[row])
						outputs = new_layer.row_outputs
						node_array.append(new_layer.nodes)
						
						
					# last row: targets are the target nodes
					elif num == layer_amount - 1:
						new_layer = Layer(target_number)
						new_layer.distribute(outputs)
						node_array.append(new_layer.nodes)
						
					# outputs become inputs for new layer
					else:
						new_layer = Layer(node_amount)
						new_layer.distribute(outputs)
						outputs = new_layer.row_outputs	
						node_array.append(new_layer.nodes)
				
				# CALL THE BACK PROPOGATION FUNCTION(node_array)
				# targets[row] is the actual target for that row
				back_propogation(node_array, node_amount, layer_amount, train_targets[row], target_number, learning_rate)
				
				# Clean out A and H values
				# Empty out inputs
				for layer in range(layer_amount):
					if layer == layer_amount - 1:
						for node in range(target_number):
							node_array[layer][node].clear_node()
							node_array[layer][node].inputs = []
					else:
						for node in range(node_amount):
							node_array[layer][node].clear_node()
							node_array[layer][node].inputs = []
				
				
			# Do the process for the rest of the rows							
			else:
				
				outputs = []
				# for every layers
				for layer in range(layer_amount):
					if layer == 0:
						for node in range(node_amount):
							# This inserts the bias, or y-intercept
							data_row = np.insert(train_data[row], 0, -1)
							# node inputs = first training row
							node_array[layer][node].array(data_row)
							a = node_array[layer][node].calculate_output(data_row)
							outputs.append(a)
							
					elif layer == layer_amount - 1:
						for node in range(target_number):
							if node == 0:
								inputs = outputs[:]
								outputs = []
							data_row = np.insert(inputs, 0, -1)
							# node inputs = prior activation
							node_array[layer][node].array(data_row)
							node_array[layer][node].calculate_output(data_row)

					else:
						for node in range(node_amount):
							# this copies the list
							# we need the values for data_row, but also new values
							if node == 0:
								inputs = outputs[:]
								outputs = []
							data_row = np.insert(inputs, 0, -1)
							# node inputs = prior activation
							node_array[layer][node].array(data_row)
							a = node_array[layer][node].calculate_output(data_row)
							outputs.append(a)
				
				back_propogation(node_array, node_amount, layer_amount, train_targets[row], target_number, learning_rate)
				
				for layer in range(layer_amount):
					if layer == layer_amount - 1:
						for node in range(target_number):
							node_array[layer][node].clear_node()
							node_array[layer][node].inputs = []
					else:
						for node in range(node_amount):
							node_array[layer][node].clear_node()
							node_array[layer][node].inputs = []
							
							
		# For every row in test_data (per epoch), get the outputs, but don't backpropogate. From these outputs we calcuate the accuracy	
		for row in range(len(test_data)):
			final_outputs = []
			# access every layers
			for layer in range(layer_amount):
				if layer == 0:
					#and every node of that layer
					for node in range(node_amount):
						# This inserts the bias, or y-intercept
						data_row = np.insert(test_data[row], 0, -1)
						# node inputs = first training row
						node_array[layer][node].array(data_row)
						a = node_array[layer][node].calculate_output(data_row)
						outputs.append(a)
						
				elif layer == layer_amount - 1:
					for node in range(target_number):
						if node == 0:
							inputs = outputs[:]
							outputs = []
						data_row = np.insert(inputs, 0, -1)
						# node inputs = prior activation
						node_array[layer][node].array(data_row)
						a = node_array[layer][node].calculate_output(data_row)
						# store all activation amounts of outputs layer nodes from every test_data row
						final_outputs.append(a)

				else:
					for node in range(node_amount):
						# this copies the list
						# we need the values for data_row, but also new values
						if node == 0:
							inputs = outputs[:]
							outputs = []
						data_row = np.insert(inputs, 0, -1)
						# node inputs = prior activation
						node_array[layer][node].array(data_row)
						a = node_array[layer][node].calculate_output(data_row)
						outputs.append(a)
			
			# append the activation values for each row			
			all_outputs.append(final_outputs)

			
			# clear the nodes between each row
			for layer in range(layer_amount):
				if layer == layer_amount - 1:
					for node in range(target_number):
						node_array[layer][node].clear_node()
						node_array[layer][node].inputs = []
				else:
					for node in range(node_amount):
						node_array[layer][node].clear_node()
						node_array[layer][node].inputs = []
									
		#print(all_outputs)
		# calculate accuracy for the dataset
		accuracy = calculate_accuracy(all_outputs, test_targets)
		error = 1 - accuracy
		error_list.append(error)
		epoch_list.append(epoch + 1)
	print(error_list)
	return error_list, epoch_list
		
def back_propogation(node_array, node_amount, layer_amount, target, target_number, learning_rate):
	
	error_array = np.zeros([layer_amount, node_amount], dtype=np.float)
	
	###############################
	# ERROR CALCULATION
	###############################
	# iterate backwards in every layer
	# -1 because it will stop at 0, or node_array[0]
	# This large for-loop does the error calculation for each node in each layer
	
	
	for layer in range(layer_amount - 1, -1, -1):
		
		# This is the actual output layer
		if layer == layer_amount - 1:
			
			# The node amount = target amount in output layer
			# -1 for the same reason. node_array[layer][0]
			for pos in range(target_number - 1, -1, -1):
				# This is the activation
				a = node_array[layer][pos].a
				error = a*(1-a)*(a - target)
				error_array[layer][pos] = error
					
		# This is every other layer
		else:
			# basically for every new node, we need to access 1 smaller position in weights. 
			# weights[node amount - 1], [node_amount -2], etc
			node_iteration = 1
			# for every node
			for pos in range(node_amount - 1, -1, -1):
				
				error = 0
				# for every node and weight of the prior layer
				# since we are going down, +1 is the prior position in the array
				for place in range(len(node_array[layer+1]) - 1, -1, -1):
					# get get weight from node K
					weight = node_array[layer+1][place].weights[node_amount - node_iteration]
					# get error from Node K
					last_error = error_array[layer+1][place]
					
					# dot product
					error += weight*last_error
					
					# the part above did the dot product part. After the last one, we want to do the a*(1-a) derivative part
					if place == 0:
						a = node_array[layer][pos].a
						error = a*(1-a)*error
						error_array[layer][pos] = error

				node_iteration += 1
				
	
	###############################
	# WEIGHT UPDATING
	###############################
	
	# For each layer
	# For each node
	# For each weight
	
	for layer in range(layer_amount):
		
		# If the last layer
		if layer == layer_amount - 1:
			# only iterate over targets, not node amounts of past layers
			for node in range(target_number):
				# for every weight and input
				for pos in range(len(node_array[layer][node].weights)):
					current = node_array[layer][node].weights[pos]
					error = error_array[layer][node]
					value_input = node_array[layer][node].inputs[pos]
					# the update weights formula
					current = current - learning_rate*(error)*(value_input)
					# update the value
					node_array[layer][node].weights[pos] = current
			
		else:
			# for each node
			for node in range(node_amount):
			# for each weight
				for pos in range(len(node_array[layer][node].weights)):
					current = node_array[layer][node].weights[pos]
					error = error_array[layer][node]
					value_input = node_array[layer][node].inputs[pos]
					# the update weights formula
					current = current - learning_rate*(error)*(value_input)
					# update the value
					node_array[layer][node].weights[pos] = current

def sklearn(train_data, train_targets, test_data, test_targets, epoch_amount, learning_rate):
	clf = MLPClassifier(hidden_layer_sizes=(4, 4), solver = 'sgd', learning_rate_init = learning_rate, max_iter = epoch_amount)

	y = clf.fit(train_data, train_targets).predict(test_data)
	
	correct = 0
	for pos in range(len(test_targets)):
		if test_targets[pos] == y[pos]:
			correct += 1
	
	return round(correct/len(test_targets), 2)
			
		
def main():
	data = read_data()
	
	train_data, test_data, train_targets, test_targets = split(data)
	
	# normalize the data
	train_data = normalize(train_data)
	test_data = normalize(test_data)
	
	# Get the target column
	targets = data[:,-1]
	#targets = data.target
	
	# get how many unique targets there are. This decides how many output nodes to have
	target_number = calculate_target_number(targets)
	
	# ask user how many layers  and nodes per layer they want
	layer_amount, node_amount, learning_rate, epoch_amount = get_amounts()
	
	sklearn_predictions = sklearn(train_data, train_targets, test_data, test_targets, epoch_amount, learning_rate)
	print(sklearn_predictions)
	
	# send it off to be fed_forward
	# create empty error, epoch list to be received
	error = []
	epochs = []
	error, epochs = feed_forward(layer_amount, train_data, test_data, target_number, train_targets, test_targets, node_amount, learning_rate, epoch_amount)
	
	d = {'Error' : error, 'Epoch' : epochs}
	neural_network = pd.DataFrame(d)

	neural_network.to_csv('/Users/Wolfe/Desktop/Winter 2018/Machine Learning/neural_network.csv', sep=',')
	
	

if __name__ == "__main__":
	main()