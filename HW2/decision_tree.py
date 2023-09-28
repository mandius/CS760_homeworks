import matplotlib.pyplot as plt
import math

def parse_dataset_file(filename):
	fh = open(filename, "r")
	entries = fh.readlines()
	final_list= []
	
	for entry in entries:
		entry_list = entry.split(" ")
		entry_list_float =[float(entry_list[0]),float(entry_list[1]),int(entry_list[2])]
		final_list.append(entry_list_float)
	return final_list

def calc_err_n (expected, actual):
	count=0
	for i in range(0,len(expected)-1):
		if not (expected[i]== actual[i]):
			count = count+1
	return count

def plt_dataset(dataset):
	X1 = []
	X2 = []
	Y = []
	red = (1,0,0)
	blue = (0,0,1)
	for i in dataset:
		X1.append(i[0])
		X2.append(i[1])
		if(i[2] == 0):
			Y.append(red)
		else:
			Y.append(blue)
	

	plt.scatter(x = X1, y= X2, c = Y)
	plt.show()
	
	
def log2(val):
	return math.log(val)/ math.log(2)		


class node:

	def __init__(self):
		self.left = None
		self.right = None
		self.is_leaf_= 0
		self.feature_index =0
		self.threshold=0.0
		self.prediction=0

	def predict(self, x):
		if self.is_leaf:
			return self.prediction
		else:
			if(x[self.feature_index]>=self.threshold):
				return self.right.predict(x)
			else:
				return self.left.predict(x)

	def get_indent(self, indent):
		return "   "*indent
	
	def print_node(self, fh, indent):
		if(self.is_leaf):
			fh.write(self.get_indent(indent) + "predict :" + str(self.prediction)+"\n")
		else:
			fh.write(self.get_indent(indent) + "if x[" + str(self.feature_index) +"] >= " + str(self.threshold) + ":\n")
			
			self.right.print_node(fh,indent+1)
			fh.write(self.get_indent(indent) + "else:\n")
			self.left.print_node(fh, indent+1)

class simple_tree:
	def __init__(self):
		self.n_nodes=0
		self.root = None

	def test(self, test_set):
		correct =0
		err= 0
		total = len(test_set)
		for i in test_set:
			if(self.predict([i[0] ,i[1]]) == i[2]):
				correct = correct +1
			else:
				err = err+1
		print("Total Items: " + str(total) + " Error: " + str(err) + " Correct: " + str(correct))
		print("Error percentage: " + str((float(err)/total)*100))
		return( err)

		

	def print_tree(self,fh):
		self.root.print_node(fh, 0)

	def predict(self, x):
		return self.root.predict(x)

	def divide_training_set(self, training_set, split):
		subset_left = []
		subset_right = []
		index = split[0]
		threshold = split[1]
		for item in training_set:
			if(item[index] >= threshold):
				subset_right.append(item)
			else:
				subset_left.append(item)
		return [subset_left, subset_right]

	
	def build_tree(self, training_set):
		self.root = self.build_subtree( training_set)

	def build_subtree(self, training_set):
		root_node = node()
		self.n_nodes = self.n_nodes+1
		candidate_splits = self.find_candidate_splits(training_set)
		stopping_criteria_l =self.stopping_criteria(candidate_splits, training_set)
		if stopping_criteria_l[0]==1:
			root_node.is_leaf=1
			root_node.prediction=stopping_criteria_l[1]
		else:
			
			root_node.is_leaf=0
			best_split = self.find_best_split(training_set, candidate_splits)
			root_node.feature_index = best_split[0]
			root_node.threshold = best_split[1]
			[training_set_left, training_set_right] = self.divide_training_set( training_set, best_split)
			root_node.left = self.build_subtree( training_set_left)
			root_node.right = self.build_subtree(training_set_right)
			
		return root_node
			
	def sort( self, training_set, index):
		sorted_set = []
		training_set_temp = training_set
		##Using bubble sort
		for i in range(0, len(training_set_temp)-1):
			for j in range(0, len(training_set_temp)-i-1):
				if(training_set_temp[j] > training_set_temp[j+1]):
					temp = training_set_temp[j]
					training_set_temp[j] = training_set_temp[j+1]
					training_set_temp[j+1] = temp
		return training_set_temp
				
	def find_candidate_splits(self, training_set):
		candidate_splits = []
		for i in range(0,2):
			training_set_sorted = self.sort( training_set, i)
			for k in range(0, len(training_set_sorted)):
				if not k==0:
					if not (training_set_sorted[k][2] == training_set_sorted[k-1][2]):
						split = [i, training_set_sorted[k][i]]
						candidate_splits.append(split)
		return candidate_splits 
					
				
		
	def calculate_majority_label(self, training_set):
		num_0 = 0
		num_1 = 0
		for i in training_set:
			if(i[2]==0):
				num_0=num_0+1
			else:	
				num_1=num_1 +1
		if num_0 > num_1:
			return 0
		else:
			return 1
		
	
	def calculate_entropy_of_split(self, training_set, split):
		items_right=0
		items_left=0
		items_total = len(training_set)
		index =split[0]
		threshold =split[1]
		for item in training_set:
			if(item[index]>= threshold):
				items_right = items_right+1
			else:
				items_left = items_left +1

		
		p_left = float(items_left)/float(items_total)
		p_right = float(items_right)/float(items_total)

		if (p_left==0) or (p_right==0):
			return 0
		
		return -1 * ( p_left*log2(p_left) + p_right*log2(p_right))


	def calc_gain_ratio(self, training_set, split):

		entropy_x = self.calculate_entropy_of_split( training_set, split)
		assert(entropy_x)

		#calculating entropy of the training set
		items_1 =0
		items_0 =0
		items_total = len(training_set)
		for item in training_set:
			if(item[2] ==0):
				items_0 = items_0+1
			else:
				items_1 = items_1 +1
		p_1 = float(items_1) / items_total
		p_0 = float(items_0) / items_total
	
		if (p_0==0) or (p_1==0):
			entropy_y=0
		else:
			entropy_y = -1* (  p_1*log2(p_1) + p_0*log2(p_0))
		
		#Split training set
		[subleft, subright] = self.divide_training_set( training_set, split)

		
		items_1 =0
		items_0 =0
		items_subleft = len(subleft)
		for item in subleft:
			if(item[2] ==0):
				items_0 = items_0+1
			else:
				items_1 = items_1 +1
		p_1 = float(items_1) / items_subleft
		p_0 = float(items_0) / items_subleft
		if (p_0==0) or (p_1==0):
			entropy_y_x_left =0
		else: 
			entropy_y_x_left = -1* (  p_1*log2(p_1) + p_0*log2(p_0))

		p_left = float(items_subleft)/ items_total 

		
		items_1 =0
		items_0 =0
		items_subright = len(subright)
		for item in subright:
			if(item[2] ==0):
				items_0 = items_0+1
			else:
				items_1 = items_1 +1
		p_1 = float(items_1) / items_subright
		p_0 = float(items_0) / items_subright
		if (p_0==0) or (p_1==0):
			entropy_y_x_right =0
		else:
			entropy_y_x_right = -1* (  p_1*log2(p_1) + p_0*log2(p_0))
		p_right = float(items_subright)/items_total

		entropy_y_x = p_right * entropy_y_x_right +  p_left * entropy_y_x_left

		information_gain = entropy_y - entropy_y_x
		
		

		gain_ratio = information_gain/entropy_x

		return gain_ratio
	

	

	def stopping_criteria(self, candidate_split, training_set):
		
		stopping_criteria_l = [1,1] #By default stop with prediction =1
		if(len(training_set) == 0):
			
			return stopping_criteria_l
		else:
			candidate_split_non_zero_entropy = []
			for index in range(0,len(candidate_split)):
				entropy = self.calculate_entropy_of_split( training_set, candidate_split[index])
				if not (entropy ==0):
					candidate_split_non_zero_entropy.append(candidate_split[index])	
			if len(candidate_split_non_zero_entropy) == 0:
				stopping_criteria_l[1] = self.calculate_majority_label(training_set)
				return stopping_criteria_l

			else:
				gain_ratios = []
				non_zero_gain_ratio_found =0
				for split in candidate_split_non_zero_entropy:
					gain_ratio = self.calc_gain_ratio(training_set, split)
					if not (gain_ratio ==0):
						non_zero_gain_ratio_found=1;
					gain_ratios.append([split,gain_ratio])
				if(non_zero_gain_ratio_found==0):
					stopping_criteria_l[1] = self.calculate_majority_label( training_set)
					
					return stopping_criteria_l
		return [0,0] 

	def find_best_split(self, training_set, candidate_split):
			candidate_split_non_zero_entropy = []
			for index in range(0,len(candidate_split)):
				entropy = self.calculate_entropy_of_split(training_set, candidate_split[index])
				if not (entropy ==0):
					candidate_split_non_zero_entropy.append(candidate_split[index])	

			gain_ratios = []
			for split in candidate_split_non_zero_entropy:
				gain_ratio = self.calc_gain_ratio( training_set, split)
				gain_ratios.append([split,gain_ratio])

			max_gain_ratio_index=0
			max_gain_ratio = gain_ratios[0][1]
			for g in range(0, len(gain_ratios)):
				if(max_gain_ratio > gain_ratios[g][1]):
					max_gain_ratio_index = g
					max_gain_ratio = gain_ratios[g][1]
			return gain_ratios[max_gain_ratio_index][0]

			

		
				
				
		
