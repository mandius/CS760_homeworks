from sklearn.inspection import DecisionBoundaryDisplay
import decision_tree
import random
import matplotlib.pyplot as plt
from sklearn import tree as sktree
import numpy as np
import draw_decision_boundary as ddb

fullset = decision_tree.parse_dataset_file("Homework_2_data/Dbig.txt")
training_set = []
for i in range(0, 8192 ):
	entry =fullset.pop(random.randint(0,len(fullset)-1))
	training_set.append(entry)

test_set = fullset

training_set_lengths = [32, 128, 512, 2048, 8192]

fh = open("problem_3.dat", "w")
fh.write("training_set_length, n_nodes, err_n\n")

for length  in training_set_lengths:
	
	training_subset = training_set[0:length-1]

	X =[]
	Y= []
	for i in training_subset:
		X.append([i[0], i[1]])
		Y.append(i[2])

	clf = sktree.DecisionTreeClassifier()
	clf = clf.fit(X,Y)

	correct =0 
	errn =0
	total = len(test_set)
	for index in range(0, len (test_set)):
		T =[]
		T.append(test_set[index][0])
		T.append(test_set[index][1])
		if(clf.predict([T]) == test_set[index][2]):
			correct = correct+1
		else:
			errn = errn+1
	#ddb.draw_decision_boundary(model_function=clf.predict, grid_abs_bound=1)
	fh.write(str(length) +", " + str(clf.tree_.node_count) + ", " + str(errn) + "\n")
fh.close() 
		
	
