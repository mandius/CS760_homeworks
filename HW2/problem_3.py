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
trainin_set_lengths_log2 = [5, 7, 9, 11, 13]
errn_list = []

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
	errn_list.append(errn)
	#ddb.draw_decision_boundary(model_function=clf.predict, grid_abs_bound=1)
	print(str(length) +", " + str(clf.tree_.node_count) + ", " + str(errn) + "\n")


plt.plot(trainin_set_lengths_log2, errn_list)
plt.xlabel("log2(n)")
plt.ylabel("errn")
plt.show()
	
