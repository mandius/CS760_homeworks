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

length_base_2 = [5, 7, 9, 11, 13]
errn =[]


for length  in training_set_lengths:
	tree = decision_tree.simple_tree()
	tree.build_tree(training_set[0:length-1])
	err=tree.test(test_set)
	errn.append(err)
	n_nodes = tree.n_nodes	
	print(str(length) +", " + str(n_nodes) + ", " + str(err) + "\n")
	ddb.draw_decision_boundary("D" + str(length)+ " Decision Boundary" ,model_function=tree.predict, grid_abs_max=1.5,grid_abs_min=-1.5, training_set=training_set[0:length-1])


plt.plot(length_base_2, errn)
plt.xlabel("log2(n)")
plt.ylabel("errn")
plt.show()
	

	




