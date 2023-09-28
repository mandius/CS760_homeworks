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

fh = open("problem_2_7.dat", "w")
fh.write("training_set_length, n_nodes, err_n\n")

for length  in training_set_lengths:
	tree = decision_tree.simple_tree()
	tree.build_tree(training_set[0:length-1])
	errn = tree.test(test_set)
	n_nodes = tree.n_nodes	
	fh.write(str(length) +", " + str(n_nodes) + ", " + str(errn) + "\n")
	ddb.draw_decision_boundary(model_function=tree.predict, grid_abs_bound=1.5, training_set=training_set[0:length-1])
fh.close()
	

	




