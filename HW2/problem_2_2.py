import decision_tree
import random
import matplotlib.pyplot as plt
from sklearn import tree as sktree
import numpy as np
import draw_decision_boundary as ddb

training_set = decision_tree.parse_dataset_file("Homework_2_data/Dproblem2.txt")

#### Our Decision tree

tree = decision_tree.simple_tree()

tree.build_tree(training_set)
print("\n\n\n  Problem2::: ")
print("Number of Nodes in the tree " + str(tree.n_nodes))

tree.test(training_set)
print(tree.root.gain_ratio_info)

tree.print_tree("Dproblem2_tree.html")

X1_Y0= []
X2_Y0= []
X1_Y1= []
X2_Y1= []
Y1=[]
Y0 = []
blue = (0,0,1)
red = (1,0,0)
for item in training_set:
	if(item[2] ==1):
		X1_Y1.append(item[0])
		X2_Y1.append(item[1])
		Y1.append(blue)	
	else:
		X1_Y0.append(item[0])
		X2_Y0.append(item[1])
		Y0.append(red)	
	
legend_labels = []
plt.scatter(x=X1_Y0, y=X2_Y0, c=Y0, s=10)
legend_labels.append("y=0")

plt.scatter(x=X1_Y1, y=X2_Y1, c=Y1, s=10)
legend_labels.append("y=1")

plt.xlabel("x[0]")
plt.ylabel("x[1]")


plt.show()




