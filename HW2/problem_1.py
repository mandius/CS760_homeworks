import decision_tree
import random
import matplotlib.pyplot as plt
from sklearn import tree as sktree
import numpy as np

training_set = decision_tree.parse_dataset_file("Homework_2_data/D1.txt")

#### Our Decision tree
fh= open("self_tree.txt", "w")
tree = decision_tree.simple_tree()

tree.build_tree(training_set)
print("\n\n\n  Self Tree::: ")
print("Number of Nodes in the tree " + str(tree.n_nodes))
tree.print_tree(fh)
tree.test(training_set)

fh.close()

####Scikit Decision tree
fh = open("scikit_tree.txt", "w")

X =[]
Y= []
for i in training_set:
	X.append([i[0], i[1]])
	Y.append(i[2])

clf = sktree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

correct =0 
err =0
total = len(training_set)
for index in range(0, len (X)):
	T =[]
	T.append(X[index][0])
	T.append(X[index][1])
	if(clf.predict([T]) == Y[index]):
		correct = correct+1
	else:
		err = err+1
text_data = sktree.export_text(clf, feature_names=["x1","x2"], class_names=["0","1"])
fh.write(text_data)


print("\n\n\n  Scikit Tree::: ")
print("Number of Nodes in the tree: " + str(clf.tree_.node_count))
print("Total Items: " + str(total) + " Error: " + str(err) + " Correct: " + str(correct))
print("Error percentage: " + str((float(err)/total)*100))
fh.close()
	

