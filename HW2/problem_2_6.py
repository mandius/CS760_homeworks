import decision_tree
import random
import matplotlib.pyplot as plt
from sklearn import tree as sktree
import numpy as np
import draw_decision_boundary as ddb

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

ddb.draw_decision_boundary(model_function=tree.predict, grid_abs_max=10.0, grid_abs_min=0, training_set=training_set)

training_set = decision_tree.parse_dataset_file("Homework_2_data/D2.txt")

#### Our Decision tree
fh= open("self_tree.txt", "w")
tree = decision_tree.simple_tree()

tree.build_tree(training_set)
print("\n\n\n  Self Tree::: ")
print("Number of Nodes in the tree " + str(tree.n_nodes))
tree.print_tree(fh)
tree.test(training_set)

ddb.draw_decision_boundary(model_function=tree.predict, grid_abs_max=1.0 , grid_abs_min=0 , training_set=training_set)

fh.close()

