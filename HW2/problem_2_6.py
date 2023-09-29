import decision_tree
import random
import matplotlib.pyplot as plt
from sklearn import tree as sktree
import numpy as np
import draw_decision_boundary as ddb

training_set = decision_tree.parse_dataset_file("Homework_2_data/D1.txt")

#### Our Decision tree

tree = decision_tree.simple_tree()

tree.build_tree(training_set)
print("\n\n\n  D1::: ")
print("Number of Nodes in the tree " + str(tree.n_nodes))

tree.test(training_set)


ddb.draw_decision_boundary("D1_decision_boundary", model_function=tree.predict, grid_abs_max=1.0, grid_abs_min=0, training_set=training_set)

training_set = decision_tree.parse_dataset_file("Homework_2_data/D2.txt")

#### Our Decision tree

tree = decision_tree.simple_tree()

tree.build_tree(training_set)
print("\n\n\n  D2::: ")
print("Number of Nodes in the tree " + str(tree.n_nodes))

tree.test(training_set)

ddb.draw_decision_boundary("D2_decision_boundary", model_function=tree.predict, grid_abs_max=1.0 , grid_abs_min=0 , training_set=training_set)


