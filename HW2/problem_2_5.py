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


tree.print_tree("D1_tree.html")


training_set = decision_tree.parse_dataset_file("Homework_2_data/D2.txt")

#### Our Decision tree
tree = decision_tree.simple_tree()

tree.build_tree(training_set)


tree.print_tree("D2_tree.html")




