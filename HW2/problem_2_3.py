import decision_tree
import random
import matplotlib.pyplot as plt
from sklearn import tree as sktree
import numpy as np
import draw_decision_boundary as ddb

training_set = decision_tree.parse_dataset_file("Homework_2_data/Druns.txt")

#### Our Decision tree
tree = decision_tree.simple_tree()

tree.build_tree(training_set)

print (tree.root.candidate_splits)
print (tree.root.gain_ratio_info)




