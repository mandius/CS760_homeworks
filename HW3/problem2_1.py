import sys
import os
import matplotlib.pyplot as plt
import math 
import numpy as np

def parse_dataset_file(filename):
	fh = open(filename, "r")
	entries = fh.readlines()
	final_list= []
	
	for entry in entries:
		entry_list = entry.split(" ")
		entry_list_float =[float(entry_list[0]),float(entry_list[1]),int(entry_list[2])]
		final_list.append(entry_list_float)
	return final_list

def euclidian_distance(p1, p2, f_size):
	dist = 0.0
	for i in range(0,f_size):
		dist = dist +   (p1[i] - p2[i])*(p1[i] - p2[i])
	return math.sqrt(dist)
	

def knn_predictor(k, f_size, test_point, training_data):
	last_min_distance = -sys.maxsize -1
	curr_min_distance = sys.maxsize
	curr_min_distance_entry = []
	nearest_neighbours = []
	for i in range(0,k):
		curr_min_distance = sys.maxsize
		for j in training_data:
			distance = euclidian_distance(test_point, j, f_size)
			if( (distance >= last_min_distance) and (distance < curr_min_distance)):
				curr_min_distance = distance
				curr_min_distance_entry =  j
		nearest_neighbours.append(curr_min_distance_entry)
		last_min_distance = curr_min_distance
	vote0 =0 
	vote1= 0
	for nn in nearest_neighbours:
		if nn[f_size] ==0:
			vote0 = vote0+1
		else:
			vote1 = vote1+1
	if(vote0>vote1):
		test_point[f_size] =0
	else:
		test_point[f_size]=1

			
		
		


training_data = parse_dataset_file("data/D2z.txt")

feature1_subarray = np.arange(-2.0, 2.1, 0.1)
feature2_subarray = np.arange(-2.0, 2.1, 0.1)
test_data = []


for i in feature1_subarray:
	for j in feature2_subarray:
		temp = [i, j, 0]
		test_data.append(temp)


for i in test_data:
	knn_predictor(3, 2, i, training_data)
	
red = (1,0,0)
blue = (0,0,1)

test_data_x10_points = []
test_data_x20_points = []
test_data_x11_points = []
test_data_x21_points = []
test_data_y0_points = []
test_data_y1_points = []


for i in test_data:
	if(i[2]==1):
		test_data_x11_points.append(i[0])
		test_data_x21_points.append(i[1])
		test_data_y1_points.append(blue)
	else:
		test_data_x10_points.append(i[0])
		test_data_x20_points.append(i[1])
		test_data_y0_points.append(red)

legend_labels = []
plt.scatter(x=test_data_x10_points, y=test_data_x20_points, c=test_data_y0_points, s=2)
legend_labels.append("test_set predicted=0")
plt.scatter(x=test_data_x11_points, y=test_data_x21_points, c=test_data_y1_points, s=2)
legend_labels.append("test_set predicted=1")
plt.legend(legend_labels)


training_data_x10_points = []
training_data_x20_points = []
training_data_x11_points = []
training_data_x21_points = []
training_data_y0_points = []
training_data_y1_points = []


for i in training_data:
	if(i[2]==1):
		training_data_x11_points.append(i[0])
		training_data_x21_points.append(i[1])
		training_data_y1_points.append(blue)
	else:
		training_data_x10_points.append(i[0])
		training_data_x20_points.append(i[1])
		training_data_y0_points.append(red)

plt.scatter(x=training_data_x10_points, y=training_data_x20_points, c=training_data_y0_points, marker="1")
legend_labels.append("training_set =0")
plt.scatter(x=training_data_x11_points, y=training_data_x21_points, c=training_data_y1_points, marker="+")
legend_labels.append("training_set =1")
plt.legend(legend_labels)







plt.show()




	






