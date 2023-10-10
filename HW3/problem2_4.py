import sys
import os
import matplotlib.pyplot as plt
import math 
import numpy as np

def parse_dataset_file(filename):
	fh = open(filename, "r")
	entries = fh.readlines()
	final_list= []
	
	for entry in range(1, len(entries)):
		entry_list = entries[entry].split(",")
		entry_list_float =[]
		for i in range(1,len(entry_list)):
			entry_list_float.append(float(entry_list[i]))
		final_list.append(entry_list_float)
#	print(len(final_list))
	return final_list


def euclidean_distance(a, b, f_size):
    a=np.array(a[0:f_size])
    b=np.array(b[0:f_size])
    return np.linalg.norm(a-b)		 



def knn_predictor(k, f_size, test_point, training_data):
	curr_min_distance_entry = []
	nearest_neighbours = []
	distances = []
	for j in training_data:
		distances.append([euclidean_distance(test_point, j, f_size), j[f_size]])

	distances = sorted(distances , key=lambda x: x[0])


	nearest_neighbours = distances[0:k]


	vote0 =0 
	vote1= 0
	for nn in nearest_neighbours:
		if nn[1] ==0:
			vote0 = vote0+1
		else:
			vote1 = vote1+1
	if(vote0>vote1):
		test_point[f_size]=0
	else:
		test_point[f_size]=1

			
		
data = parse_dataset_file("data/emails.csv")
#print(len(data))
#print(len(data[552]))

#normalize_data_set(data)

partition_size =1000

partitions = [[],[],[],[],[]]
partitions[0] = data[0:partition_size]
partitions[1] = data[partition_size: 2*partition_size]
partitions[2] = data[2*partition_size: 3*partition_size]
partitions[3] = data[3*partition_size: 4*partition_size]
partitions[4] = data[4*partition_size: 5*partition_size]
y_a = []

for knn in [1,3,5,7,10]:
	accuracy =0
	for i in range(0,5):
		test_set = partitions[i]
		training_set =[]
		tp=0
		tn=0
		fp=0
		fn=0
		precision=0
		recall=0
	
		for j in range(0,5):
			if not j==i:
				training_set = training_set + partitions[j]
	
	
		check_set = []
	
		for k in test_set:
			temp = k[0:3000] 
			temp.append(0)
			#print(len(temp))
			check_set.append(temp)
		
	
		for l in check_set:
			knn_predictor(knn, 3000, l, training_set)
	
		for m in range(0, len(check_set)):
			if((test_set[m][3000]==0) and (check_set[m][3000]==0)):
				tn = tn+1
			elif ((test_set[m][3000]==0) and (check_set[m][3000]==1)):
				fp = fp+1
			elif ((test_set[m][3000]==1) and (check_set[m][3000]==0)):
				fn = fn+1
			elif ((test_set[m][3000]==1) and (check_set[m][3000]==1)):
				tp = tp+1
	
				
		accuracy = accuracy + float(tp + tn)/float(tp+tn+fp+fn)
	accuracy = (accuracy/5)*100
	y_a.append(accuracy)
	print("k value = "+ str(knn) + "  Accuracy = "+ str(accuracy))

x_a = [1,3,5,7,10]



plt.plot(x_a, y_a)
plt.xlabel("K", fontsize=10) # set x-axis label
plt.ylabel("Accuracy(%)", fontsize=10) # set y-axis label
plt.show()
	






