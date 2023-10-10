import sys
import os
import matplotlib.pyplot as plt
import math 
import numpy as np

class LogisticRegression():
    def __init__(self, alpha, iterations, threshold):
        self.alpha = alpha
        self.iterations = iterations
        self.threshold = threshold

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        for i in range(self.iterations):
            self.weights_update()
        return self

    def weights_update(self):
        A = 1 / (1 + np.exp(- (self.X.dot(self.W) + self.b)))

        temp = (A - self.Y.T)
        temp = np.reshape(temp, self.m)
        dW = np.dot(self.X.T, temp) / self.m
        db = np.sum(temp) / self.m

        self.W = self.W - self.alpha * dW
        self.b = self.b - self.alpha * db

        return self

    def predict(self, X):
        P = 1 / (1 + np.exp(- (X.dot(self.W) + self.b)))
        Y = np.where(P > self.threshold, 1, 0)
        return Y

			
		


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


	

log_reg = LogisticRegression(0.5, 5000, 0.5)		
		
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

for i in range(0,5):
	test_set = partitions[i]
	training_set =[]
	tp=0
	tn=0
	fp=0
	fn=0
	accuracy =0
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
	


	#build the predictor:
	x_train = np.zeros((len(training_set), 3000))
	y_train = np.zeros(len(training_set))

	for tr in range (0, len(training_set)):
		x_train[tr] = np.array(training_set[tr][0:3000])
		y_train[tr] = np.array(training_set[tr][3000])

	log_reg.fit(x_train, y_train)
	print("Training Done")
	
	

	for l in check_set:
		l[3000] = log_reg.predict(np.array(l[0:3000]))

	for m in range(0, len(check_set)):
		if((test_set[m][3000]==0) and (check_set[m][3000]==0)):
			tn = tn+1
		elif ((test_set[m][3000]==0) and (check_set[m][3000]==1)):
			fp = fp+1
		elif ((test_set[m][3000]==1) and (check_set[m][3000]==0)):
			fn = fn+1
		elif ((test_set[m][3000]==1) and (check_set[m][3000]==1)):
			tp = tp+1

			
	accuracy = float(tp + tn)/float(tp+tn+fp+fn)
	recall = float(tp)/float(tp+fn)
	precision = float(tp)/float(tp+fp)

	print("Fold " +  str(i) + ": "+ "Accuracy = "+ str(accuracy) + " Recall = "+ str(recall) + " Precision = "+ str(precision))
	






