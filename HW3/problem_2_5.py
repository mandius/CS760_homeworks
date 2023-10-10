import sys
import os
import matplotlib.pyplot as plt
import math 
import numpy as np
from sklearn import metrics

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

def euclidean_distance(a, b):
    a=np.array(a)
    b=np.array(b)
    return np.linalg.norm(a-b)



def knn_predictor(k, f_size, test_point, training_data):
	curr_min_distance_entry = []
	nearest_neighbours = []
	distances = []
	for j in training_data:
		distances.append([euclidean_distance(test_point, j), j[f_size]])

	distances = sorted(distances , key=lambda x: x[0])


	nearest_neighbours = distances[0:k]


	vote0 =0 
	vote1= 0
	for nn in nearest_neighbours:
		if nn[1] ==0:
			vote0 = vote0+1
		else:
			vote1 = vote1+1
	return (float(vote1)/k)






class LogisticRegression():
    def __init__(self, alpha, iterations):
        self.alpha = alpha
        self.iterations = iterations
        

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
       
        return P










			
		
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


test_set = partitions[4]
training_set = partitions[0] + partitions[1] + partitions[2] + partitions[3] 



tp=0
tn=0
fp=0
fn=0
precision=0
recall=0


FPR1 =[]
TPR1 = []

check_set = []
confidence_knn = []

for k in test_set:
	temp = k[0:3000] 
	temp.append(0)
	#print(len(temp))
	check_set.append(temp)


for l in check_set:
	confidence_knn.append(knn_predictor(5, 3000, l,  training_set))

true_y_knn = []
for i in test_set:
	true_y_knn.append(i[3000])


FPR1, TPR1, thresholds = metrics.roc_curve(true_y_knn, confidence_knn)


print("For KNN")
print(metrics.roc_auc_score(true_y_knn, confidence_knn))

check_set = []

true_y_log_reg=[]
for k in test_set:
	temp = k[0:3000] 
	temp.append(0)
	#print(len(temp))
	true_y_log_reg.append(k[3000])
	check_set.append(temp)

log_reg = LogisticRegression(0.5, 5000)

#build the predictor:
x_train = np.zeros((len(training_set), 3000))
y_train = np.zeros(len(training_set))

for tr in range (0, len(training_set)):
	x_train[tr] = np.array(training_set[tr][0:3000])
	y_train[tr] = np.array(training_set[tr][3000])

log_reg.fit(x_train, y_train)
print("Training Done")


threshold =0 
FPR2 =[]
TPR2 = []

confidence_log_reg = []

	
for l in check_set:
	confidence_log_reg.append(log_reg.predict(np.array(l[0:3000])))



FPR2, TPR2, thresholds =  metrics.roc_curve(true_y_log_reg, confidence_log_reg)
print("For Log Reg")
print(metrics.roc_auc_score(true_y_log_reg, confidence_log_reg))




plt.plot(FPR1.tolist(), TPR1.tolist())
plt.plot(FPR2.tolist(), TPR2.tolist())
plt.xlabel("False Positive Rate", fontsize=10) # set x-axis label
plt.ylabel("True Positive Rate", fontsize=10) # set y-axis label

legends_label = []
legends_label.append("KNN with K=5")
legends_label.append("Logistic Regression")
plt.legend(legends_label)
plt.show()
	
