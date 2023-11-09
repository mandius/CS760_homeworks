import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.linalg import svd

def dro(X):
	X_t = X.T
	
	dim_mean = np.zeros((X_t.shape[0], X_t.shape[1]))
	for i in range(X_t.shape[0]):
		mean = np.mean(X_t[i])
		dim_mean[i] = np.repeat(mean, X_t.shape[1])
	Y = X - dim_mean.T
	U, S, Vt = svd(Y, full_matrices=True)
	#print(S)
	x_axis = range(S.shape[0])
	plt.plot(x_axis, S, marker="+", markersize=2)
	plt.xlabel("Singular Value Number")
	plt.ylabel("Singular Value")
	plt.show()




fh = open("data/data1000D.csv", "r")
lines = fh.readlines()
D =1000
n = len(lines)
print(n,D)
X = np.zeros((n,D), dtype=float)

for i  in range(n):
	temp = lines[i].strip().split(",")
	for j in range(len(temp)):
		X[i, j] = float(temp[j].strip())


dro(X)





	
