
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.linalg import svd


def demeaned_pca(X,d):
	X_t = X.T
	
	dim_mean = np.zeros((X_t.shape[0], X_t.shape[1]))
	for i in range(X_t.shape[0]):
		mean = np.mean(X_t[i])
		dim_mean[i] = np.repeat(mean, X_t.shape[1])
	X_demeaned = X - dim_mean.T


	X_pca, X_reconstructed_demeaned  = pca(X_demeaned , d)

	
	X_reconstructed = X_reconstructed_demeaned + dim_mean.T

	return X_pca, X_reconstructed

def dro(X, d):
	X_t = X.T
	
	dim_mean = np.zeros((X_t.shape[0], X_t.shape[1]))
	for i in range(X_t.shape[0]):
		mean = np.mean(X_t[i])
		dim_mean[i] = np.repeat(mean, X_t.shape[1])
	Y = X - dim_mean.T
	U, S, Vt = svd(Y, full_matrices=True)
	Sd = np.diag(S)

	At = Sd[:d, :d] @ Vt[:d]
	b = dim_mean.T[0]
	

	X_pca =  np.zeros((X.shape[0],d))

	
	for i in range(X_pca.shape[0]):
		X_pca[i] =  (np.linalg.inv(At@At.T) @ At @((X[i]-b).reshape(X.shape[1],1))).T

	

	X_reconstructed = X_pca@At + dim_mean.T

	return X_pca, X_reconstructed	


def normalised_pca(X,d):
	X_t = X.T
	
	dim_mean = np.zeros((X_t.shape[0], X_t.shape[1]))
	for i in range(X_t.shape[0]):
		mean = np.mean(X_t[i])
		dim_mean[i] = np.repeat(mean, X_t.shape[1])
	X_demeaned = X - dim_mean.T

	dim_std = np.zeros((X_t.shape[0], X_t.shape[1]))
	for i in range(X_t.shape[0]):
		std = np.std(X_t[i])
		dim_std[i] = np.repeat(std, X_t.shape[1])
	
	X_normalised = X_demeaned / dim_std.T
	

	X_pca, X_reconstructed_normalised  = pca(X_normalised , d)

	
	X_reconstructed = X_reconstructed_normalised * dim_std.T + dim_mean.T

	return X_pca, X_reconstructed



def pca (X, d):
	print(X.shape)
	U, S, Vt = svd(X, full_matrices=True)
	X_pca = X.dot(Vt[:d].T)
	X_reconstructed = X_pca.dot(Vt[:d])
	


	return X_pca, X_reconstructed


def calculate_for_file(filename, D, d, plot=0):
	fh = open(filename, "r")
	lines = fh.readlines()

	n = len(lines)
	X = np.zeros((n,D), dtype=float)
	
	for i  in range(n):
		temp = lines[i].strip().split(",")
		for j in range(len(temp)):
			X[i, j] = float(temp[j].strip())
	
	X_pca, X_reconstructed  = pca(X, d)
	
	reconstruction_error =0
	print(X[0].shape)
	for i in range(X.shape[0]):
		
		reconstruction_error = reconstruction_error + np.linalg.norm(X[i]-X_reconstructed[i]) * np.linalg.norm(X[i]-X_reconstructed[i]) 

	print("Reconstruction Error for Buggy PCA:")
	print(reconstruction_error/X.shape[0])

	if(plot):
		X_n = []
		Y_n = []
		X_r = []
		Y_r = []
		for i in range(X.shape[0]):
			X_n.append(X[i][0])
			Y_n.append(X[i][1])
			X_r.append(X_reconstructed[i][0])
			Y_r.append(X_reconstructed[i][1])

		plt.scatter(X_n, Y_n, c="blue", marker="o", s=2, label = "Original Points")
		plt.scatter(X_r, Y_r, c="red", marker="x", s=2, label = "Reconstructed Points")
		plt.legend(["Original Points", "Reconstructed Points"])	
		plt.xlim(0,10)
		plt.ylim(0,10)	
		plt.show()

	X_pca, X_reconstructed  = demeaned_pca(X, d)
	
	reconstruction_error =0
	print(X[0].shape)
	for i in range(X.shape[0]):
		
		reconstruction_error = reconstruction_error + np.linalg.norm(X[i]-X_reconstructed[i]) * np.linalg.norm(X[i]-X_reconstructed[i]) 


	print("Reconstruction Error for demeaned PCA:")
	print(reconstruction_error/X.shape[0])

	if(plot):
		X_n = []
		Y_n = []
		X_r = []
		Y_r = []
		for i in range(X.shape[0]):
			X_n.append(X[i][0])
			Y_n.append(X[i][1])
			X_r.append(X_reconstructed[i][0])
			Y_r.append(X_reconstructed[i][1])

		plt.scatter(X_n, Y_n, c="blue", marker="o", s=2, label = "Original Points")
		plt.scatter(X_r, Y_r, c="red", marker="x", s=2, label = "Reconstructed Points")
		plt.legend(["Original Points", "Reconstructed Points"])	
		plt.xlim(0,10)
		plt.ylim(0,10)	
		plt.show()	

	X_pca, X_reconstructed  = normalised_pca(X, d)
	
	reconstruction_error =0
	print(X[0].shape)
	for i in range(X.shape[0]):
		
		reconstruction_error = reconstruction_error + np.linalg.norm(X[i]-X_reconstructed[i]) * np.linalg.norm(X[i]-X_reconstructed[i]) 

	print("Reconstruction Error for normalized PCA:")
	print(reconstruction_error/X.shape[0])

	if(plot):
		X_n = []
		Y_n = []
		X_r = []
		Y_r = []
		for i in range(X.shape[0]):
			X_n.append(X[i][0])
			Y_n.append(X[i][1])
			X_r.append(X_reconstructed[i][0])
			Y_r.append(X_reconstructed[i][1])

		plt.scatter(X_n, Y_n, c="blue", marker="o", s=2, label="Original Points")
		plt.scatter(X_r, Y_r, c="red", marker="x", s=2, label = "Reconstructed Points")
		plt.legend(["Original Points", "Reconstructed Points"])	
		plt.xlim(0,10)
		plt.ylim(0,10)	
		plt.show()	


	X_pca, X_reconstructed  = dro(X, d)
	
	reconstruction_error =0
	print(X[0].shape)
	for i in range(X.shape[0]):
		
		reconstruction_error = reconstruction_error + np.linalg.norm(X[i]-X_reconstructed[i]) * np.linalg.norm(X[i]-X_reconstructed[i]) 

	print("Reconstruction Error for DRO:")
	print(reconstruction_error/X.shape[0])

	if(plot):
		X_n = []
		Y_n = []
		X_r = []
		Y_r = []
		for i in range(X.shape[0]):
			X_n.append(X[i][0])
			Y_n.append(X[i][1])
			X_r.append(X_reconstructed[i][0])
			Y_r.append(X_reconstructed[i][1])

		plt.scatter(X_n, Y_n, c="blue", marker="o", s=2, label="Original Points")
		plt.scatter(X_r, Y_r, c="red", marker="x", s=2, label = "Reconstructed Points")
		plt.legend(["Original Points", "Reconstructed Points"])	
		plt.xlim(0,10)
		plt.ylim(0,10)	
		plt.show()	



calculate_for_file("data/data2D.csv", 2, 1, plot=1)
#calculate_for_file("data/data1000D.csv", 1000, 30)


