import numpy as np
import random
import matplotlib.pyplot as plt
import math
import sys
from scipy.stats  import multivariate_normal

def distance(p1, p2):
	return np.linalg.norm(p1 -p2)

def kmeans(dataset1, dataset2, dataset3, mean1, mean2, mean3):
	dataset = np.concatenate((dataset1, dataset2))
	dataset = np.concatenate((dataset, dataset3))
	np.random.shuffle(dataset)
	num_restarts = 20
	objectives = []
	correct_percentage = []
	for restart in range(0, num_restarts):
		#randomly assign the cluster centres:
		cluster_centres = []
		num_rows = dataset.shape[0]
		cluster_centre_indices = np.random.choice(num_rows, size=3)
		cluster_centres.append(dataset[cluster_centre_indices[0]])
		cluster_centres.append(dataset[cluster_centre_indices[1]])
		cluster_centres.append(dataset[cluster_centre_indices[2]])
		
	
		#start forming the clusters:
		objective = sys.maxsize *2 
		last_objective = sys.maxsize * 2 + 1
	
		while(last_objective - objective > 0.1):
				
			last_objective = objective
			clusters = [[], [], []]
	
			for point in dataset:
				l2_norms =  [distance(point, cluster_centres[0]), distance(point, cluster_centres[1]), distance(point, cluster_centres[2])]
				cluster_number = l2_norms.index(min(l2_norms))
				clusters[cluster_number].append(point)
	
			for i in range(0, len(clusters)):
				cluster_centres[i] = np.mean(clusters[i], axis=0)
	
			objective =0
			for i in range(0, len(clusters)):
				for point in clusters[i]:
					objective = objective + np.square(distance(point, cluster_centres[i]))
	
			cluster_numbers =[0,1,2]
			cluster_numbers_new = []
			#find the closest mapping for first initial cluster:
			l2_norms = []
			for i in cluster_numbers:
				l2_norms.append(distance(mean1, cluster_centres[i]))
		
			cluster_number = cluster_numbers[l2_norms.index(min(l2_norms))]
			cluster_numbers_new.append(cluster_number)
			cluster_numbers.remove(cluster_number)
	
			l2_norms = []
			for i in cluster_numbers:
				l2_norms.append(distance(mean2, cluster_centres[i]))
		
			cluster_number = cluster_numbers[l2_norms.index(min(l2_norms))]
			cluster_numbers_new.append(cluster_number)
			cluster_numbers.remove(cluster_number)
	
			
			l2_norms = []
			for i in cluster_numbers:
				l2_norms.append(distance(mean3, cluster_centres[i]))
		
			cluster_number = cluster_numbers[l2_norms.index(min(l2_norms))]
			cluster_numbers_new.append(cluster_number)
			cluster_numbers.remove(cluster_number)
	
	
			correct = 0
		
			#print(cluster_numbers_new)
			#print(type(clusters))
			#print(type(clusters[0]))
			#print(type(clusters[0][0]))
			for point in dataset1:
				for point2 in clusters[cluster_numbers_new[0]]:
					if(np.all(point == point2, axis=0)):
						correct =  correct+1
			
			for point in dataset2:
				for point2 in clusters[cluster_numbers_new[1]]:
					if(np.all(point == point2, axis=0)):
						correct =  correct+1
	
			for point in dataset3:
				for point2 in clusters[cluster_numbers_new[2]]:
					if(np.all(point == point2, axis=0)):
						correct =  correct+1	
	
		#print(objective, (correct/len(dataset))*100)
		objectives.append(objective)
		correct_percentage.append((correct/len(dataset))*100)
			
	min_restart_index = objectives.index(min(objectives))
	return[objectives[min_restart_index] , correct_percentage[min_restart_index]]	

def gmm(dataset1, dataset2, dataset3, mean1, mean2, mean3):
	dataset = np.concatenate((dataset1, dataset2))
	dataset = np.concatenate((dataset, dataset3))


	num_restarts = 20
	objectives = []
	correct_percentage = []
	for restart in range(0, num_restarts):
		np.random.shuffle(dataset)
		
		#Randomly initialize the base parameters.
		means = []
		means.append(np.mean(dataset[0:100], axis=0))
		means.append(np.mean(dataset[100:200], axis=0))
		means.append(np.mean(dataset[200:300], axis=0))
		#print(means)
		covariances = []
		covariances.append(np.array([[1,0],[0,1]]))
		covariances.append(np.array([[1,0],[0,1]]))
		covariances.append(np.array([[1,0],[0,1]]))
	
		theta = []
		theta.append(float(1)/float(3))
		theta.append(float(1)/float(3))
		theta.append(float(1)/float(3))
	
		#start forming the clusters:
		objective =  (sys.maxsize * 2  )
		last_objective = (sys.maxsize * 2 +1  )
	
		while(last_objective - objective  > 0.1):
			last_objective = objective
			distribution1 = multivariate_normal(means[0], covariances[0])
			distribution2 = multivariate_normal(means[1], covariances[1])
			distribution3 = multivariate_normal(means[2], covariances[2])
	
			posterior = np.repeat(np.array([np.zeros(dataset.shape[0], dtype=float)]), 3, axis = 0)
			#print(posterior.shape)
			# E Step
			for i in range(0, dataset.shape[0]):
				point = dataset[i]
				pdf1 = distribution1.pdf(point)
				pdf2 = distribution2.pdf(point)
				pdf3 = distribution3.pdf(point)
	
				denom  = pdf1 * theta[0] + pdf2 * theta[1] + pdf3 * theta[2]
				w1 = (pdf1*theta[0])/denom
				w2 = (pdf2*theta[1])/denom
				w3 = (pdf3*theta[2])/denom
	
				posterior[0][i] = float(w1)
				posterior[1][i] = float(w2)
				posterior[2][i] = float(w3)
	
			#M Setp:
			theta[0] = np.mean(posterior[0])
			theta[1] = np.mean(posterior[1])
			theta[2] = np.mean(posterior[2])
	
	
			new_posterior = np.repeat(posterior[0].reshape(1,300),2, axis=0).T
			means[0] = np.sum(np.multiply(new_posterior, dataset), axis=0)/ np.sum(posterior[0])
		
			new_posterior = np.repeat(posterior[1].reshape(1,300),2, axis=0).T	
			means[1] = np.sum(np.multiply(new_posterior, dataset), axis=0)/ np.sum(posterior[1])
	
			new_posterior = np.repeat(posterior[2].reshape(1,300),2, axis=0).T				
			means[2] = np.sum(np.multiply(new_posterior, dataset), axis=0)/ np.sum(posterior[2])
			
			
		
			means_array = np.repeat(means[0].reshape(1,2), len(dataset), axis = 0)
			diff_array = dataset - means_array
			#print(diff_array.shape)
			new_posterior = np.repeat(posterior[0].reshape(1,300),2, axis=0).T
			diff_array_scaled = new_posterior *diff_array
			numerator = diff_array_scaled.T @ diff_array
			denominator = np.sum(posterior[0])
	
			covariances[0] = numerator/denominator	
	
			
			means_array = np.repeat(means[1].reshape(1,2), len(dataset), axis = 0)
			diff_array = dataset - means_array
			#print(diff_array.shape)
			new_posterior = np.repeat(posterior[1].reshape(1,300),2, axis=0).T
			diff_array_scaled = new_posterior *diff_array
			numerator = diff_array_scaled.T @ diff_array
			denominator = np.sum(posterior[1])
	
			covariances[1] = numerator/denominator	
	
			means_array = np.repeat(means[2].reshape(1,2), len(dataset), axis = 0)
			diff_array = dataset - means_array
			#print(diff_array.shape)
			new_posterior = np.repeat(posterior[2].reshape(1,300),2, axis=0).T
			diff_array_scaled = new_posterior *diff_array
			numerator = diff_array_scaled.T @ diff_array
			denominator = np.sum(posterior[2])
	
			covariances[2] = numerator/denominator	
		
	
			distributions =[ ]
			#Recompute the paramters
			distributions.append(multivariate_normal(means[0], covariances[0]))
			distributions.append(multivariate_normal(means[1], covariances[1]))
			distributions.append(multivariate_normal(means[2], covariances[2]))
			
	
			objective =0
	
			for point in dataset:
				pdf1 = distributions[0].pdf(point)
				pdf2 = distributions[1].pdf(point)
				pdf3 = distributions[2].pdf(point)
	
				objective = objective + np.log(pdf1*theta[0] +pdf2*theta[1] + pdf3*theta[2])

			objective = -1 * objective
			cluster_numbers =[0,1,2]
			cluster_numbers_new = []
			#find the closest mapping for first initial cluster:
			l2_norms = []
			for i in cluster_numbers:
				l2_norms.append(distance(mean1, means[i]))
			
			cluster_number = cluster_numbers[l2_norms.index(min(l2_norms))]
			cluster_numbers_new.append(cluster_number)
			cluster_numbers.remove(cluster_number)
		
			l2_norms = []
			for i in cluster_numbers:
				l2_norms.append(distance(mean2, means[i]))
			
			cluster_number = cluster_numbers[l2_norms.index(min(l2_norms))]
			cluster_numbers_new.append(cluster_number)
			cluster_numbers.remove(cluster_number)
		
			
			l2_norms = []
			for i in cluster_numbers:
				l2_norms.append(distance(mean3, means[i]))
			
			cluster_number = cluster_numbers[l2_norms.index(min(l2_norms))]
			cluster_numbers_new.append(cluster_number)
			cluster_numbers.remove(cluster_number)
		
		
			correct = 0
			
			#print(cluster_numbers_new)
			#print(type(clusters))
			#print(type(clusters[0]))
			#print(type(clusters[0][0]))
	
			for point in dataset:
				pdfs = [distributions[cluster_numbers_new[0]].pdf(point), distributions[cluster_numbers_new[1]].pdf(point), distributions[cluster_numbers_new[2]].pdf(point)]
				index = pdfs.index(max(pdfs))
	
				if index == 0:
					for point2 in dataset1:
						if(np.all(point ==point2, axis=0)):
							correct = correct+1 
				elif index == 1:
					for point2 in dataset2:
						if(np.all(point ==point2, axis=0)):
							correct = correct+1
	
				elif  index==2:
					for point2 in dataset3:
						if(np.all(point ==point2, axis=0)):
							correct = correct+1
			#print(last_objective, objective, (correct/len(dataset))*100)
			objectives.append(objective)
			correct_percentage.append((correct/len(dataset))*100)

			 	
	min_restart_index = objectives.index(min(objectives))
	return [objectives[min_restart_index] , correct_percentage[min_restart_index]]					
			

		
		
				

sigma_values = [0.5, 1, 2, 4, 8]


gmm_objective = []
kmeans_objective = []
gmm_accuracy = [] 
kmeans_accuracy = []
for sigma in sigma_values:
	mean1 = np.array([-1,-1])
	covariance1 = sigma * np.array([[2,0.5], [0.5,1]])

	mean2 = np.array([1,-1])
	covariance2 = sigma * np.array([[1,-0.5], [-0.5,2]])

	mean3 = np.array([0,1])
	covariance3 = sigma * np.array([[1,0], [0,2]])


	dataset1 =  np.random.multivariate_normal(mean1, covariance1, 100)
	dataset2 =  np.random.multivariate_normal(mean2, covariance2, 100)
	dataset3 =  np.random.multivariate_normal(mean3, covariance3, 100)
	result1 = kmeans(dataset1, dataset2, dataset3, mean1, mean2, mean3)
	print(result1)

	kmeans_objective.append(result1[0])
	kmeans_accuracy.append(result1[1])
	result2 = gmm(dataset1, dataset2, dataset3, mean1, mean2, mean3)
	print(result2)
	
	gmm_objective.append(result2[0])
	gmm_accuracy.append(result2[1])


print(gmm_objective)
print(kmeans_objective)
print(gmm_accuracy) 
print(kmeans_accuracy)



plt.plot(sigma_values, kmeans_objective)
plt.xlabel("sigma")
plt.ylabel("objective")
plt.title("Kmeans Objective vs sigma")
plt.show()

plt.plot(sigma_values, kmeans_accuracy)
plt.xlabel("sigma")
plt.ylabel("accuracy")
plt.title("Kmeans Accuracy vs sigma")
plt.show()
	
	
plt.plot(sigma_values, gmm_objective)
plt.xlabel("sigma")
plt.ylabel("objective")
plt.title("GMM Objective vs sigma")
plt.show()

plt.plot(sigma_values, gmm_accuracy)
plt.xlabel("sigma")
plt.ylabel("accuracy")
plt.title("GMM Accuracy vs sigma")
plt.show()

