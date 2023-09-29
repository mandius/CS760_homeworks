from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import math
import random
import numpy as np
import matplotlib.pyplot as plt


def cal_log_mse (poly, test_set):
	y_val= []
	for i in range(0,len(test_set)):
		y_val.append(Polynomial(poly.coef[::-1])(test_set[i]))

	err_n = []
	for i in range(0,len(test_set)):
		err_n.append(math.sin(test_set[i]) - y_val[i])

	sum_err = 0
	for i in err_n:
		sum_err = sum_err + i*i
	sum_err = sum_err/len(test_set)

	return math.log(sum_err)
		




a=0
b=math.pi
training_set_base = np.linspace(a,b,100).tolist()
test_set = np.linspace(a,b,50).tolist()
y= []
for i in training_set_base:
	y.append(math.sin(i))

poly = lagrange(training_set_base,y)

#Calculate training_error

print ("MSE on training set without noise ", str(cal_log_mse(poly, training_set_base)))
print ("MSE on test set without noise ", str(cal_log_mse(poly, test_set)))


variance = [1, 1.5, 2, 2.5, 3, 3.5, 4]

for var in variance:
	noise = np.random.normal(0, var, 100)

	training_set_with_noise = []
	
	for i in range(0,len(training_set_base)):
		training_set_with_noise.append(training_set_base[i] +noise[i])
	
	y=[]
	for i in training_set_with_noise:
		y.append(math.sin(i))

	poly = lagrange(training_set_with_noise,y)
	
	print ("MSE on training set with noise with variance "+ str(var) + " is: "+  str(cal_log_mse(poly, training_set_with_noise)))
	
	print ("MSE on test set with noise with variance "+ str(var) + " is: "+  str(cal_log_mse(poly, test_set)))







	 





