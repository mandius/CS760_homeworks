from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import math
import random
import numpy as np
import matplotlib.pyplot as plt


a=0
b=8*math.pi
x = np.linspace(a,b,100).tolist()
y= []
for i in x:
	y.append(math.sin(i))

poly = lagrange(x,y)

#print(poly.coef)
y_val= []
for i in range(0,len(x)):
	y_val.append(Polynomial(poly.coef[::-1])(x[i]))
plt.plot(x, y_val, marker= "o")
plt.show()

sum_err =0

print(Polynomial(poly.coef[::-1]).coef)

for i in range(0,len(x)):

	err = Polynomial(poly.coef[::-1])(x[i]) - y[i]
	sum_err = sum_err + err*err

mse = sum_err/100




print ("MSE on training set without noise ", str(sum_err))

variance = [0.1, 0.5, 1, 1.5, 2]

for var in variance:
	noise = np.random.normal(0, var, 100)
	
	for i in range(0,len(x)):
		x[i] =x[i] +noise[i]
	
	y=[]
	for i in x:
		y.append(math.sin(i))

	poly = lagrange(x,y)
	
	plt.scatter(x, y, marker= "o")
	plt.show()

	for i in range(0,len(x)):
		err = Polynomial(poly.coef[::-1])(x[i]) - y[i]
		sum_err = sum_err + err*err
	
	mse = sum_err/100
	print ("MSE on training set with noise of variance  ", str(var), " is: " , str(mse))







	 





