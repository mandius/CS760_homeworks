dataset_l = [100, 500, 1000, 5000, 10000, 50000]
#For scratch neural network 0 weight initialization
#accuracy = [0.1135, 0.1135, 0.1028, 0.1135, 0.4389, 0.5097]

#For scratch neural network with random weight initialization
#accuracy =  [0.3994, 0.6497, 0.7555, 0.8749, 0.902, 0.9509]

#For pytorch base neural network with random weights.
accuracy=  [0.2129, 0.7255, 0.8365, 0.9018, 0.917, 0.9593]

#For pytorch base neural network with zero weights.
accuracy= [0.1135, 0.1135, 0.1028, 0.7994, 0.9079, 0.95]


accuracy_p = []
import matplotlib.pyplot as plt

for a in accuracy:
	accuracy_p.append(a*100)


plt.plot(dataset_l, accuracy_p)
plt.grid()
plt.xlabel("Training Dataset Size")
plt.ylabel("Accuracy")
plt.xscale("log")
plt.show()
