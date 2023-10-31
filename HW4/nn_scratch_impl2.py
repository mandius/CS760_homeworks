

import numpy as np

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import torch

import time





d = 28 * 28

d1 = 300

k = 10

bsize = 32

alpha = 0.01




class CrossNeuralModel():

	def __init__(self, sizes, bsize ,epochs=20, alpha=0.01):

		self.sizes = sizes

		self.epochs = epochs

		self.alpha = alpha

		self.bsize = bsize

		self.init_params()

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def softmax(self, x):

		exes = np.exp(x)

		deno = np.sum(exes)

		deno.resize(exes.shape[0], 1)

		return exes / deno

	def init_params(self):

		input_layer = int(self.sizes[0])

		hidden_1 = int(self.sizes[1])

		output_layer = int(self.sizes[2])

		# Random initialization of weights between -1 and 1

		self.w1 = np.random.uniform(low=-1, high=1, size=(input_layer, hidden_1))

		self.w2 = np.random.uniform(low=-1, high=1, size=(hidden_1, output_layer))

		# Zero initialization of weights

		#self.w1 = np.zeros((input_layer, hidden_1))

		#self.w2 = np.zeros((hidden_1, output_layer))

	def forward(self, inputs):

		self.linear_1 = np.matmul(self.w1.T, inputs)

		self.out1 = self.sigmoid(self.linear_1)

		self.linear2 = np.matmul(self.w2.T, self.out1)

		self.out2 = self.softmax(self.linear2)

		'''
		print("******************** Printing Shapes forward **********************")
		print("******************** inputs ***************************************")
		print(inputs.shape)
		print("******************** linear_1 *************************************")
		print(self.linear_1.shape)
		print("******************** out1     *************************************")
		print(self.out1.shape)
		print("******************* linear2  **************************************")
		print(self.linear2.shape)
		print("******************* out2     **************************************")
		print(self.out2.shape) 
		print("******************* w1 *****************")
		print(self.w1.shape)

		print("******************* w2 *****************")
		print(self.w2.shape)

		'''

		return self.out2

	def backward(self, x_train, y_train, output):
		
		#print("******************** Printing Shapes backward **********************")
		#print("******************** x_train ***************************************")
		#print(x_train.shape)
		#print("******************** y_train *************************************")
		#print(y_train.shape)

		d_loss = output - y_train

		#print("******************** dloss     *************************************")
		#print(d_loss.shape)

	
		delta_w2 =  np.matmul(self.out1, d_loss.T)
		#print("******************* delta_w2  **************************************")
		#print(delta_w2.shape)


		d_out_1 = np.matmul(self.w2, d_loss)

		#print("******************* d_out_1  **************************************")
		#print(d_out_1.shape)


		d_linear_1 = d_out_1 * self.sigmoid(self.linear_1) * (1 - self.sigmoid(self.linear_1))

		#print("******************* d_linear_1  **************************************")
		#print(d_linear_1.shape)

		delta_w1 =  np.matmul(x_train, d_linear_1.T)


		#print("******************* delta_w1  **************************************")
		#print(delta_w1.shape)





		return delta_w1, delta_w2

	def update_weights(self, w1_update, w2_update):

		self.w1 -= self.alpha * w1_update
		self.w2 -= self.alpha * w2_update

	def calculate_loss(self, y, y_hat):

		batch_size = y.shape[0]

		y = y.numpy()

		loss = np.sum(np.multiply(y, np.log(y_hat)))

		loss = -(1. / batch_size) * loss

		return loss

	def calculate_metrics(self, data_loader):

		losses = []

		correct = 0

		total = 0

		for i, data in enumerate(data_loader):
			x, y = data

			y_onehot = torch.zeros(y.shape[0], 10)

			y_onehot[range(y_onehot.shape[0]), y] = 1

			flattened_input = x.view(-1, 28 * 28)

			output = self.forward(flattened_input)

			predicted = np.argmax(output, axis=1)

			correct += np.sum((predicted == y.numpy()))

			total += y.shape[0]


			loss = self.calculate_loss(y_onehot, output)

			losses.append(loss)


		return (correct / total), np.mean(np.array(losses))

	def train(self, train_data_mnist, train_length):
		for i in range(self.epochs):
			print("***************************Epoch = "+ str(i) + "/" + str(self.epochs)+"*****************************")
			j=0
			while(j< train_length): 
				w1_grad = np.zeros((d, d1))
				w2_grad = np.zeros((d1, k))
      
				for k1 in range(self.bsize):
					train_data, train_label = train_data_mnist[j+k1]
					x = np.ndarray.flatten(train_data.numpy()).reshape(-1,1)
					y = np.zeros([self.sizes[2],1])
					y[train_label] = 1
					y_cap = self.forward(x)
					w1_inc, w2_inc = self.backward(x, y, y_cap)

					w1_grad = w1_grad + w1_inc
					w2_grad = w2_grad + w2_inc
				j = j+ self.bsize
				w1_grad = (1./self.bsize) * w1_grad
				w2_grad = (1./self.bsize) * w2_grad
				self.update_weights(w1_grad, w2_grad)
	
			
    
			
            


if __name__ == '__main__':
	model = CrossNeuralModel(sizes=[d, d1, k],alpha=alpha, bsize = bsize, epochs=100)
    
	mnist_train_data = datasets.MNIST('.', train=True,download=True, transform=transforms.ToTensor())
	mnist_test_data =  datasets.MNIST('.', train=False,download=True, transform=transforms.ToTensor())
	#print(len(mnist_train_data))
	accuracy = []
	correct_list = []
	train_lengths = [100, 500, 1000, 5000, 10000, 50000]
	for train_length in train_lengths:
		print("**************************** train_length = " + str(train_length) +  "********************************")
		model.train( mnist_train_data, train_length)
		#print("************************ w1 ***********************")
		#print(model.w1)
		#print("************************ w2 ***********************")
		#print(model.w2) 
		correct =0
		for i in range(len(mnist_test_data)):
			test_data, test_label = mnist_test_data[i]
			y = np.zeros([k,1])
			y[test_label]=1
			x = np.ndarray.flatten(test_data.numpy()).reshape(-1,1);
			y_cap = model.forward (x)
			test_label_pred = np.argmax(y_cap)
			if (test_label ==  test_label_pred):
				correct = correct +1
		correct_list.append(correct)
		accuracy.append((float(correct)/float(len(mnist_test_data)))*100)
		print((float(correct)/float(len(mnist_test_data)))*100)
	
	print(len(mnist_test_data))
	print(correct)
	print(train_lengths)
	print(accuracy)
	
	plt.plot(train_lengths, accuracy)
	plt.grid()
	plt.xlabel("Training Dataset Size")
	plt.ylabel("Accuracy")
	plt.xscale("log")
	plt.show()
     
     
	


