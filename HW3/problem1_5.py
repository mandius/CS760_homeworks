import sys
import os
import matplotlib.pyplot as plt

import sys
import os
import matplotlib.pyplot as plt


def plot_roc(data):
	act_pos =0
	act_neg=0
	
	for i in data:
		if(i[1]==0):
			act_neg= act_neg+1
		else:
			act_pos= act_pos+1
	data_point_TPR =[]
	data_point_FPR = []
	
	threshold =0
	FP=act_neg
	TP=act_pos
	
	TPR = TP/act_pos
	FPR = FP/act_neg
	
	data_point_TPR.append(TPR)
	data_point_FPR.append(FPR)
	
	last_TPR = TPR
	
	for i in range(0, len(data)):
		if(data[i][1]==0):
			FP = FP-1.0
			FPR = FP/act_neg
			
			data_point_TPR.append(TPR)
			data_point_FPR.append(FPR)
		else:
			TP = TP-1.0
			TPR = TP/act_pos
			data_point_TPR.append(TPR)
			data_point_FPR.append(FPR)

	plt.plot(data_point_FPR, data_point_TPR, marker="o")
	plt.xlabel("False Positive Rate", fontsize=10) # set x-axis label
	plt.ylabel("True Positive Rate", fontsize=10) # set y-axis label
	plt.show()
	
			
			



data = [[0.1,0],[0.2,0],[0.3,1],[0.4,1],[0.45,0],[0.55,1],[0.7,1],[0.8,0],[0.85,1],[0.95,1]]

plot_roc(data)
