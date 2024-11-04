# Support vector classiciation of RNA seq data using the Radial Basis Function Kernel (rbf)
from sklearn.svm import SVC
import numpy as np

# File: rbfSVC_RNASeq.py
#	This file defines the Support Vector Machine using a Radial Basis Function Kernel. It fits the training data and the
#	samples to the classifier. Then, it takes training data and makes predictions, returning the 
#	results of the predictions.

# PARAMETERS:
	# decision function shape : 'ovo' one-vs-one, 'ovr' one-vs-rest, none
	# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, 
	# decision_function_shape=None, degree=3, gamma='auto', 
	# kernel='rbf', max_iter=-1, probability=False, random_state=None, 
	# shrinking=True, tol=0.001, verbose=False)

rbfSVC = SVC(C=1.0, class_weight=None, coef0=0.0, 
decision_function_shape=None, degree=3, gamma='auto', 
kernel='rbf', max_iter=-1, probability=False, random_state=None, 
shrinking=True, tol=0.001, verbose=False)

# Source: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
def fitTrainingData(training_data, nSamples):
	# fit(X,Y)
		# x = 2D array of cells
		# y = 1D list of classifiers i.e. [group1, group1, group2, group2]

	# makes lists into np arrays
	training_dataNP = np.array(training_data)
	nSamplesNP = np.array(nSamples)

	rbfSVC.fit(training_dataNP, nSamplesNP)

def predictTestData(testing_data):
	# make list into np array
	testing_dataNP = np.array(testing_data)

	# predict the values
	predicted = rbfSVC.predict(testing_dataNP)

	return predicted
