from sklearn.ensemble import RandomForestClassifier
import sys
import numpy as np

# File: randomForest_RNASeq.py
#	This file defines the Random Forest Classifier. It fits the training data and the
#	samples to the classifier. Then, it takes training data and makes predictions, returning the 
#	results of the predictions.

rfClf = RandomForestClassifier(n_estimators=10)

# Source: http://scikit-learn.org/stable/modules/ensemble.html
def fitTrainingData(training_data, nSamples):
	# fit(X,Y)
		# x = 2D array of cells
		# y = 1D list of classifiers i.e. [group1, group1, group2, group2]

	# makes lists into np arrays
	training_dataNP = np.array(training_data)
	nSamplesNP = np.array(nSamples)

	rfClf.fit(training_dataNP, nSamplesNP)

def predictTestData(testing_data):
	# make list into np array
	testing_dataNP = np.array(testing_data)

	# predict the values
	predicted = rfClf.predict(testing_dataNP)

	return predicted