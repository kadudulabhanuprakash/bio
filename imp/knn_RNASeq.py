from sklearn.neighbors import KNeighborsClassifier
import sys
import numpy as np

# File: knn_RNASeq.py
#	This file defines the K Nearest Neighbor classifier. It allows the user to specificy the number of neighbors used 
#	and then fits the training data and the samples to the classifier. Then, it takes training data and makes predictions,
#	returning the results of the predictions.



# Parameters:
#	- n_neighbors: number of neighbors to use by default for k_neighbors queries
#	- weights: weight function used in prediction -- uniform, distance (inverse of distance)
#	- algorithm: ball_tree, kd_tree, brute, auto (attempts to decide most appropriate based on values passed to fit)
# 	- leaf size: size passed to BallTree or KDTree (default = 30)
#	- metric: distance metric used for the tree (minkowski and p=2 is euclidean)
#	- p: power parameter for the minkowski metric (1 - manhattan distance, 2 - euclidean distance)
#	- n_jobs: the number of parallel jobs to run for neighbors search
knn = KNeighborsClassifier()

def initializeKnn(n_neighbors):
	global knn
	knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=n_neighbors, p=2,
           weights='uniform')

# Source: http://scikit-learn.org/dev/modules/neural_networks_supervised.html
def fitTrainingData(training_data, nSamples):
	# fit(X,Y)
		# x = 2D array of cells
		# y = 1D list of classifiers i.e. [group1, group1, group2, group2]

	# makes lists into np arrays
	training_dataNP = np.array(training_data)
	nSamplesNP = np.array(nSamples)

	knn.fit(training_dataNP, nSamplesNP)

def predictTestData(testing_data):
	# make list into np array
	testing_dataNP = np.array(testing_data)

	# predict the values
	predicted = knn.predict(testing_dataNP)

	return predicted