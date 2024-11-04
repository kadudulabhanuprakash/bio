from sklearn.neural_network import MLPClassifier
import numpy as np

# File: neuralNetwork_RNASeq.py
#	This file defines the Multi-Layer Perceptron (Neural Network). It fits the training data and the
#	samples to the classifier. Then, it takes training data and makes predictions, returning the 
#	results of the predictions.


mlpClf = MLPClassifier(activation='relu', algorithm='adam', alpha=1e-05,
       batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(100,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

# Source: http://scikit-learn.org/dev/modules/neural_networks_supervised.html
def fitTrainingData(training_data, nSamples):
	# fit(X,Y)
		# x = 2D array of cells
		# y = 1D list of classifiers i.e. [group1, group1, group2, group2]

	# makes lists into np arrays
	training_dataNP = np.array(training_data)
	nSamplesNP = np.array(nSamples)

	mlpClf.fit(training_dataNP, nSamplesNP)

def predictTestData(testing_data):
	# make list into np array
	testing_dataNP = np.array(testing_data)

	# predict the values
	predicted = mlpClf.predict(testing_dataNP)

	return predicted