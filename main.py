import sys
import time
import numpy as np
from RNASeqData import RNASeqData
import preprocess
import rbfSVC_RNASeq
# import neuralNetwork_RNASeq
import knn_RNASeq
import randomForest_RNASeq
import analysis

# File: main.py
#
#	This file processes the command line arguments supplied and runs the program using the selected classifier. Based on user input,
#	the program decides whether to down sample, cross validate, and which classifier to use. This file defines the four classifiers
# 	supplied to the user for classification: Support Vector Machine using Radial Basis Function Kernel, Mutli-Layer Perceptron (Neural Network), 
#	K-Nearest Neighbor, and Random Forest. After calling the preproccess code and running the classification, this class sends the results
#	to an analysis file that performs evaluations and writes the results to file


def rbfSVC(trainingData, testingData, trainingDataTargets, testingDataTargets):
	# fit training data to rbf svc
	rbfSVC_RNASeq.fitTrainingData(trainingData, trainingDataTargets)

	# predict values using rbf support vector machine
	rbfSVC_predictionResults = rbfSVC_RNASeq.predictTestData(testingData)

	return rbfSVC_predictionResults

def mlp(trainingData, testingData, trainingDataTargets, testingDataTargets):
	# fit training data to multi layer perceptron
	neuralNetwork_RNASeq.fitTrainingData(trainingData, trainingDataTargets)

	# predict values using neural network
	neuralNetwork_predictionResults = neuralNetwork_RNASeq.predictTestData(testingData)
	
	return neuralNetwork_predictionResults

def knn(trainingData, testingData, trainingDataTargets, testingDataTargets):
	# fit training data to knn
	knn_RNASeq.fitTrainingData(trainingData, trainingDataTargets)

	# predict the values using knn classifier
	knn_predictionResults = knn_RNASeq.predictTestData(testingData)

	return knn_predictionResults

def rf(trainingData, testingData, trainingDataTargets, testingDataTargets):
	# fit training data to rf
	randomForest_RNASeq.fitTrainingData(trainingData, trainingDataTargets)

	# predict the values using random forest classifier
	rf_predictionResults =  randomForest_RNASeq.predictTestData(testingData)

	return rf_predictionResults

if __name__ == '__main__':
	t0 = time.clock()
	print "start"
	
	# check for correct number of args
	if int(sys.argv[3]) == 3 and len(sys.argv) != 7:
		print "Usage: python main.py <raw_data_file> <annotations_file> <classifier [1,2,3,4] - svm, nn, knn, rf> <down sample? --> 0,1> <cross validate? --> 0,1> <n_neighbors (only if knn)>"
		sys.exit(0)
	
	if int(sys.argv[3]) != 3 and len(sys.argv) != 6:
		print "Usage: python main.py <raw_data_file> <annotations_file> <classifier [1,2,3,4] - svm, nn, knn, rf> <down sample? --> 0,1> <cross validate? --> 0,1> <n_neighbors (only if knn)>"
		sys.exit(0)

	raw_data_file = sys.argv[1]
	annotations_file = sys.argv[2]
	classifier = int(sys.argv[3])
	downSampleFlag = False
	crossValidateFlag = False
	if sys.argv[4] == "1":
		downSampleFlag = True
	if sys.argv[5] == "1":
		crossValidateFlag = True

	n_neighbors = -1
	if classifier == 3: # using knn, need to supply param for number of neighbors
		n_neighbors = int(sys.argv[6])
		knn_RNASeq.initializeKnn(n_neighbors) # initialize the classifier with n_neighbors



	print "Using:"
	print " - raw data: {raw}".format(raw=raw_data_file)
	print " - annotations: {ann}".format(ann=annotations_file)
	
	if classifier == 1:
		print " - Using Radial Basis Function Kernel Support Vector Machine"
	elif classifier == 2:
		print " - Using Multi-Layer Perceptron (Neural Network)"
	elif classifier == 3:
		print " - Using K Nearest Neighbor Classifier with k = {k}".format(k=n_neighbors)
	elif classifier == 4:
		print " - Using Random Forest Classifier"
	else:
		print "** ERROR: invalid classifier selection"
		sys.exit(0)
	
	if downSampleFlag:
		print "** Down sampling enabled **"
	else:
		print "** Down sampling disabled **"
	if crossValidateFlag:
		print "** Cross validation enabled **"
	else:
		print "** Cross validation disabled **"


	# initialize the data set class
	data = RNASeqData(raw_data_file, annotations_file)
	
	# read raw RNA seq data into memory
	data.setRawData(preprocess.loadRawData(data.getRawDataFileName()))

	# read cell identifier annotations into memory
	data.setCellIdentifierAnnotations(preprocess.loadCellIdentifierAnnotations(data.getAnnotationsFileName(), 
		data.getNumCellsRaw()))

	# read molecule count annotations into memory
	data.setMoleculeCountAnnotations(preprocess.loadMoleculeCountAnnotations(data.getAnnotationsFileName(),
		data.getNumCellsRaw()))

	if downSampleFlag:
		# down sample the data by cluster size --> MAKE DOWN SAMPLING A CLA OPTION
		#	 i.e. scale all cluster size to the smallest cluster (by number of cells)
		# save down sampled data and random indices for accessing corresponding annotations
		downSampleClusterData, randIndices = preprocess.downSampleByClusterSize(data.getRawData(), 
			data.getCellIdentifierAnnotations())

		# add the data and random indices reference to the data class
		data.setDSClusterData(downSampleClusterData)
		data.setRandIndicesFromDS(randIndices)

		# down sample the data by the cell with the least number of molecules
		data.setDSCluster_MoleculeData(preprocess.downSampleByMoleculeCount(data.getDSClusterData(),
			data.getMoleculeCountAnnotations(), data.getRandIndices()))

		if crossValidateFlag:
			# make 10-fold cross validation data
			data.makeCrossValidationTrainingAndTestingData(downSampleFlag)

			folds = data.getFolds()

			foldsKey = data.getFoldsKey()

			# make sure the data is parallel
			if len(folds) != len(foldsKey) or len(folds[0]) != len(foldsKey[0]):
				print "error: folds and folds key are not parallel data sets"
				sys.exit(0)

			iterator = 0 # we'll use this to iterate through folds and use each as the training data
			foldsEvaluations = []
			while iterator < 10:
				testingData = folds[iterator]
				testingDataKey = foldsKey[iterator]

				# make 2D arrays of training cells and keys
				trainingFolds = []
				trainingKeys = []
				i = 0
				while i < 10:
					if i != iterator:
						for cell in folds[i]:
							trainingFolds.append(cell)
						for key in foldsKey[i]:
							trainingKeys.append(key)
					i += 1
				
				if classifier == 1:
					# ***************** RBF SVC *****************
					# fit and make predictions
					rbfSVC_predictionResults = rbfSVC(trainingFolds, testingData, trainingKeys, testingDataKey)

					# add the accuracies for this fold to accuracies list
					foldsEvaluations.append(analysis.calculateEvaluations(rbfSVC_predictionResults, testingDataKey))
					# ***************** END RBF SVC *****************

				elif classifier == 2:
					# ***************** MLP *****************
					# fit and make predictions
					neuralNetwork_predictionResults = mlp(trainingFolds, testingData, trainingKeys, testingDataKey)

					# add the accuracies for this fold to accuracies list
					foldsEvaluations.append(analysis.calculateEvaluations(neuralNetwork_predictionResults, testingDataKey))
					# ***************** END MLP *****************

				elif classifier == 3:
					# ***************** KNN *****************
					# fit and make predictions
					knn_predictionResults = knn(trainingFolds, testingData, trainingKeys, testingDataKey)

					# add the accuracies for this fold to the accuracies list
					foldsEvaluations.append(analysis.calculateEvaluations(knn_predictionResults, testingDataKey))
					# ***************** END KNN *****************

				elif classifier == 4:
					# ***************** RF *****************
					# fit and make predictions
					rf_predictionResults = rf(trainingFolds, testingData, trainingKeys, testingDataKey)

					# add the accuracies for this fold to the accuracies list
					foldsEvaluations.append(analysis.calculateEvaluations(rf_predictionResults, testingDataKey))
					# ***************** END RF *****************


				# increment iterator to process the next fold as testing data
				iterator += 1
				print "finished fold #{num}".format(num=iterator)


			if classifier == 1:
				# ***************** RBF SVC *****************
				analysis.analyzeAndWriteToFile("Radial Basis Function Support Vector Machine", rbfSVC_predictionResults, testingDataKey, foldsEvaluations, 10, 0)
				# ***************** END RBF SVC *****************
			elif classifier == 2:
				# ***************** MLP *****************
				analysis.analyzeAndWriteToFile("Multi-Layer Perceptron (Neural Network)", neuralNetwork_predictionResults, testingDataKey, foldsEvaluations, 10, 0)
				# ***************** END MLP *****************
			elif classifier == 3:
				# ***************** KNN *****************
				analysis.analyzeAndWriteToFile("KNearestNeighbor Classifier_{k}".format(k=n_neighbors), knn_predictionResults, testingDataKey, foldsEvaluations, 10, 0)
				# ***************** END KNN *****************
			elif classifier == 4:
				# ***************** RF *****************
				analysis.analyzeAndWriteToFile("Random Forest Classifier", rf_predictionResults, testingDataKey, foldsEvaluations, 10, 0)
				# ***************** END RF *****************

		else:
			# partition the down sampled data set into 70% training and 30% testing
			data.makeDSTrainingAndTestingData()

			if classifier == 1:
				# ***************** RBF SVC *****************
				rbfSVC_predictionResults = rbfSVC(data.getDSTrainingData(), data.getDSTestingData(), data.getDSTargetValues(),
					data.getDSTestingDataTargetValues())

				# analyze results using robust evaluations
				foldsEvaluations = [] # single fold list but we still need to use a 3D list
				foldsEvaluations.append(analysis.calculateEvaluations(rbfSVC_predictionResults, data.getDSTestingDataTargetValues()))

				analysis.analyzeAndWriteToFile("Radial Basis Function Support Vector Machine", rbfSVC_predictionResults, data.getDSTestingDataTargetValues(), foldsEvaluations, 1, 1)
				# ***************** END RBF SVC *****************
			elif classifier == 2:
				# ***************** MLP *****************
				neuralNetwork_predictionResults = mlp(data.getDSTrainingData(), data.getDSTestingData(), data.getDSTargetValues(),
					data.getDSTestingDataTargetValues())

				# analyze results using robust evaluations
				foldsEvaluations = [] # single fold list but we still need to use a 3D list
				foldsEvaluations.append(analysis.calculateEvaluations(neuralNetwork_predictionResults, data.getDSTestingDataTargetValues()))

				analysis.analyzeAndWriteToFile("Multi-Layer Perceptron (Neural Network)", neuralNetwork_predictionResults, data.getDSTestingDataTargetValues(), foldsEvaluations, 1, 1)
				# ***************** END MLP *****************
			elif classifier == 3:
				# ***************** KNN *****************
				knn_predictionResults = knn(data.getDSTrainingData(), data.getDSTestingData(), data.getDSTargetValues(),
					data.getDSTestingDataTargetValues())

				foldsEvaluations = [] # single fold list but we still need to use a 3D list
				foldsEvaluations.append(analysis.calculateEvaluations(knn_predictionResults, data.getDSTestingDataTargetValues()))

				analysis.analyzeAndWriteToFile("KNearestNeighbor Classifier_{k}".format(k=n_neighbors), knn_predictionResults, data.getDSTestingDataTargetValues(), foldsEvaluations, 1, 1)
				# ***************** END KNN *****************
			elif classifier == 4:
				# ***************** RF *****************
				rf_predictionResults = rf(data.getDSTrainingData(), data.getDSTestingData(), data.getDSTargetValues(),
					data.getDSTestingDataTargetValues())

				foldsEvaluations = [] # single fold list but we still need to use a 3D list
				foldsEvaluations.append(analysis.calculateEvaluations(rf_predictionResults, data.getDSTestingDataTargetValues()))

				analysis.analyzeAndWriteToFile("Random Forest Classifier", rf_predictionResults, data.getDSTestingDataTargetValues(), foldsEvaluations, 1, 1)
				# ***************** END RF *****************

	else:
		if crossValidateFlag:
			# make 10-fold cross validation data
			data.makeCrossValidationTrainingAndTestingData(downSampleFlag)

			folds = data.getFolds()

			foldsKey = data.getFoldsKey()

			# make sure the data is parallel
			if len(folds) != len(foldsKey) or len(folds[0]) != len(foldsKey[0]):
				print "error: folds and folds key are not parallel data sets"
				sys.exit(0)

			iterator = 0 # we'll use this to iterate through folds and use each as the training data
			foldsEvaluations = []
			while iterator < 10:
				testingData = folds[iterator]
				testingDataKey = foldsKey[iterator]

				# make 2D arrays of training cells and keys
				trainingFolds = []
				trainingKeys = []
				i = 0
				while i < 10:
					if i != iterator:
						for cell in folds[i]:
							trainingFolds.append(cell)
						for key in foldsKey[i]:
							trainingKeys.append(key)
					i += 1

				if classifier == 1:
					# ***************** RBF SVC *****************
					# fit and make predictions
					rbfSVC_predictionResults = rbfSVC(trainingFolds, testingData, trainingKeys, testingDataKey)

					# add the accuracies for this fold to accuracies list
					foldsEvaluations.append(analysis.calculateEvaluations(rbfSVC_predictionResults, testingDataKey))
					# ***************** END RBF SVC *****************
				elif classifier == 2:
					# ***************** MLP *****************
					# fit and make predictions
					neuralNetwork_predictionResults = mlp(trainingFolds, testingData, trainingKeys, testingDataKey)

					# add the accuracies for this fold to accuracies list
					foldsEvaluations.append(analysis.calculateEvaluations(neuralNetwork_predictionResults, testingDataKey))
					# ***************** END MLP *****************
				elif classifier == 3:
					# ***************** KNN *****************
					# fit and make predictions
					knn_predictionResults = knn(trainingFolds, testingData, trainingKeys, testingDataKey)

					# add the accuracies for this fold to accuracies list
					foldsEvaluations.append(analysis.calculateEvaluations(knn_predictionResults, testingDataKey))
					# ***************** END KNN *****************
				elif classifier == 4:
					# ***************** RF *****************
					# fit and make predictions
					rf_predictionResults = rf(trainingFolds, testingData, trainingKeys, testingDataKey)

					# add the accuracies for this fold to accuracies list
					foldsEvaluations.append(analysis.calculateEvaluations(rf_predictionResults, testingDataKey))
					# ***************** END RF *****************


				# increment iterator to process the next fold as testing data
				iterator += 1
				print "finished fold #{num}".format(num=iterator)

			if classifier == 1:
				# ***************** RBF SVC *****************
				analysis.analyzeAndWriteToFile("Radial Basis Function Support Vector Machine", rbfSVC_predictionResults, testingDataKey, foldsEvaluations, 10, 2)
				# ***************** END RBF SVC *****************
			elif classifier == 2:
				# ***************** MLP *****************
				analysis.analyzeAndWriteToFile("Multi-Layer Perceptron (Neural Network)", neuralNetwork_predictionResults, testingDataKey, foldsEvaluations, 10, 2)
				# ***************** END MLP *****************
			elif classifier == 3:
				# ***************** KNN *****************
				analysis.analyzeAndWriteToFile("KNearestNeighbor Classifier_{k}".format(k=n_neighbors), knn_predictionResults, testingDataKey, foldsEvaluations, 10, 2)
				# ***************** END KNN *****************
			elif classifier == 4:
				# ***************** RF *****************
				analysis.analyzeAndWriteToFile("Random Forest Classifier", rf_predictionResults, testingDataKey, foldsEvaluations, 10, 2)
				# ***************** END RF *****************

		else:
			# partition the data set into 70% training and 30% testing
			data.makeTrainingAndTestingData()

			if classifier == 1:
				# ***************** RBF SVC *****************
				rbfSVC_predictionResults = rbfSVC(data.getTrainingData(), data.getTestingData(), data.getTrainingDataTargetValues(),
					data.getTestingDataTargetValues())

				# analyze results using robust evaluations
				foldsEvaluations = [] # single fold list but we still need to use a 3D list
				foldsEvaluations.append(analysis.calculateEvaluations(rbfSVC_predictionResults, data.getTestingDataTargetValues()))

				analysis.analyzeAndWriteToFile("Radial Basis Function Support Vector Machine", rbfSVC_predictionResults, data.getTestingDataTargetValues(), foldsEvaluations, 1, 3)
				# ***************** END RBF SVC *****************
			
			elif classifier == 2:
				# ***************** MLP *****************
				neuralNetwork_predictionResults = mlp(data.getTrainingData(), data.getTestingData(), data.getTrainingDataTargetValues(),
					data.getTestingDataTargetValues())

				# analyze results using robust evaluations
				foldsEvaluations = [] # single fold list but we still need to use a 3D list
				foldsEvaluations.append(analysis.calculateEvaluations(neuralNetwork_predictionResults, data.getTestingDataTargetValues()))

				analysis.analyzeAndWriteToFile("Multi-Layer Perceptron (Neural Network)", neuralNetwork_predictionResults, data.getTestingDataTargetValues(), foldsEvaluations, 1, 3)
				# ***************** END MLP *****************

			elif classifier == 3:
				# ***************** KNN *****************
				knn_predictionResults = knn(data.getTrainingData(), data.getTestingData(), data.getTrainingDataTargetValues(),
					data.getTestingDataTargetValues())

				# analyze results using robust evaluations
				foldsEvaluations = []

				foldsEvaluations.append(analysis.calculateEvaluations(knn_predictionResults, data.getTestingDataTargetValues()))

				analysis.analyzeAndWriteToFile("KNearestNeighbor Classifier_{k}".format(k=n_neighbors), knn_predictionResults, data.getTestingDataTargetValues(), foldsEvaluations, 1, 3)
				# ***************** END KNN *****************

			elif classifier == 4:
				# ***************** RF *****************
				rf_predictionResults = rf(data.getTrainingData(), data.getTestingData(), data.getTrainingDataTargetValues(),
					data.getTestingDataTargetValues())

				# analyze results using robust evaluations
				foldsEvaluations = []

				foldsEvaluations.append(analysis.calculateEvaluations(rf_predictionResults, data.getTestingDataTargetValues()))

				analysis.analyzeAndWriteToFile("Random Forest Classifier", rf_predictionResults, data.getTestingDataTargetValues(), foldsEvaluations, 1, 3)
				# ***************** END RF *****************

	print "\nprogram execution: {t} seconds".format(t=time.clock()-t0)
	print "exiting"