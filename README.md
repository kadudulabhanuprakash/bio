#### Braden Katzman & Emily Berghoff
#### Columbia University Spring 2016
#### COMS W4761 - Computational Genomics
#### Professor Itsik Pe'er
#### Final Project



## Abstract:

	This program classifies single cell types from RNA-seq data in mice. Our approach is to use supervised
	machine learning algorithms on a single cell RNA-seq dataset with a marked training set and unmarked
	testing set. We used 4 machine learning algorithms with the hopes of determining the best supervised
	learning algorithm to classify unmarked cells.

## Machine Learning Algorithms:
- Support Vector Machine using Radial Basis Function Kernel

- Random Forests, Decision Trees

- KNN

- Multi-Layer Perceptron (Neural Network)

## Conclusions:
	We found that the Neural Network gave the best performance, and the Support Vector Machine
	gave the worst performance.
	

## Technical Notes:

	This project is written in Python using version 2.7. All machine learning algorithms are from 
	the scikit-learn library which uses SciPy, NumPy and matplotlib to implement data mining and 
	data analysis tools. The stable version of scikit-learn is available here:

	- Scikit-learn: http://scikit-learn.org/stable/
	
	*** IMPORTANT NOTE: The Multi-Layer Perceptron is not supported in the stable version of 
	scikit (as of 5/7/16). In order to use this classifier, the development version needs to 
	be used. As such, the import for the Neural Network file is commented out in main.py so 
	that the program compiles on the stable version of scikit-learn. In order to use the 
	Multi-Layer Perceptron, download and compile the development version of scikit-learn and 
	uncomment 'import neuralNetwork_RNASeq' on line 7 of main.py. Then, the neural network
	classifier can be used.

## Input Files:

- GSE60361C13005Expression.txt: this file holds the raw RNA seq data. It is organized by
cells (columns) and their gene expression levels (rows). It is roughly 3000 cells x 20000 
genes in size

- expressionmRNAAnnotations.txt: this file holds the annotations for the raw data including 
molecule count and cell type (1-9) classification

	*** IMPORTANT NOTE:
	Github's file size limit is 100MB. Both the raw data and annotations files exceed these 
	limits. As such, both files are compressed into a directory, 'RawData_Annotations_Compress.zip'.
	Upon cloning the source code, unpack this .zip and either supply the path into the directory
	to pass the files to main.py, or move the 2 files into the root directory of the project and
	just supply the file name. 

## Output Files:

- RNASeq_SingleCellClassification_Results: this directory holds the results from each of the 
classifiers. There are 4 run options in terms of preprocessing, and each of these options has
a subdirectory in the results folder. The two options which have a total of 4 combinations are
down sampling or not down sampling, and 10-fold cross validation or 1-fold cross validation. 
In order to converge on the 'true' results of the classifiers, each classifier is run 5 times 
on each of the 4 preprocessing run options. In addition, as the user has the ability to supply 
the number of neighbors to be used for KNN, there are 5 runs for each K = 1-8, on each of the
4 preprocessing run options.

## How to Run the Program:

	The usage of the program is as follows:

- python main.py {raw_data} {data_annotations} {classifier 1-4} {down_sample - 0 or 1} {cross_validate - 0 or 1} {n_neighbors}

	* To select a classifier, supply a number 1-4 corresponding to:
	- 1 = SVM
	- 2 = Neural Network
	- 3 = KNN
	- 4 = Random Forest

	* To enable down sampling, supply 1. To disable, supply 0

	* To enable cross validation, supply 1. To disable, supply 0

	*** ONLY IF USING KNN:
	* Supply a valid integer n_neighbors


Examples:

	KNN with 5 neighbors, down sampling enabled, cross validation disabled:

	- python main.py GSE60361C13005Expression.txt expressionmRNAAnnotations.txt 3 1 0 5

	SVM with down sampling disabled, cross validation enabled

	- python main.py GSE60361C13005Expression.txt expressionmRNAAnnotations.txt 1 0 1



## Description of Each File:

- main.py: 
	This file processes the command line arguments supplied and runs the program using the 
	selected classifier. Based on user input, the program decides whether to down sample, 
	cross validate, and which classifier to use. This file defines the four classifiers supplied
	to the user for classification: Support Vector Machine using Radial Basis Function Kernel,
	Mutli-Layer Perceptron (Neural Network), K-Nearest Neighbor, and Random Forest. After 
	calling the preproccess code and running the classification, this class sends the 
	results to an analysis file that performs evaluations and writes the results to file.

- RNASeqData.py
	This class object represents the RNA Seq Data. It holds the raw data, the annotations,
	and provides methods for partitioning the data. The partitions (for both down sampling
	and non down sampling and cross validation and no cross validation) randomly make partitions
	of the data for both training and testing, while simultaneously holding the annotations
	for the randomly selected testing data. The class also provides accessor methods for all data,
	annotations, training data, testing data, and training data target values to evaluate performance.

- preprocess.py
	This file does all of the preprocessing work before running classification. This file loads
	raw data and annotations into memory. Next, this file is used to down dample by both
	cluster size and molecule count.

- analysis.py
	After classification, this file is used to evaluate the performance of the classifier and 
	write the results to an output file. First, this file creates a confusion matrix and then 
	computes the accuracy, sensitivity, specificity, MCC, and F1 Score at both the class level 
	and the global level across the supplied number of folds (10 folds for cross validation and
	1 fold for non-cross validation). This class also uses a basic metric of merely counting
	the number of correct classifications that was used initially to check the performance of the
	classifiers. After evaluating, the results are written to a file.

- knn_RNASeq.py
	This file defines the K Nearest Neighbor classifier. It allows the user to specificy the 
	number of neighbors used and then fits the training data and the samples to the classifier.
	Then, it takes training data and makes predictions, returning the results of the predictions.

- neuralNetwork_RNASeq.py
	This file defines the Multi-Layer Perceptron (Neural Network). It fits the training data and the
	samples to the classifier. Then, it takes training data and makes predictions, returning the 
	results of the predictions.

- randomForest_RNASeq.py
	This file defines the Random Forest Classifier. It fits the training data and the
	samples to the classifier. Then, it takes training data and makes predictions, returning the 
	results of the predictions.

- rbfSVC_RNASeq.py
	This file defines the Support Vector Machine using a Radial Basis Function Kernel. It fits 
	the training data and the samples to the classifier. Then, it takes training data and makes
	predictions, returning the results of the predictions.



## Bibliography:
	Project motivated by data from:

- Zeisel, A. et al. Cell types in the mouse cortex and hippocampus revealed by single-cell RNA-seq. Science 347, 1138-1142 (2015).