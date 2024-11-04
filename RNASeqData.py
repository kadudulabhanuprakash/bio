import sys
import random

class RNASeqData(object):

    def _init_(self, raw_data_file, annotations_file):
        print("\ninitializing RNASeqData")
        self.raw_data_file = raw_data_file
        self.annotations_file = annotations_file

    def setRawData(self, rawData):
        self.rawData = rawData

    def setDSClusterData(self, dsClusterData):
        self.dsClusterData = dsClusterData

    def setDSCluster_MoleculeData(self, dsCluster_MoleculeData):
        self.dsCluster_MoleculeData = dsCluster_MoleculeData

    def setCellIdentifierAnnotations(self, cellIdentifierAnnotations):
        self.cellIdentifierAnnotations = cellIdentifierAnnotations

    def setMoleculeCountAnnotations(self, moleculeCountAnnotations):
        self.moleculeCountAnnotations = moleculeCountAnnotations

    def setRandIndicesFromDS(self, randIndices):
        self.randIndices = sorted(randIndices)

    def makeDSTrainingAndTestingData(self):
        print("\npartitioning data set - 70% training, 30% testing")

        trainingData = []
        testingData = []

        types = [[] for _ in range(9)]
        typesIdxs = [[] for _ in range(9)]

        for iterator, idx in enumerate(self.randIndices):
            _type = int(self.cellIdentifierAnnotations[idx])

            if 1 <= _type <= 9:
                types[_type - 1].append(self.dsCluster_MoleculeData[iterator])
                typesIdxs[_type - 1].append(idx)

        numCells = len(types[0])
        for i in range(1, 9):
            if len(types[i]) != numCells:
                print(f"error: not all clusters have {numCells} cells")
                return

        indices = list(range(numCells))
        numTrainingCellsPerCluster = int(numCells * 0.7)

        trainingCellIdxs = [random.sample(indices, numTrainingCellsPerCluster) for _ in range(9)]

        trainingCells = [[] for _ in range(9)]
        trainingCellIdxsAnn = [[] for _ in range(9)]
        testingCellIdxsAnn = [[] for _ in range(9)]

        for i in range(9):
            for iterator, cell in enumerate(types[i]):
                if iterator in trainingCellIdxs[i]:
                    trainingCells[i].append(cell)
                    trainingCellIdxsAnn[i].append(typesIdxs[i][iterator])
                else:
                    testingData.append(cell)
                    testingCellIdxsAnn[i].append(typesIdxs[i][iterator])

        self.dsTrainingData = []
        targetValuesIdxs = []

        for i in range(9):
            if len(trainingCells[i]) != len(trainingCellIdxsAnn[i]):
                print(f"error: discrepancy between type {i+1} training cells and type {i+1} training cell indices")
            else:
                for cell, idx in zip(trainingCells[i], trainingCellIdxsAnn[i]):
                    self.dsTrainingData.append(cell)
                    targetValuesIdxs.append(idx)

        self.dsTargetValues = [int(self.cellIdentifierAnnotations[idx]) for idx in targetValuesIdxs]
        self.dsTestingData = testingData

        testingDataIdxsAnn = [idx for sublist in testingCellIdxsAnn for idx in sublist]
        self.dsTestingDataTargetValues = [int(self.cellIdentifierAnnotations[idx]) for idx in testingDataIdxsAnn]

        numTrainingDataCells = len(self.dsTrainingData)
        numTestingDataCells = len(self.dsTestingData)

        print(f"number training cells = {numTrainingDataCells}")
        print(f"number testing cells = {numTestingDataCells}")
        print("reference:")
        print(f"- total down-sampled cells = {len(self.dsCluster_MoleculeData)}")
        print(f"- down-sampled cells * 0.7 = {int(len(self.dsCluster_MoleculeData)*0.7)} --> approx.")
        print(f"- down-sampled cells * 0.3 = {int(len(self.dsCluster_MoleculeData)*0.3)} --> approx.")