from random import seed, randrange, random
from csv import reader
from math import exp
import numpy as np
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score

# Ada-Act Activation Function
def ada_act(k0, k1, x):
    return k0 + k1 * x

# Load a CSV file
def loadCsv(filename):
    trainSet = []
    lines = csv.reader(open(filename, 'r'))
    dataset = list(lines)
    for i in range(len(dataset)):
        for j in range(4):
            dataset[i][j] = float(dataset[i][j])
        trainSet.append(dataset[i])
    return trainSet

# Min-max scaling for normalization
def minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

# Rescale dataset columns to the range 0-1
def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            
# Rescale dataset columns to the range 0-1
def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            
# Convert string column to float
def column_to_float(dataset, column):
    for row in dataset:
        try:
            row[column] = float(row[column])
        except ValueError:
            print("Error with row", column, ":", row[column])
            pass

