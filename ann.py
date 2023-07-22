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