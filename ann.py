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

# Convert string column to integer
def column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation using Ada-Act
def transfer(activation, k0, k1):
    return ada_act(k0, k1, activation)

# Forward propagate input to a network output
def forward_propagate(network, row, k0, k1):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation, k0, k1)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of an neuron output with Ada-Act
def transfer_derivative(output, k1):
    return k1

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected, k1):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'], k1)

# Update network weights with error
def update_weights(network, row, l_rate, mu):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                temp = l_rate * neuron['delta'] * inputs[j] + mu * neuron['prev'][j]
                neuron['weights'][j] += temp
                neuron['prev'][j] = temp
            temp = l_rate * neuron['delta'] + mu * neuron['prev'][-1]
            neuron['weights'][-1] += temp
            neuron['prev'][-1] = temp

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs, mu, k0, k1):
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row, k0, k1)
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1
            backward_propagate_error(network, expected, k1)
            update_weights(network, row, l_rate, mu)

