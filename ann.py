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

