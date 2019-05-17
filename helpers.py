import numpy as np
import random

def class_counts( data):
    counts = {}
    for row in data:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def draw(weights):
    choice = random.uniform( 0, sum( weights ) )
    i = 0
    for weight in weights:
        choice -= weight
        if choice <= 0:
            return i
        i += 1

def normalize(weights):
    norm = sum(weights)
    return tuple(m / norm for m in weights)

def make_two_class(x, y):
    y_2c = []
    x_2c = []
    for i in range(len(y)):
        if y[i] == 0 or y[i] == 1:
            y_2c.append(y[i])
            x_2c.append(x[i])
    return np.array(x_2c), np.array(y_2c)
