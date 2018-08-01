import numpy as np
from sklearn.datasets import load_iris



"""
Create a new 2d numpy array.
Append the target class, for ever array of data labels. This reduces the
complexity when computing the gini index, as we don't need to make reference
to the dataset.target array
"""
dataset = load_iris()
processed_data = np.empty([len(dataset.data), len(dataset.data[0]) + 1])
for i in range(len(dataset.data)):
    processed_data[i] = np.append(dataset.data[i], dataset.target[i])


"""
Counts the number of label classes for an array of data.
Because the label is in the last index of every row, this is easy
"""
def class_counts(data):
    counts = {}  # a dictionary of label -> count.
    for row in data:
        label = row[-1] #
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

"""
A Decision is used to partition a dataset.
This class records the column/feature and the value "splitting" on
"""
class Decision:
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value
    def match(self, against):
        return against[self.feature] >= self.value


"""
Partitions a dataset.
For each row in the dataset, check if it matches the question.
If so, add it to true_rows otherwise add it to false
"""
def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

"""
Calculate the Gini for a list of rows.
"""
def gini(rows):

    counts = class_counts(rows)
    impurity = 1
    for label in counts:
        prob_of_label = counts[label] / float(len(rows))
        impurity -= prob_of_label**2
    return impurity


"""
Information Gain.
The uncertainty of the starting node, minus the weighted impurity of
two child nodes.
"""
def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

"""
Find the best question to ask by iterating over every feature / value
and calculating the information gain. Inefficient AF
"""
def find_best_split(rows):
    optimal_gain = 0  # keep track of the best information gain
    optimal_decision = None  # keep train of the column / Value that produced best gain
    current_gini = gini(rows)

    # for each feature, col here equals 1 feature
    for col in range( len( rows[0] ) - 1 ):
        # unique values in the colum
        values = set( [row[col] for row in rows] )
        # for each value
        for val in values:
            d = Decision( col, val )
            true_rows, false_rows = partition( rows, d )
            # If one branch has len 0, nothing was split.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            g = info_gain(true_rows, false_rows, current_gini)

            # If this gain is better than present best gain, record the gain, as well as the question
            if g >= optimal_gain:
                optimal_gain, optimal_decision = g, d

    return optimal_gain, optimal_decision

best_g, best_d = find_best_split(processed_data)

print(best_g)
print("col: %s, val: %s" % (best_d.feature, best_d.value))
