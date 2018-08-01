import numpy as np
from sklearn.datasets import load_iris

dataset = load_iris()

# Create a new numpy array length of dataset, but include extra column for target
# This reduces the complexity of computing the number of classes, when doing a gini
# calculation as we don't need to refer to the correct target class from the target array array
processed_data = np.empty([len(dataset.data), len(dataset.data[0]) + 1])
for i in range(len(dataset.data)):
    processed_data[i] = np.append(dataset.data[i], dataset.target[i])

"""Counts the number of each type of example in a dataset."""
def class_counts(data):
    counts = {}  # a dictionary of label -> count.
    for row in data:
        label = row[-1] # label is appended to the end
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

class Question:
    """
        A Question is used to partition a dataset.
        This class records the column and the value of the question
    """
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        return example[self.column] >= self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


def partition(rows, question):
    """Partitions a dataset.
    For each row in the dataset, check if it matches the question.
    If so, add it to true_rows otherwise add it to false
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    """
    Calculate the Gini Impurity for a list of rows.
    """
    counts = class_counts(rows)
    impurity = 1
    for label in counts:
        prob_of_label = counts[label] / float(len(rows))
        impurity -= prob_of_label**2
    return impurity


def info_gain(left, right, current_uncertainty):
    """Information Gain.
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

current_uncertainty = gini(processed_data)
print(current_uncertainty)

def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the column / Value that produced best gain
    current_uncertainty = gini(rows)
    num_features = len(rows[0]) - 1  # number of columns. This will change for us

    for col in range(n_features):  # for each feature, col here equals 1 feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value
            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # If this gain is better than present best gain, record the gain, as well as the question
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


best_gain, best_question = find_best_split(processed_data)

print(best_gain)
print("col: %s, val: %s" % (best_q.column, best_question.value))
