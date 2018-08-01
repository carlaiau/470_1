from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
#iris = load_iris()
#print(iris.feature_names)
#print(iris.target_names)
#print(iris.data[0])
#print(iris.target[0])
#for i in range(len(iris.target)):
#    print("%d\t%s\t%s" % (i, iris.target[i], iris.data[i]))


# Toy dataset.# Toy d
# Format: each row is an example.
# The last column is the label.
# The first two columns are features.
# Feel free to play with it by adding more features & examples.
# Interesting note: I've written this so the 2nd and 5th examples
# have the same features, but different labels - so we can see how the
# tree handles this case.

training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 2, 'Grape'],
    ['Red', 2, 'Grape'],
    ['Yellow', 3, 'Lemon'],
    ['Yellow', 3, 'Lemon']
]

# Returns unique values in column
def unique_vals(data, col):
    return set([row[col] for row in data])

"""Counts the number of each type of example in a dataset."""
# in our dataset format, the label is always the last column
# This will need modified to count the classes of the target data
def class_counts(data):
    counts = {}  # a dictionary of label -> count.
    for row in data:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)

class Question:
    """A Question is used to partition a dataset.
    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


def partition(rows, question):
    """Partitions a dataset.
    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    """Calculate the Gini Impurity for a list of rows.
    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
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


current_uncertainty = gini(training_data)
print(current_uncertainty)


# How much information do we gain by partioning on 'Green'?
true_rows, false_rows = partition(training_data, Question(0, 'Green'))
print(info_gain(true_rows, false_rows, current_uncertainty))
print(true_rows)
print(false_rows)

#What about if we partioned on 'Red' instead?
print()
true_rows, false_rows = partition(training_data, Question(0,'Red'))
print(info_gain(true_rows, false_rows, current_uncertainty))
print(true_rows)
print(false_rows)




def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns. This will change for us

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

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


best_gain, best_q = find_best_split(training_data)

print(best_gain)
print("col: %s, val: %s" % (best_q.column, best_q.value))
