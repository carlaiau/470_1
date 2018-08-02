import numpy as np
from sklearn.datasets import load_iris

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
class Question:
    def __init__( self, feature, value ):
        self.feature = feature
        self.value = value
    def match( self, against ):
        return against[self.feature] >= self.value

    def __repr__(self):
        return "Is %s , >= %s?" % (
            self.feature, str(self.value))


"""
Partitions a dataset.
For each row in the dataset, check if it matches the question.
If so, add it to true_rows otherwise add it to false
"""
def partition( rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match( row ):
            true_rows.append( row )
        else:
            false_rows.append( row )
    return true_rows, false_rows

"""
Calculate the Gini for a list of rows.
"""
def gini( rows ):

    counts = class_counts( rows )
    impurity = 1
    for label in counts:
        prob_of_label = counts[label] / float( len( rows ) )
        impurity -= prob_of_label**2
    return impurity


"""
Information Gain.
The uncertainty of the starting node, minus the weighted impurity of
two child nodes.
"""
def info_gain( left, right, current_uncertainty):
    p = float( len( left ) ) / ( len( left ) + len( right ) )
    return current_uncertainty - p * gini( left) - ( 1 - p ) * gini( right )

"""
Find the best question to ask by iterating over every feature / value
and calculating the information gain. Inefficient AF
"""
def find_best_split( rows ):
    optimal_gain = 0  # keep track of the best information gain
    optimal_question = None  # keep train of the column / Value that produced best gain
    current_gini = gini( rows )

    # for each feature, col here equals 1 feature
    for col in range( len( rows[0] ) - 1 ):
        # unique values in the colum
        values = set( [row[col] for row in rows] )
        # for each value
        for val in values:
            q = Question( col, val )
            true_rows, false_rows = partition( rows, q )
            # If one branch has len 0, nothing was split.
            if len( true_rows ) == 0 or len( false_rows ) == 0:
                continue

            # Calculate the information gain from this split
            g = info_gain( true_rows, false_rows, current_gini )

            # If this gain is better than present best gain, record the gain, as well as the question
            if g >= optimal_gain:
                optimal_gain, optimal_question = g, q

    return optimal_gain, optimal_question


class Leaf:
    """A Leaf node classifies data.
    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    """A Decision Node asks a question.
    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    """Builds the tree.
    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)

"""
World's most elegant tree printing function.
"""
def print_tree(node, spacing=""):


    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "\t")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "\t")


"""
See the 'rules of recursion' above.
"""
def classify(row, node):


    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs



if __name__ == '__main__':

    """
    Create a new 2d numpy array.
    Append the target class, for ever array of data labels. This reduces the
    complexity when computing the gini index, as we don't need to make reference
    to the dataset.target array
    """
    dataset = load_iris()
    processed_data = np.empty( [len( dataset.data ), len( dataset.data[0] ) + 1] )
    for i in range( len( dataset.data ) ):
        processed_data[i] = np.append( dataset.data[i], dataset.target[i] )

    my_tree = build_tree(processed_data)

    print_tree(my_tree)
