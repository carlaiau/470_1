import numpy as np
from sklearn.datasets import load_iris


"""
Counts the number of label classes for an array of data.
Because the label is in the last index of every row, this is easy
"""
def class_counts( data ):
    counts = {}
    for row in data:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


"""
A Question is used to partition a dataset.
This class records the column/feature and the value that we are comparing with
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
A Leaf node classifies data.
This holds a dictionary of class (e.g., "Apple") -> number of times
it appears in the rows from the training data that reach this leaf.
"""
class Leaf:
    def __init__( self, rows ):
        self.predictions = class_counts( rows )

"""
A Decision Node asks a question.
We reference the question, the true branch,
and the false branch. Both branchs exist, otherwise
this Decision Node would be a
Leaf node as there is no further branching
"""
class Decision_Node:

    def __init__( self, question, true_branch, false_branch ):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


class Tree:

    def __init__ (self, dataset):
        processed_data = np.empty( [len( dataset.data ), len( dataset.data[0] ) + 1] )
        for i in range( len( dataset.data ) ):
            processed_data[i] = np.append( dataset.data[i], dataset.target[i] )
        self.data = processed_data
        self.root_node = None

    """
    Partitions a dataset.
    For each row in the dataset, check if it matches the question.
    If so, add it to true_rows otherwise add it to false_rows
    """
    def partition( self, rows, question):
        true_rows   = []
        false_rows  = []
        for row in rows:
            if question.match( row ):
                true_rows.append( row )
            else:
                false_rows.append( row )
        return true_rows, false_rows

    """
    Calculate the Gini for a list of rows.
    """
    def gini( self, rows ):

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
    def info_gain( self, left, right, current_uncertainty):
        prob_left = float( len( left ) ) / ( len( left ) + len( right ) )
        prob_right = 1 - prob_left
        info_gain = current_uncertainty;
        info_gain -= prob_left * self.gini( left )
        info_gain -= prob_right * self.gini ( right )
        return info_gain

    """
    Find the best question to ask by iterating over every feature / value
    and calculating the information gain. Inefficient AF
    """
    def find_best_split( self, rows ):
        most_gain = 0  # keep track of the best information gain
        best_question = None  # keep train of the column / Value that produced best gain
        current_gini = self.gini( rows )

        # for each feature, col here equals 1 feature
        for col in range( len( rows[0] ) - 1 ):
            # unique values in the colum
            values = set( [row[col] for row in rows] )
            # for each value
            for val in values:
                q = Question( col, val )
                true_rows, false_rows = self.partition( rows, q )

                # If one branch has len 0, nothing was split.
                if len( true_rows ) == 0 or len( false_rows ) == 0:
                    continue

                # Calculate the information gain from this split
                g = self.info_gain( true_rows, false_rows, current_gini )

                # If this gain is better than present best gain, record
                if g >= most_gain:
                    most_gain = g
                    best_question = q

        return most_gain, best_question

    """
    Start the intial building of the tree by
    with the class dataset but this cannot be called recrusively
    """
    def create(self):
        self.root_node = self.build(self.data)
        for i in range(len(self.data)):
            print(self.data[i])
    """
    Builds the tree.
    Recursive AF!
    """
    def build( self, rows ):

        # Attempt to partition dataset on each attribute.
        # Return the best question that gives us the most information gain
        gain, question = self.find_best_split( rows )

        # Base case: no further info gain
        # we'll return a leaf, and stop the recursion.
        if gain == 0:
            return Leaf( rows )

        true_rows, false_rows = self.partition( rows, question )

        # Build the true branch via recursion
        true_branch = self.build( true_rows )

        # Build the false branch via recursion
        false_branch = self.build( false_rows )

        # Return a Question node.
        # This records the best feature / value to ask at this point,
        # as well as the branches to follow
        # dependingo on the answer.
        return Decision_Node(question, true_branch, false_branch)


    def start_dump(self):
        self.dump(self.root_node, "")

    def dump( self, node, spacing=""):


        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            print (spacing + "Predict", node.predictions)
            return

        # Print the question at this node
        print (spacing + str(node.question))

        # Call this function recursively on the true branch
        print (spacing + '--> True:')
        self.dump(node.true_branch, spacing + "\t")

        # Call this function recursively on the false branch
        print (spacing + '--> False:')
        self.dump(node.false_branch, spacing + "\t")


    """
    See the 'rules of recursion' above.
    """
    def classify( self, row, node):


        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            return node.predictions

        # Decide whether to follow the true-branch or the false-branch.
        # Compare the feature / value stored in the node,
        # to the example we're considering.
        if node.question.match(row):
            return self.classify(row, node.true_branch)
        else:
            return self.classify(row, node.false_branch)


    def print_leaf(self, counts):
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

    my_tree = Tree(dataset)

    my_tree.create()

    my_tree.start_dump()
