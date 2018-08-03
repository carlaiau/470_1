import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


def class_counts( data ):
    """
    Counts the number of label classes for an array of data.
    Because the label is in the last index of every row, this is easy
    """
    counts = {}
    for row in data:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

class Question:
    """
    A Question is used to partition a dataset.
    This class records the column/feature and the value that we are comparing with
    """
    def __init__( self, feature, value ):
        self.feature = feature
        self.value = value

    def match( self, against ):
        return against[self.feature] >= self.value

    def __repr__(self):
        return "Is %s , >= %s?" % (
            self.feature, str(self.value))

class Node:
    """
    A Leaf node classifies data.
    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.

    A Decision Node asks a question.
    We reference the question, the true branch,
    and the false branch. Both branchs exist, otherwise
    this Decision Node would be a
    Leaf node as there is no further branching
    """
    def __init__( self, is_decision = 0, rows = None, question = None, true_branch = None, false_branch = None):
        if is_decision is 1:
            self.is_decision = 1
            self.question = question
            self.true_branch = true_branch
            self.false_branch = false_branch
        else:
            self.is_decision = 0;
            self.predictions = class_counts( rows )

    def fancy_print(self):
        if self.is_decision is 0:
            print(self.predictions)

    def top_pick(self):
        if self.is_decision is 0:
            best_class = 0;
            best_number = 0
            for pred_class, pred_number in self.predictions.items():
                if pred_number > best_number:
                    best_class = pred_class
                    best_number = pred_number
            return best_class

class Tree:

    def partition( self, rows, question):
        """
        Partitions a dataset.
        For each row in the dataset, check if it matches the question.
        If so, add it to true_rows otherwise add it to false_rows
        """
        true_rows   = []
        false_rows  = []
        for row in rows:
            if question.match( row ):
                true_rows.append( row )
            else:
                false_rows.append( row )
        return true_rows, false_rows

    def gini( self, rows ):
        """
        Calculate the Gini for a list of rows.
        """
        counts = class_counts( rows )
        impurity = 1
        for label in counts:
            prob_of_label = counts[label] / float( len( rows ) )
            impurity -= prob_of_label**2
        return impurity

    def info_gain( self, left, right, current_uncertainty):
        """
        Information Gain.
        The uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        """
        prob_left = float( len( left ) ) / ( len( left ) + len( right ) )
        prob_right = 1 - prob_left
        info_gain = current_uncertainty;
        info_gain -= prob_left * self.gini( left )
        info_gain -= prob_right * self.gini ( right )
        return info_gain

    def find_best_split( self, rows ):
        """
        Find the best question to ask by iterating over every feature / value
        and calculating the information gain. Inefficient AF
        """
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

                # If one branch has length less than our stopping_criteria then no split
                if len( true_rows ) < self.stopping_criteria or len( false_rows ) < self.stopping_criteria:
                    continue

                # Calculate the information gain from this split
                g = self.info_gain( true_rows, false_rows, current_gini )

                # If this gain is better than present best gain, record
                if g >= most_gain:
                    most_gain = g
                    best_question = q

        return most_gain, best_question

    def fit(self, x, y):
        """
        Importing training data and setting stopping criteria

        Create a new 2d numpy array.
        Append the target class, for ever array of data labels. This reduces the
        complexity when computing the gini index, as we don't need to make reference
        to the y array

        Start the intial building of the tree via recursive building,
        """
        merged_train = np.empty( [len( x ), len( x[0] ) + 1] )
        for i in range( len( x ) ):
            merged_train[i] = np.append( x[i], y[i] )
        self.data = merged_train
        self.stopping_criteria = len( self.data) / 20 # 10% of dataset
        self.root_node = self.build(self.data)

    def build( self, rows ):
        """
        Builds the tree.
        Recursive AF!
        """

        # Attempt to partition dataset on each attribute.
        # Return the best question that gives us the most information gain
        gain, question = self.find_best_split( rows )

        # Base case: no further info gain
        # we'll return a leaf, and stop the recursion.
        if gain == 0:
            return Node(0, rows)

        true_rows, false_rows = self.partition( rows, question )

        # Build the true branch via recursion
        true_branch = self.build( true_rows )

        # Build the false branch via recursion
        false_branch = self.build( false_rows )

        # Return a Question node.
        # This records the best feature / value to ask at this point,
        # as well as the branches to follow
        # dependingo on the answer.
        return Node(1, None, question, true_branch, false_branch)

    def predict( self, x, y):
        n = len(x)
        correct = 0;
        for i in range(n):
            if self.classify(x[i], self.root_node) == y[i]:
                correct += 1
        print("N: %s\tC: %s\t%.2f%%" % (n, correct, (correct/n * 100)))

    def classify( self, row, node):

        # Base Case: Leaf!
        if node.is_decision is 0:
            return node.top_pick()

        # Decide whether to follow the true-branch or the false-branch.
        # Compare the feature / value stored in the node,
        # to the example we're considering.
        if node.question.match(row):
            return self.classify(row, node.true_branch)
        else:
            return self.classify(row, node.false_branch)


    # Printing functions used for debuging
    def debug_output( self ):
        self.dump( self.root_node, "" )

    def dump( self, node, spacing="" ):

        # Base Case: Leaf!
        if node.is_decision is 0:
            print ( spacing + "Predict", node.predictions )
            return

        # Print the question at this node
        print (spacing + str( node.question))

        # Call this function recursively on the true branch
        print (spacing + '--> True:')
        self.dump(node.true_branch, spacing + "\t")

        # Call this function recursively on the false branch
        print (spacing + '--> False:')
        self.dump(node.false_branch, spacing + "\t")


if __name__ == '__main__':
    iris = load_iris()
    for i in range(100):
        x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

        clf = Tree()
        clf.fit(x_train, y_train)
        clf.predict(x_test, y_test)
    #scores = cross_val_score(my_tree, dataset.data, dataset.target, cv=5)
    #print(scores)
