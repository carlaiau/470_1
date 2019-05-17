import numpy as np
from tree import Tree

class RandomForest():

    """
    Random Forest.
    Uses an ensemble of decision trees trained on random subset of features with
    a random subset of the data
    """
    def __init__( self, n_trees=10, max_features=None, stopping_criteria = 1):
        self.n_trees = n_trees
        self.max_features = max_features

        # this is to remove the samples/10 default stopping_criteria of the trees
        self.stopping_criteria = stopping_criteria

        # Initialize decision trees
        self.the_forest = []
        for tree in range(n_trees):
            self.the_forest.append(Tree())


    # Methods to match skilearn interface specification
    def get_params(self, deep=True):
        return {
            "n_trees": self.n_trees,
            "max_features": self.max_features,
            "stopping_criteria": self.stopping_criteria
        }

    # Methods to match skilearn interface specification
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


    def fit( self, x, y ):
        n_features = np.shape(x)[1]
        if not self.max_features:
            self.max_features = int( np.sqrt( n_features ) )


        # Choose one random subset of the data for each tree
        subsets = []
        n_samples = np.shape(x)[0]
        x_y = np.concatenate((x, y.reshape((1, len(y))).T), axis=1)
        np.random.shuffle(x_y)
        # Uses 50% of training samples without replacements
        subsample_size = int(n_samples / 2)
        for _ in range(self.n_trees):
            x_indice = np.random.choice(
                range(n_samples), size=np.shape(range(subsample_size)), replace=False
            )
            x = x_y[x_indice][:, :-1]
            y = x_y[x_indice][:, -1]
            subsets.append([x, y])

        for i in range( self.n_trees ):
            x_subset, y_subset = subsets[i]

            # Feature bagging (select random subsets of the features)
            idx = np.random.choice( range( n_features ), size=self.max_features, replace=True )

            # Save the indices of the features for prediction
            self.the_forest[i].feature_indices = idx

            # Choose the features corresponding to the indices
            x_subset = x_subset[:, idx]

            # Fit the tree to the data
            self.the_forest[i].fit( x_subset, y_subset, stopping_criteria = self.stopping_criteria )


    def predict( self, x):
        # Height number of samples, width number of trees
        predictions = np.empty( ( x.shape[0], len( self.the_forest ) ) )

        # Let each tree make a prediction on the data
        for i, tree in enumerate( self.the_forest ):
            # Indices of the features that the tree has trained on
            idx = tree.feature_indices

            prediction = tree.get_predictions( x[:, idx] )

            predictions[:, i] = prediction

        predictions = np.array(predictions)



        top_voted = np.empty( predictions.shape[0] )

        # We vote!
        for i in range( len( predictions ) ):
            votes = np.unique(predictions[i], return_counts=True )
            if len( votes[0] ) == 1: # all Trees agree on one class
                top_class = votes[0][0];
            else:
                top_class_votes = 0
                top_class_index = 0
                for j in range( len( votes[1] ) ):
                    if votes[1][j] > top_class_votes:
                        top_class_votes = votes[1][j]
                        top_class_index = j

                top_class = votes[0][top_class_index]
            top_voted[i] = top_class

        return top_voted




    def score( self, x, y):
        predictions = self.predict( x )
        n = len(x)
        correct = 0;
        for i in range(n):
            if predictions[i] == y[i]:
                correct += 1

        return correct / n
