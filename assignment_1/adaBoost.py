import numpy as np
import math
from helpers import draw, class_counts, normalize
from tree import Tree

class AdaBoost():
    """
    Boosting method that uses a number of weak classifiers in
    ensemble to make a strong classifier. This implementation uses decision
    stumps, which is a one level Decision Tree.
    """
    def __init__(self, n_stumps=100):
        self.n_stumps = n_stumps
        self.stumps_used = 0
        # array of alphas, one for each stump
        self.stumps = [None] * self.n_stumps
        self.alphas = [0] * self.n_stumps
        self.best_stump = None

    # Methods to match skilearn interface specification
    def get_params(self, deep=True):
        return {
            "n_stumps": self.n_stumps

        }

    # Methods to match skilearn interface specification
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self,x,y):
        n_samples, n_features = np.shape(x)

        # Initalise weights to normalised value
        self.weights = np.full(n_samples, (1/n_samples))
        # Iterate through classifiers
        for i in range(self.n_stumps):

            random_indexs = []
            for j in range(int(len(x))):
                random_indexs.append(draw(self.weights))

            stump = Tree(is_stump=True, indexs=random_indexs, stopping_criteria=1)

            stump.fit(x, y)


            # Use model to give us predictions
            predictions = stump.get_predictions(x)

            # Make sure that errors is the same length as y
            errors = np.zeros([len(y)])

            for j in range(len(random_indexs)):
                if predictions[random_indexs[j]] == y[ random_indexs[j]]:
                    errors[random_indexs[j]] = 1
                else:
                    errors[random_indexs[j]] = -1

            sum_error = sum(w for (z, w) in zip(errors, self.weights) if z < 0)
            new_alpha = 0.5 * math.log( ( 1 - sum_error) / (.0001 + sum_error))

            self.alphas[i] = new_alpha
            self.stumps[i] = stump
            self.stumps_used += 1
            if -1 not in errors: # This has got 100% training accuracy
                break

            self.weights = normalize( [ w * math.exp( -self.alphas[i] * p)
                               for (w, p) in zip(self.weights, errors)]) # <<<<



    def predict(self, x, y):

        #predictions = self.stumps[self.n_stumps - 1].get_predictions(x)

        alpha_sum = 0
        for i in range(self.stumps_used):
            #print(self.alphas[i])
            alpha_sum += self.alphas[i]

        all_results = []
        for i in range(self.stumps_used):
            pred_y = self.stumps[i].get_predictions(x)
            for j in range(len(pred_y)):
                if pred_y[j] == 0:
                    pred_y[j] = -1
                pred_y[j] *= self.alphas[i]/alpha_sum
            all_results.append(pred_y)

        col_totals = [sum(x) for x in zip(*all_results)]

        predictions = []
        for i in range(len(col_totals)):
            if(col_totals[i] > 0):
                predictions.append(1)
            else:
                predictions.append(0)

        #print(predictions)
        count_correct = 0
        for i in range(len(predictions)):
            if predictions[i] == y[i]:
                count_correct += 1

        return(count_correct / len(y))




    def score(self, x, y):
        return self.predict(x, y)
        #return self.stumps[self.n_stumps - 1].score(x, y)
