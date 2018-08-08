import random
import boosting
from utils import sign
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def simpleTest():

   def target(x):
      if x[2] > 0.5 or x[3] > 0.5:
         return 1 if random.random() > 0.05 else -1
      return -1


   examples = [[random.random() for _ in range(10)] for _ in range(1000)]
   labels = [target(x) for x in examples]
   trainingData = list(zip(examples, labels))

   testData = [[random.random() for _ in range(10)] for _ in range(1000)]
   testLabels = [target(x) for x in testData]


   def testCoordinate(samples, j):
       values = [sign(x[j] - 0.5) * y for (x,y) in samples]
       return len([z for z in values if z > 0]) / len(values)


   def bestCoordinate(samples, n):
       return max(range(n), key=lambda j: testCoordinate(samples, j))


   # find the single coordinate and a threshold value that works best
   def singleCoordinateLearner(drawExample):
       samples = [drawExample() for _ in range(100)]
       n = len(samples[0][0])

       j = bestCoordinate(samples, n)
       return lambda x: x[j] > 0.5


   finalH, finalDistr = boosting.boost(trainingData, singleCoordinateLearner, 100)

   finalError = len([x for x in testData if finalH(x) != target(x)]) / len(testData)
   print(finalError)



def error(h, data):
   return sum(1 for x,y in data if h(x) != y) / len(data)


def runAdult():
   from data import adult
   from decisionstump import buildDecisionStump
   train, test = adult.load()

   print(test)
   """
   weakLearner = buildDecisionStump
   rounds = 20

   h = boosting.boost(train, weakLearner, rounds)
   print("Training error: %G" % error(h, train))
   print("Test error: %G" % error(h, test))
   """

def carlsTest():
    import numpy as np
    from decisionstump import buildDecisionStump
    dataset = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.20)

    # Make this into a two class problem
    train = []
    for i in range(len(y_train)):
        if y_train[i] == 0:
            train.append([x_train[i], -1])
        elif y_train[i] == 1:
            train.append([x_train[i], 1])

    test = []
    for i in range(len(y_test)):
        if y_test[i] == 0 or y_test[i] == 1:
            test.append([x_test[i], y_test[i]])

    weakLearner = buildDecisionStump
    rounds = 20
    h = boosting.boost(train, weakLearner, rounds)
    print("Training error: %G" % error(h, train))
    print("Test error: %G" % error(h, test))


if __name__ == "__main__":
   runAdult()
