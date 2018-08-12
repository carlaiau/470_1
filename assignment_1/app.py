import numpy as np
import time

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits

from tree import Tree
from randomForest import RandomForest
from adaBoost import AdaBoost
from helpers import make_two_class

def get_desired_dataset():
    dataset_selector = int(input(
    "\nPlease Choose a Sklearn classification dataset\n" +
    "1: Iris\n2: Cancer\n3: Wine\n4: Digits\n"
    ))
    return dataset_selector;

def get_desired_samples():
    samples_selector = int(input(
    "\nHow many samples of cross validation do you want to do?\n" +
    "Be mindful of my terribly inefficient Python abilities.\n"
    ))
    if samples_selector > 100:
        samples_selector = 100
        print("\nFor the love of god.\nSamples have been capped at 100!")
    return samples_selector


def manual_validation(type, dataset, samples, stopping_criterion=0, n_trees=10, max_features=None, n_stumps=100):
    """
    This is a manual version of the cross validation
    test. I'm using this so I can output both training
    and test error on a per iteration basis, used for
    the exploration functions
    """
    average_train = 0
    average_test  = 0
    average_time = 0
    print("\nTrain\tTest\tTrain Time")
    for i in range(samples):
        data = dataset.data
        target = dataset.target
        if type == "adaboost":
            data, target = make_two_class(data, target)
        x_train, x_test, y_train, y_test = train_test_split(
            data, target, test_size=0.2)
        if type == "tree":
            classifier = Tree()
        elif type == "randomforest":
            classifier = RandomForest(n_trees=n_trees, max_features=max_features)
        else:
            classifier = AdaBoost(n_stumps=n_stumps)

        tic = time.clock()
        if type == "tree":
            classifier.fit(x_train, y_train, stopping_criterion)
        else:
            classifier.fit(x_train, y_train)

        toc = time.clock()
        train_score = classifier.score(x_train, y_train)
        test_score = classifier.score(x_test, y_test)

        average_train += train_score
        average_test += test_score
        time_diff = toc - tic
        average_time += time_diff

        print("%0.2f\t%0.2f\t%0.4f" %
            (train_score * 100 , test_score * 100, time_diff ))

    total_toc = time.clock
    print("\n=== Averages ===")
    print("%0.2f\t%0.2f\t%0.4f" %
        (average_train*100/samples, average_test*100/samples, average_time / samples))

def run_cross_validation(name, classifier, x, y, samples):
    print("%s is running " % name)
    cv = ShuffleSplit(n_splits=samples, test_size=0.2)
    tic = time.clock()
    scores = cross_val_score(classifier, x, y, cv=cv)
    toc = time.clock()

    print("Accuracy: %0.2f (+/- %0.2f) Time: %0.4f (Each: %0.4f)\n" %
        (scores.mean(), scores.std() * 2, toc - tic, ( toc - tic) / samples) )


def compare(binary=0, dataset_index = 0, samples = 10):

    datasets = [
        load_iris(),
        load_breast_cancer(),
        load_wine(),
        load_digits()
    ]
    dataset = datasets[dataset_index]

    if binary == 1:
        print("Binary Comparison is only comparing between data in classes 0 and 1")
        x, y = make_two_class(dataset.data, dataset.target)
    else:
        x = dataset.data
        y = dataset.target

    #classifier = Tree()
    #name = "Tree"
    #run_cross_validation(name, classifier, x, y, samples)

    classifier = RandomForest()
    name = "RandomForest"
    run_cross_validation(name, classifier, x, y, samples)

    if binary == 1:
        classifier = AdaBoost()
        name = "AdaBoost"
        run_cross_validation(name, classifier, x, y, samples)

def decision_tree_explore():
    print("\nDecision Tree Explore!")
    had_enough = "n"
    while had_enough != "y" and had_enough != "Y":
        dataset_index = 0
        while dataset_index not in [1,2,3,4]:
            dataset_index = get_desired_dataset()

        dataset_index -= 1 # switch it back to array indexs
        datasets = [
            load_iris(),
            load_breast_cancer(),
            load_wine(),
            load_digits()
        ]
        dataset = datasets[dataset_index]

        samples_selector = get_desired_samples()
        stopping_criterion = 0
        _input = input(
            "\nDo you want to change the stopping Criterion?\n" +
            "This training dataset has n: " + str(int(len(dataset.data) * 0.8)) +
            ". Please Enter a new error\n" +
            "criterion or 'enter' for the default (n/10)\n"
        )
        if _input != "":
            stopping_criterion = int(_input)
        manual_validation("tree", dataset, samples_selector, stopping_criterion=stopping_criterion)
        had_enough = input("\nReturn to main? (y/n)\n")





def random_forest_explore():
    print("\nGoing for a walk in the FOREST!")
    had_enough = "n"
    while had_enough != "y" and had_enough != "Y":
        dataset_index = 0
        while dataset_index not in [1,2,3,4]:
            dataset_index = get_desired_dataset()
        dataset_index -= 1 # switch it back to array indexs
        datasets = [
            load_iris(),
            load_breast_cancer(),
            load_wine(),
            load_digits()
        ]
        dataset = datasets[dataset_index]
        samples_selector = get_desired_samples()
        n_trees = 10
        _input = input(
            "\nDo you want to change the number of trees in the forest?\n" +
            "Please Enter a number or 'enter' for the default (10)\n"
        )
        if _input != "":
            n_trees = int(_input)
        max_features = None
        _input = input(
            "\nDo you want to change the max features of each tree?\n" +
            "This dataset has " + str(np.shape(dataset.data)[1]) + " features\n"
            "Please Enter a number or enter for the default (sqrt(features))\n"
        )
        if _input != "":
            max_features = int(_input)
        manual_validation("randomforest", dataset, samples_selector, n_trees = n_trees, max_features = max_features)
        had_enough = input("\nReturn to main? (y/n)\n")

def adaboost_explore():
    print("\nArghhhh Dah Boost Explore!")

    had_enough = "n"
    while had_enough != "y" and had_enough != "Y":
        dataset_index = 0
        while dataset_index not in [1,2,3,4]:
            dataset_index = get_desired_dataset()
        dataset_index -= 1 # switch it back to array indexs
        datasets = [
            load_iris(),
            load_breast_cancer(),
            load_wine(),
            load_digits()
        ]
        dataset = datasets[dataset_index]
        samples_selector = get_desired_samples()
        n_stumps = 100
        _input = input(
            "\nDo you want to change the number of stumps to boost?\n" +
            "Please Enter a number or 'enter' for the default (100)\n"
        )
        if _input != "":
            n_stumps = int(_input)
        manual_validation("adaboost", dataset, samples_selector, n_stumps = n_stumps)
        had_enough = input("\nReturn to main? (y/n)\n")



if __name__ == '__main__':




    """
    I'm not checking for input errors here
    """
    what_we_gonna_do = 0;
    while what_we_gonna_do == 0:
        print("\n\n\n==============================================")
        print("Welcome to Carl's CLI, what do you want to do?\n")
        what_we_gonna_do = int(input(
        "1: Binary Compare all three ML methods\n" +
        "2: Non-Binary Compare between Decision Tree and Random Forest\n" +
        "3: Explore a certain ML Method\n"
        ))
        if what_we_gonna_do == 1 or what_we_gonna_do == 2:
            dataset_selector = 0
            while dataset_selector not in [1,2,3,4]:
                dataset_selector = get_desired_dataset()
            if(what_we_gonna_do == 1):
                compare(1, dataset_selector - 1, get_desired_samples())
            else:
                compare(0, dataset_selector - 1, get_desired_samples())

            what_we_gonna_do = 0

        elif what_we_gonna_do == 3:
            method_selector = 0
            while method_selector not in [1,2,3]:
                method_selector = int(input(
                    "\n1: Naked Decision Tree\n" +
                    "2: Random Forest\n" +
                    "3: AdaBoost\n"
                ))
                if method_selector == 1:
                    decision_tree_explore()
                elif method_selector == 2:
                    random_forest_explore()
                elif method_selector == 3:
                    adaboost_explore()
            what_we_gonna_do = 0
        else: # Default
            what_we_gonna_do = 0
    exit()
