import numpy as np
import csv
import collections

from sklearn import datasets, linear_model
from sklearn import svm
from sklearn.model_selection import check_cv
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import is_classifier

from lib.genetic_selection import GeneticSelectionCV

K_FOLD_CROSS_VALIDATION = 3
TRAINING_DATASET_LOCATION = "./dataset/ovarian_info.csv"
RAW_DATASET_LOCATION = "./dataset/ovarian_norm_random.csv"

def get_dataset_from_file(csvfile, leave_header_row=True):

    dataset = list(csv.reader(open(csvfile)))
    dataset = np.array(dataset)

    if leave_header_row:
        dataset = dataset[1:]

    X = dataset[:, :-1].astype(float)
    # To convert column vector to 1d array
    y = np.ravel(dataset[:, -1:]) 

    print "Loaded %d samples with %d features" % (X.shape[0], X.shape[1], )
    return X, y


def fit_and_get_score(estimator, X, y):
    cv = check_cv(cv=K_FOLD_CROSS_VALIDATION, y=y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=None)

    scores = []
    for train, test in cv.split(X, y):
        score = _fit_and_score(estimator=estimator, X=X, y=y, scorer=scorer,
                               train=train, test=test, verbose=0, parameters=None,
                               fit_params=None)
        scores.append(score)

    scores_mean = np.mean(scores)

    return scores_mean


def pre_GA(estimator, X, y):
    
    score = fit_and_get_score(estimator, X, y)
    print "Fitness score before applying GA: ",
    print score

def post_GA(estimator, X_selected, y):

    score = fit_and_get_score(estimator, X_selected, y)
    print "Fitness score after applying GA: ",
    print score


def pre_filter_stats(estimator):
    X, y = get_dataset_from_file(RAW_DATASET_LOCATION)
    score = fit_and_get_score(estimator, X, y)
    print "Fitness score before applying filter: ",
    print score


def test():
    
    X, y = get_dataset_from_file(TRAINING_DATASET_LOCATION)
    estimator = svm.SVC()

    estimator.fit(X, y)

    X_test, y_test = get_dataset_from_file("./dataset/test.csv", False)
    print y_test
    print estimator.predict(X_test[0])
    print estimator.predict(X_test[1])
    

def main():

    np.set_printoptions(threshold=np.nan)
    #test()
    estimator = svm.SVC()

    pre_filter_stats(estimator)

    X, y = get_dataset_from_file(TRAINING_DATASET_LOCATION)
    pre_GA(estimator, X, y)

    # GA start

    selector = GeneticSelectionCV(estimator,
                                  cv=K_FOLD_CROSS_VALIDATION,
                                  verbose=1,
                                  scoring="accuracy",
                                  n_population=50,
                                  crossover_proba=0.8,
                                  mutation_proba=0.1,
                                  n_generations=40,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  caching=True,
                                  n_jobs=-1)
    selector = selector.fit(X, y)

    # Post GA statistics

    print "Number of features selected: " + str(collections.Counter(selector.support_)[True])
    
    X_selected = X[:, np.array(selector.support_)]
    post_GA(estimator, X_selected, y)


if __name__ == "__main__":
    main()