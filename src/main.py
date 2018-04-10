import numpy as np
import csv
import collections

from sklearn import datasets, linear_model
from sklearn import svm
from sklearn.model_selection import check_cv
from sklearn.metrics.scorer import check_scoring, make_scorer
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import is_classifier

from lib.genetic_selection import GeneticSelectionCV

K_FOLD_CROSS_VALIDATION = 3
TRAINING_DATASET_LOCATION = "./dataset/ovarian_info.csv"
RAW_DATASET_LOCATION = "./dataset/ovarian_norm_random.csv"
THRESHOLD_DELTA = 0.5 # Needs to be changed appropriately on loading dataset
THRESHOLD_ALPHA = 0

def get_dataset_from_file(csvfile, leave_header_row=True):

    global THRESHOLD_ALPHA
    global THRESHOLD_DELTA

    dataset = list(csv.reader(open(csvfile)))
    dataset = np.array(dataset)

    if leave_header_row:
        dataset = dataset[1:]

    X = dataset[:, :-1].astype(float)
    # To convert column vector to 1d array
    y = np.ravel(dataset[:, -1:]) 


    count_dict = collections.Counter(y)

    majority_class_count = max(count_dict[count_dict.keys()[0]], count_dict[count_dict.keys()[1]])
    minority_class_count = min(count_dict[count_dict.keys()[0]], count_dict[count_dict.keys()[1]])

    diff_class_count = minority_class_count - majority_class_count
    sum_class_count = minority_class_count + majority_class_count

    THRESHOLD_DELTA = diff_class_count / float(sum_class_count + 2 * THRESHOLD_ALPHA)

    print "Loaded %d samples with %d features" % (X.shape[0], X.shape[1], )
    print "Threshold offset: %.2f" % (THRESHOLD_DELTA, )

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

    
def custom_scorer(estimator, X, y): 

    global THRESHOLD_DELTA
    
    predicted_proba = estimator.predict_proba(X)
    labels_in_sorted_order = sorted(set(y))
    threshold = 0.5 + THRESHOLD_DELTA
    correct_count = 0
    for index, sample in enumerate(predicted_proba):
        label_index = 0 if sample[0] >= threshold else 1
        if y[index] == labels_in_sorted_order[label_index]:
            correct_count = correct_count + 1

    accuracy = correct_count / float(len(y))

    return accuracy


def main():

    np.set_printoptions(threshold=np.nan)

    estimator = svm.SVC(probability=True)
    pre_filter_stats(estimator)

    X, y = get_dataset_from_file(TRAINING_DATASET_LOCATION)
    pre_GA(estimator, X, y)

    # GA start
    selector = GeneticSelectionCV(estimator,
                                  cv=K_FOLD_CROSS_VALIDATION,
                                  verbose=1,
                                  scoring="accuracy",
                                  n_population=7,
                                  crossover_proba=0.95,
                                  mutation_proba=0.01,
                                  n_generations=40,
                                  tournament_size=2,
                                  caching=True,
                                  n_jobs=8,
                                  hall_of_fame_size=1,
                                  please_kill_yourself_count=30,
                                  micro_ga=False)

    selector = selector.fit(X, y)

    # Post GA statistics

    print "Number of features selected: " + str(collections.Counter(selector.support_)[True])
    
    X_selected = X[:, np.array(selector.support_)]
    post_GA(estimator, X_selected, y)


if __name__ == "__main__":
    main()