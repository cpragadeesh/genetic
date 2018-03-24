import numpy as np
import csv
from sklearn import datasets, linear_model
from sklearn.neighbors import KNeighborsClassifier

from lib.genetic_selection import GeneticSelectionCV

def get_dataset_from_file(csvfile):

    dataset = list(csv.reader(open(csvfile)))
    dataset = np.array(dataset[1:])

    X = dataset[:, :-1].astype(float).tolist()
    # To convert column vector to 1d array
    y = np.ravel(dataset[:, -1:]) 

    return X, y

def main():

    np.set_printoptions(threshold=np.nan)

    X, y = get_dataset_from_file("./dataset/lung_info.csv")

    estimator = KNeighborsClassifier()

    selector = GeneticSelectionCV(estimator,
                                  cv=5,
                                  verbose=1,
                                  scoring="accuracy",
                                  n_population=50,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=40,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  caching=True,
                                  n_jobs=-1)
    selector = selector.fit(X, y)

    print(selector.support_)


if __name__ == "__main__":
    main()