import pandas as pd
import numpy as np
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing

data = pd.read_csv("wine.data")
answers = data[data.columns[0]]
objects = data[data.columns[1:]]

partition_generator = KFold(len(data.index), n_folds=5, random_state=42, shuffle=True)


def cross_validate(objects, answers, partition_generator):
    best_k = 1
    best_val = 0
    for k in range(1, 51):
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, objects, answers, None, partition_generator)
        score = scores.mean()
        if score > best_val:
            best_k = k
            best_val = score
    return best_k, best_val

print cross_validate(objects, answers, partition_generator)

normalized = objects.apply(lambda x: preprocessing.scale(x))

print cross_validate(normalized, answers, partition_generator)