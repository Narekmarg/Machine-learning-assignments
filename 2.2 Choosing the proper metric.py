import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import datasets
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

boston = datasets.load_boston()
objects = boston.data
answers = boston.target
objects = preprocessing.scale(objects)

partition_generator = KFold(len(objects), n_folds=5, random_state=42, shuffle=True)

best_p = 0
best_result = -100000000

for p in np.linspace(1, 11, 200):
    clf = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
    scores = cross_val_score(clf, objects, answers, 'mean_squared_error')
    score = scores.mean()
    if score > best_result:
        best_p = p
        best_result = score
print best_p
