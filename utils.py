from typing import List

import numpy as np
import copy


def mean_squared_error(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    mse = ((np.subtract(y_true, y_pred))**2).mean()
    return mse


def f1_score(real_labels, predicted_labels):
    assert len(real_labels) == len(predicted_labels)

    tp, fn, fp, tn = 0, 0, 0, 0

    for x, y in zip(real_labels, predicted_labels):
        if x == 1 and y == 1:
            tp += 1
        elif x == 1 and y == 0:
            fp += 1
        elif x == 0 and y == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp+fp != 0 else 0
    recall = tp / (tp + fn) if tp+fn != 0 else 0
    if not precision and not recall:
        return 0
    f1 = float((2*precision*recall) / (precision+recall))

    return f1 


def polynomial_features(features, k):
    new_features = []
    for i,feat in enumerate(features):
        temp = []
        for j in range(1,k+1):
            temp += [item**j for item in feat]
        new_features.append(temp)

    return new_features


def euclidean_distance(point1, point2):
    dis = 0
    for x,y in zip(point1,point2):
        dis += (x-y)**2
    return (dis)**0.5


def inner_product_distance(point1, point2):
    dis = 0
    for x,y in zip(point1,point2):
        dis += x*y
    return dis


def gaussian_kernel_distance(point1, point2):
    dis = 0
    for x,y in zip(point1,point2):
        dis += (x-y)**2
    return -np.exp((-1/2)*(dis))    


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features):
        new_features = [[0.0]*len(features[0]) for _ in range(len(features))]
        for i in range(len(features)):
            div = np.dot(features[i], features[i])**0.5 
            for j in range(len(features[0])):
                new_features[i][j] = features[i][j] / div if div != 0 else 0
        return new_features

class MinMaxScaler:
    def __init__(self):
        pass

    def __call__(self, features):

        minmax = []
        new_features = [[0.0]*len(features[0]) for _ in range(len(features))]
        for i in range(len(features[0])):
            col_values = [row[i] for row in features]
            minn, maxx = min(col_values), max(col_values)
            minmax.append([minn, maxx])

        for i,row in enumerate(features):
            for j in range(len(row)):
                new_features[i][j] = (row[j]-minmax[j][0]) / (minmax[j][1]-minmax[j][0])
        return new_features