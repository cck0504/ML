from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt
import random

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        self.nb_features = nb_features
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features, labels):
        e = 0.00000001
        for times in range(self.max_iteration):
            flag = True
            div = np.linalg.norm(self.w) + e
            rand_index = list(range(len(features)))
            np.random.shuffle(rand_index)
            for i in range(len(rand_index)):
                cur_margin = np.dot(self.w, features[rand_index[i]]) / div
                if (labels[rand_index[i]] * np.dot(self.w, features[rand_index[i]]) < 0 or (-self.margin/2 < cur_margin < self.margin/2)): #
                    flag = False
                    self.w = np.add(self.w, labels[rand_index[i]] * np.array(features[rand_index[i]]))
            if flag == True:
                return True
        return False

    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        w = np.array(self.w)
        prod = np.dot(features, w.T)
        y_pred = prod.flatten().tolist()
        for i in range(len(y_pred)):
            y_pred[i] = 1 if y_pred[i] > 0 else -1
        return y_pred

    def get_weights(self):
        return self.w
    