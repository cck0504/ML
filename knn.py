from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy

class KNN:

    def __init__(self, k, distance_function):
        self.k = k
        self.distance_function = distance_function

    def train(self, features, labels):
    	self.x_features = features
    	self.y_labels = labels

    def predict(self, features):
    	res = []
    	for feat in features:
    		distances = []
    		for i,x_feat in enumerate(self.x_features):
    			dist = self.distance_function(feat, x_feat)
    			distances.append((x_feat, dist))
    		distances.sort(key=lambda x:x[1])
    		neighbors = []
    		for i in range(self.k):
    			neighbors.append(distances[i][0])
    		temp = []
    		for i in neighbors:
    			temp.append(self.y_labels[self.x_features.index(i)])
    		count = {}
    		for item in temp:
    			count[item] = count.get(item, 0) + 1
    		count = sorted(count.items(), key=lambda x:x[1], reverse=True)
    		res.append(count[0][0])
    	return res


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
