from __future__ import division, print_function

from typing import List

import numpy
import scipy

class LinearRegression:
    def __init__(self, nb_features):
        self.nb_features = nb_features

    def train(self, features, values):
        self.X = numpy.append(numpy.ones((len(features), 1)),numpy.mat(features), axis=1)
        self.y = numpy.transpose(numpy.mat(values))
        self.get_weights()

    def predict(self, features):
        w = numpy.array(self.W)
        new_features = numpy.append(numpy.ones((len(features), 1)),numpy.mat(features), axis=1)
        prod = numpy.dot(new_features, w)
        y_pred = prod.flatten().tolist()[0]
        return y_pred

    def get_weights(self):
        self.W = numpy.mat([0.0 for _ in range(len(self.X)+1)])
        self.W = numpy.matrix.dot(numpy.linalg.pinv(numpy.matrix.dot(numpy.transpose(self.X), self.X)), numpy.matrix.dot(numpy.transpose(self.X),self.y))
        return self.W


class LinearRegressionWithL2Loss:
    def __init__(self, nb_features, alpha):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features, values):
        self.X = numpy.append(numpy.ones((len(features), 1)),numpy.mat(features), axis=1)
        self.y = numpy.transpose(numpy.mat(values))
        self.get_weights()

    def predict(self, features):
        w = numpy.array(self.W)
        new_features = numpy.append(numpy.ones((len(features), 1)),numpy.mat(features), axis=1)
        prod = numpy.dot(new_features, w)
        y_pred = prod.flatten().tolist()[0]
        return y_pred

    def get_weights(self):
        self.W = numpy.mat([0.0 for _ in range(len(self.X)+1)])
        identity = numpy.identity(len(numpy.matrix.dot(numpy.transpose(self.X), self.X)))
        inverse = numpy.linalg.pinv(numpy.add(numpy.matrix.dot(numpy.transpose(self.X), self.X), self.alpha*identity))
        self.W = numpy.matrix.dot(inverse, numpy.matrix.dot(numpy.transpose(self.X),self.y))
        return self.W
        

if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
