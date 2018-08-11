import numpy as np


class KMeans():
    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        centroids = np.zeros((self.n_cluster, D))
        for i, num in enumerate(np.random.choice(N, self.n_cluster, replace=False)):
            centroids[i] = x[num]
        memberships = np.zeros(N,)
        distor = float('inf')
        n_iter = 0

        for i in range(self.max_iter):
            n_iter = i+1
            distor_new = 0
            sum_rik = np.zeros((self.n_cluster,1))
            sum_rik_ni = np.zeros((self.n_cluster, D))

            for j, row in enumerate(x):
                sub = centroids - row
                dot = np.sum(np.multiply(sub, sub), axis=1)
                k = memberships[j] = int(np.argmin(dot))
                distor_new += dot[k]
                sum_rik[k] += 1
                sum_rik_ni[k] += row

            distor_new /= N
            if abs(distor - distor_new) <= self.e:
                break

            distor = distor_new

            centroids = np.divide(sum_rik_ni, sum_rik)

        return (centroids, memberships, n_iter)

class KMeansClassifier():
    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        
        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)

        k_means = KMeans(self.n_cluster, self.max_iter, self.e)
        centroids, memberships, i = k_means.fit(x)
        centroid_labels = np.zeros((self.n_cluster, D))

        for i, m in enumerate(memberships):
            centroid_labels[int(m)][y[i]]+=1
        centroid_labels = np.argmax(centroid_labels, axis=1)

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape

        p_label = np.zeros(N,)
        for i, row in enumerate(x):
            sub = self.centroids - row
            k = int(np.argmin(np.sum(np.multiply(sub, sub), axis=1)))
            p_label[i] = self.centroid_labels[k]
        return p_label
