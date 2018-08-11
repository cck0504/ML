import numpy as np
from kmeans import KMeans


class GMM():
    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):

            k_means = KMeans(self.n_cluster, self.max_iter, self.e)
            centroids, memberships, i = k_means.fit(x)

            sum_gamma = np.zeros(self.n_cluster,)
            sum_xi = np.zeros((self.n_cluster, D))
            sum_covar = np.zeros((self.n_cluster, D, D))
            covars = np.zeros((self.n_cluster, D, D))
            pi = np.zeros(self.n_cluster,)

            for j in range(N):
                k = int(memberships[j])
                sum_gamma[k] += 1
                sum_covar[k] += np.dot(np.matrix(x[j]-centroids[k]).T,np.matrix(x[j]-centroids[k]))
            for j in range(self.n_cluster):
                covars[j] = sum_covar[j] / sum_gamma[j]
            pi = sum_gamma / N

        elif (self.init == 'random'):

            centroids = np.zeros((self.n_cluster, D))
            for i in range(self.n_cluster):
                for j in range(D):
                    centroids[i][j] = np.random.rand()
            covars = np.array([np.identity(D).tolist() for _ in range(self.n_cluster)])
            pi = np.ones(self.n_cluster,) / self.n_cluster

        else:
            raise Exception('Invalid initialization provided')

        n_iter = 0
        gamma = np.zeros((N, self.n_cluster))
        self.means = centroids
        self.variances = covars
        self.pi_k = pi   
        eps = 0.001*np.identity(D)

        log_likelihood = self.compute_log_likelihood(x)

        for t in range(self.max_iter):
            n_iter += 1
            sum_gamma = np.zeros(self.n_cluster,)
            sum_xi = np.zeros((self.n_cluster, D))
            sum_covar = np.zeros((self.n_cluster, D, D))

            for i in range(N):
                px_sum = 0
                normal_box = np.zeros((N, self.n_cluster))
                subt = np.matrix(x[i]-centroids)
                for j in range(self.n_cluster):
                    exponential = np.exp((-1/2)*subt[j]*np.linalg.inv(covars[j]+eps)*subt[j].T).tolist()[0][0]
                    normal = (1 / np.sqrt(np.power((2*np.pi),D)*np.linalg.det(covars[j]+eps)))*exponential
                    normal_box[i][j] = normal

                gamma[i] = np.multiply(pi, normal_box[i])
                px_sum += np.sum(gamma[i])

                gamma[i] /=  px_sum
                sum_gamma = np.add(sum_gamma, gamma[i])
                sum_xi += np.multiply(np.matrix(gamma[i]).T, x[i])

                for j in range(self.n_cluster):
                    sum_covar[j] += gamma[i][j]*np.dot(subt[j].T, subt[j])

            for j in range(self.n_cluster):
                centroids[j] = sum_xi[j] / sum_gamma[j]
                covars[j] = sum_covar[j] / sum_gamma[j]

            pi = sum_gamma / N

            self.means = centroids
            self.variances = covars
            self.pi_k = pi   

            log_likelihood_new = self.compute_log_likelihood(x)
            if abs(log_likelihood - log_likelihood_new) <= self.e:
                break
            log_likelihood = log_likelihood_new

        return n_iter

    def sample(self, N):

        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        X, D = self.means.shape
        samples = np.zeros((N,D))
        for n in range(N):
            multi = np.argmax(np.random.multinomial(1, self.pi_k))
            sample = np.random.multivariate_normal(self.means[multi], self.variances[multi])
            samples[n] = sample
        return samples

    def compute_log_likelihood(self, x):

        assert len(x.shape) == 2,  'x can only be 2 dimensional'

        N, D = x.shape
        log_likelihood = 0
        eps = 0.001*np.identity(D)
        for i in range(N):
            px = 0
            for j in range(self.n_cluster):
                subt = np.matrix(x[i]-self.means[j])
                exponential = np.exp((-1/2)*subt*np.linalg.inv(self.variances[j]+eps)*subt.T).tolist()[0][0]
                normal = (1 / np.sqrt(np.power((2*np.pi),D)*np.linalg.det(self.variances[j]+eps)))*exponential
                px += self.pi_k[j]*normal
            log_likelihood += np.log(px)
        return float(log_likelihood)
