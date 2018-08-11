import numpy as np

def pca(X = np.array([]), no_dims = 50):
    Y = np.array([])
    M = np.array([])

    N, D = X.shape

    cov_mat = np.cov(X.T)

    eig_val, eig_vec = np.linalg.eig(cov_mat)

    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(D)]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    M = eig_pairs[0][1].reshape(D,1)
    for i in range(1, no_dims):
        M = np.hstack((M, eig_pairs[i][1].reshape(D,1)))

    Y = np.dot(X, M)


    return Y, M

def decompress(Y = np.array([]), M = np.array([])):
    X_hat = np.array([])
    
    X_hat = np.dot(Y, M.T)

    return X_hat

def reconstruction_error(orig = np.array([]), decompressed = np.array([])):
    error = 0

    error = np.mean((np.power((orig-decompressed), 2)), axis=0)

    return error

def load_data(dataset='mnist_subset.json'):
    import json


    with open(dataset, 'r') as f:
        data_set = json.load(f)
    mnist = np.vstack((np.asarray(data_set['train'][0]), 
                    np.asarray(data_set['valid'][0]), 
                    np.asarray(data_set['test'][0])))
    return mnist

if __name__ == '__main__':
    
    import argparse
    import sys


    mnist = load_data()
    compression_rates = [2, 10, 50, 100, 250, 500]
    with open('pca_output.txt', 'w') as f:
        for cr in compression_rates:
            Y, M = pca(mnist - np.mean(mnist, axis=0), cr)
            
            decompressed_mnist = decompress(Y, M)
            decompressed_mnist += np.mean(mnist, axis=0)
            
            total_error = 0.
            for mi, di in zip(mnist, decompressed_mnist):
                error = reconstruction_error(mi, di)
                f.write(str(error))
                f.write('\n')
                total_error += error
            print('Total reconstruction error after compression with %d principal '\
                'components is %f' % (cr, total_error))



