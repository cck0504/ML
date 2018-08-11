from __future__ import division, print_function

import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from matplotlib import cm

def binary_train(X, y, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    one = np.matrix(np.ones(N)).T
    X1 = np.append(one,X, axis=1)
    const_max = 0
    w1 = np.append(b,w)
    for _ in range(max_iterations):
        wTx = np.dot(w1, X1.T)
        def apply_sig(a):
            return sigmoid(a)
        sig_vec = np.vectorize(apply_sig)
        sig_wTx = sig_vec(wTx)
        sig_y = np.subtract(sig_wTx, y)
        sigma = np.dot(sig_y, X1) / N
        if np.any(sigma) == False:
            print(np.all(sigma), sigma)
            print("iteration: ", _)
            break

        w1 = np.subtract(w1, step_size*sigma)
    
    w1 = w1.tolist()[0]
    b,w = np.array(w1[:1]), np.array(w1[1:])

    assert w.shape == (D,)
    return w, b


def binary_predict(X, w, b):
    N, D = X.shape
    preds = np.zeros(N) 
     
    one = np.matrix(np.ones(N)).T
    X1 = np.append(one,X, axis=1)
    w1 = np.append(b,w)
    wTx = np.dot(w1, X1.T)  

    def apply_sig(a):
        return 1.0 if sigmoid(a) >= 0.5 else 0.0
    sig_vec = np.vectorize(apply_sig)
    preds = np.add(preds, sig_vec(wTx).tolist()[0])

    assert preds.shape == (N,) 
    return preds


def multinomial_train(X, y, C, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    one = np.matrix(np.ones(N)).T
    X1 = np.append(one,X, axis=1)
    w1 = np.append(np.matrix(b).T,w, axis=1)
    
    for _ in range(max_iterations):
        wTx = np.exp(np.dot(w1, X1.T))
        col_sum = np.zeros(N)
        for n in range(N):
            col_sum[n] = np.sum(wTx[:, n])
        for c in range(C):
            yc = [1.0 if yi == c else 0.0 for yi in y]
            wTx_c = wTx[[c],:]   

            def apply_softmax(a, b):
                a = a / b
                return a
            softmax_vec = np.vectorize(apply_softmax)
            soft_wTx_c = softmax_vec(wTx_c, col_sum)

            sig_y = np.subtract(soft_wTx_c, yc)
            sigma = np.dot(sig_y, X1) / N

            if np.any(sigma) == False:
                print(np.all(sigma), sigma)
                print("iteration: ", _)
                break
            w1[[c],:] = np.subtract(w1[[c],:], step_size*sigma)
        
    b = np.array(w1.T[0].tolist()[0])
    w = np.delete(w1, 0, 1)

    assert w.shape == (C, D)
    assert b.shape == (C,)
    return w, b


def multinomial_predict(X, w, b):
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N) 
  
    one = np.matrix(np.ones(N)).T
    X1 = np.append(one,X, axis=1)
    w1 = np.append(np.matrix(b).T,w, axis=1)

    wTx = np.exp(np.dot(w1, X1.T)) 
    def apply_softmax(a, b):
        a = a / b
        return a
    softmax_vec = np.vectorize(apply_softmax)
    pred_set = np.matrix(np.zeros((C, N)))

    for n in range(N):
        col_sum = np.sum(wTx[:,n])
        pred_set[:,n] = softmax_vec(wTx[:,n], col_sum)

    preds = np.array(np.argmax(pred_set, axis=0).tolist()[0])

    assert preds.shape == (N,)
    return preds


def OVR_train(X, y, C, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    N, D = X.shape
    
    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    one = np.matrix(np.ones(N)).T
    X1 = np.append(one,X, axis=1)
    w1 = np.append(np.matrix(b).T,w, axis=1)
    
    for _ in range(max_iterations):
        wTx = np.dot(w1, X1.T)
        for c in range(C):
            yc = [1.0 if yi == c else 0.0 for yi in y]
            wTx_c = wTx[[c],:]            
            def apply_sig(a):
                return sigmoid(a)
            sig_vec = np.vectorize(apply_sig)
            sig_wTx_c = sig_vec(wTx_c)
            sig_y = np.subtract(sig_wTx_c, yc)
            sigma = np.dot(sig_y, X1) / N
            if np.any(sigma) == False:
                print(np.all(sigma), sigma)
                print("iteration: ", _)
                break
            w1[[c],:] = np.subtract(w1[[c],:], step_size*sigma)
    b = np.array(w1.T[0].tolist()[0])
    w = np.delete(w1, 0, 1)

    assert w.shape == (C, D), 'wrong shape of weights matrix'
    assert b.shape == (C,), 'wrong shape of bias terms vector'
    return w, b


def OVR_predict(X, w, b):
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N) 

    one = np.matrix(np.ones(N)).T
    X1 = np.append(one,X, axis=1)
    w1 = np.append(np.matrix(b).T,w, axis=1)
    wTx = np.dot(w1, X1.T)    
    def apply_sig(a):
        return sigmoid(a)
    sig_vec = np.vectorize(apply_sig)
    pred_set = sig_vec(wTx)
    preds = np.array(np.argmax(pred_set, axis=0).tolist()[0])

    assert preds.shape == (N,)
    return preds

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def accuracy_score(true, preds):
    return np.sum(true == preds).astype(float) / len(true)

def run_binary():
    from data_loader import toy_data_binary, \
                            data_loader_mnist 

    print('Performing binary classification on synthetic data')
    X_train, X_test, y_train, y_test = toy_data_binary()
        
    w, b = binary_train(X_train, y_train)
    
    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))
    
    print('Performing binary classification on binarized MNIST')
    X_train, X_test, y_train, y_test = data_loader_mnist()

    binarized_y_train = [0 if yi < 5 else 1 for yi in y_train] 
    binarized_y_test = [0 if yi < 5 else 1 for yi in y_test] 
    
    w, b = binary_train(X_train, binarized_y_train)
    
    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' % 
            (accuracy_score(binarized_y_train, train_preds),
             accuracy_score(binarized_y_test, preds)))

def run_multiclass():
    from data_loader import toy_data_multiclass_3_classes_non_separable, \
                            toy_data_multiclass_5_classes, \
                            data_loader_mnist 
    
    datasets = [(toy_data_multiclass_3_classes_non_separable(), 
                        'Synthetic data', 3), 
                (toy_data_multiclass_5_classes(), 'Synthetic data', 5), 
                (data_loader_mnist(), 'MNIST', 10)]

    for data, name, num_classes in datasets:
        print('%s: %d class classification' % (name, num_classes))
        X_train, X_test, y_train, y_test = data
        
        print('One-versus-rest:')
        w, b = OVR_train(X_train, y_train, C=num_classes)
        train_preds = OVR_predict(X_train, w=w, b=b)
        preds = OVR_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))
    
        print('Multinomial:')
        w, b = multinomial_train(X_train, y_train, C=num_classes)
        train_preds = multinomial_predict(X_train, w=w, b=b)
        preds = multinomial_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))


if __name__ == '__main__':
    
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", )
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.output:
            sys.stdout = open(args.output, 'w')

    if not args.type or args.type == 'binary':
        run_binary()

    if not args.type or args.type == 'multiclass':
        run_multiclass()
        