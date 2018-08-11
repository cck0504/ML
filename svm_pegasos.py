import json
import numpy as np


def objective_function(X, y, w, lamb):
    sum_hinge = 0
    for i, row in enumerate(X):
        sum_hinge += max(0, 1-y[i]*np.dot(w.T,row))
    obj_value = (lamb*np.power(np.linalg.norm(w),2) / 2.0) + (sum_hinge / len(X))

    return obj_value


def pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations):
    np.random.seed(0)
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    N = Xtrain.shape[0]
    D = Xtrain.shape[1]

    w = np.reshape(w, (w.shape[0]),)

    train_obj = []

    for iter in range(1, max_iterations + 1):
        A_t = np.floor(np.random.rand(k) * N).astype(int)
        sum_yx = np.zeros(D,)
        for i in A_t:
            if ytrain[i]*np.dot(w.T, Xtrain[i]) < 1.0:
                sum_yx = np.add(sum_yx, ytrain[i]*Xtrain[i])
        eta = 1.0 / (lamb*iter)
        w_half = np.add((1.0-eta*lamb)*w, (eta/k)*sum_yx)
        w = min(1, ((1.0/np.sqrt(lamb))/np.linalg.norm(w_half)))*w_half
        obj_value = objective_function(Xtrain, ytrain, w, lamb)
        train_obj.append(obj_value)
    return w, train_obj


def pegasos_test(Xtest, ytest, w, t = 0.):
    Xtest = np.array(Xtest)
    ytest = np.array(ytest)
    ypred = np.dot(Xtest, w)
    for i,y in enumerate(ypred):
        ypred[i] = -1 if y < t else 1
    cnt = 0
    for yp, yl in zip(ypred, ytest):
        if yp == yl:
            cnt += 1
    test_acc = cnt/len(ytest)
    return test_acc

def data_loader_mnist(dataset):

    with open(dataset, 'r') as f:
            data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xvalid = valid_set[0]
    yvalid = valid_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    Xtrain = np.hstack((np.ones((len(Xtrain), 1)), np.array(Xtrain))).tolist()
    Xvalid = np.hstack((np.ones((len(Xvalid), 1)), np.array(Xvalid))).tolist()
    Xtest = np.hstack((np.ones((len(Xtest), 1)), np.array(Xtest))).tolist()

    for i, v in enumerate(ytrain):
        if v < 5:
            ytrain[i] = -1.
        else:
            ytrain[i] = 1.
    for i, v in enumerate(ytest):
        if v < 5:
            ytest[i] = -1.
        else:
            ytest[i] = 1.

    return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest


def pegasos_mnist():

    test_acc = {}
    train_obj = {}

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = data_loader_mnist(dataset = 'mnist_subset.json')

    max_iterations = 500
    k = 100
    for lamb in (0.01, 0.1, 1):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    lamb = 0.1
    for k in (1, 10, 1000):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    return test_acc, train_obj


def main():
    test_acc, train_obj = pegasos_mnist()
    print('mnist test acc \n')
    for key, value in test_acc.items():
        print('%s: test acc = %.4f \n' % (key, value))

    with open('pegasos.json', 'w') as f_json:
        json.dump([test_acc, train_obj], f_json)


if __name__ == "__main__":
    main()
