import math
import numpy as np


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    Q = I - unit / n

    return np.dot(np.dot(Q, K), Q)


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / sigma / sigma
    np.exp(KX, KX)
    return KX


def HSIC(X, Y, kernel='rbf', sigma=None):
    if kernel == 'rbf':
        return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))
    else:
        return np.sum(centering(np.matmul(X, X.T)) * centering(np.matmul(Y, Y.T)))

def CKA(X, Y, kernel='rbf', sigma=None):
    return HSIC(X, Y, kernel, sigma) / np.sqrt(HSIC(X, X, kernel, sigma) * HSIC(Y, Y, kernel, sigma))

if __name__ == '__main__':
    X = np.random.randn(10, 5)
    Y = np.random.randn(10, 5)
    print(CKA(X, Y))
    print(CKA(X, X))