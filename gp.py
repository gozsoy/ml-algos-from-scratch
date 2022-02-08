import numpy as np
from numpy.random import multivariate_normal

# implement rbf kernel
def rbf(x1,x2,output_scale = 0.8,length_scale= 0.5):
    
    return output_scale * np.exp(-np.sum((x1 - x2)**2)/(2*(length_scale**2)))

# implement linear kernel
def linear(x1,x2):
    return np.dot(x1,x2)


def compute_gram(X,Y = None, kernel = rbf):

    if Y is None:
        Y = X

    k = lambda x: np.apply_along_axis(f,1,X,x)
    f = lambda x,y: kernel(x,y)

    return np.apply_along_axis(k,axis = 1, arr = Y).T


class GaussianProcess:

    def __init__(self, kernel, alpha = 1e-10):
        self.alpha = alpha
        self.C_n = None
        self.train_X = None
        self.train_y = None

        if kernel == 'rbf':
            self.kernel = rbf
        elif kernel == 'linear':
            self.kernel = linear
        else:
            raise NotImplementedError()

    def fit(self,train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y

        self.C_n = compute_gram(train_X) + np.identity(train_X.shape[0]) * self.alpha
        return

    def predict(self,test_X, return_cov = False, return_std = False):

        c = compute_gram(test_X) + np.identity(test_X.shape[0]) * self.alpha

        if self.C_n is None: # prior
            mean = np.zeros(test_X.shape[0])
            cov = c

        else: # posterior
            k = compute_gram(X = self.train_X, Y = test_X)
            mean = k.T@np.linalg.inv(self.C_n)@self.train_y
            cov = c - k.T@np.linalg.inv(self.C_n)@k

        if return_cov:
            return mean,cov
        elif return_std:
            std = np.sqrt(cov.diagonal())
            return mean,std
        else:
            return mean

    
    def sample_y(self,test_X):

        c = compute_gram(X = test_X) + np.identity(test_X.shape[0]) * self.alpha

        if self.C_n is None: # sampling from prior
            test_y = multivariate_normal(mean= np.zeros(len(test_X)), cov = c)

        else: # sampling from posterior
            k = compute_gram(X = self.train_X, Y = test_X)

            test_y = multivariate_normal(mean= k.T@np.linalg.inv(self.C_n)@self.train_y, cov = c - k.T@np.linalg.inv(self.C_n)@k)

        return test_y


