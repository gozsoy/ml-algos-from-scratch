import numpy as np
import scipy.stats as stats


class GaussianNaiveBayes:

    def __init__(self):
        return

    # compute parameters given data (prior and likelihood)
    def fit(self, X_train, y_train):

        # compute priors
        self.priors = np.unique(y_train, return_counts=True)[1]/len(y_train)
        
        # compute likelihoods size = (feature x label)
        self.feat_size = X_train.shape[1]
        self.lbl_size = len(self.priors)

        self.likelihoods_mu = np.zeros((self.feat_size, self.lbl_size))
        self.likelihoods_std = np.zeros((self.feat_size, self.lbl_size))

        for lbl_idx in range(self.lbl_size):
            for feat_idx in range(self.feat_size):

                temp = X_train[y_train == lbl_idx, feat_idx]
                
                self.likelihoods_mu[feat_idx, lbl_idx] = np.mean(temp)
                self.likelihoods_std[feat_idx, lbl_idx] = np.std(temp)
        
        return
    
    # predict for one sample
    def _predict(self, temp_x):

        # store computed scores
        scores = np.zeros(self.lbl_size)

        # for each candidate label, compute posterior score
        for lbl_idx in range(self.lbl_size):
            post_score = self.priors[lbl_idx]

            for feat_idx in range(self.feat_size):

                temp_mu = self.likelihoods_mu[feat_idx, lbl_idx]
                temp_std = self.likelihoods_std[feat_idx, lbl_idx]

                post_score *= stats.norm(loc=temp_mu,
                                         scale=temp_std).pdf(temp_x[feat_idx])
            
            scores[lbl_idx] = post_score
        
        # return highest scored label
        return np.argmax(scores)

    # predict for given test matrix
    def predict(self, X_test):
        
        return list(map(lambda x: self._predict(x), X_test))