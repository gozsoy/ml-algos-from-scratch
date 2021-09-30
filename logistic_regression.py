import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression:

    def __init__(
        self,
        penalty = 'l2',
        C = 1.0,
        multiclass = 'auto', # to be implemented
        l1_ratio = 0.2,
        lr_rate = 0.05,
        max_iter = 1000,
        cutoff_prob = 0.5) -> None:
        
        self.penalty = penalty
        self.C = C
        self.multiclass = multiclass
        self.l1_ratio = l1_ratio # only used in elasticnet.
        self.lr_rate = lr_rate
        self.max_iter = max_iter
        self.cutoff_prob = cutoff_prob

        self.weights = None

    def fit(self,X,y):
        feat_cnt = X.shape[1]
        smple_cnt = X.shape[0]
        X = np.concatenate((np.ones((smple_cnt,1)),X),axis = 1) # append column of 1 for bias
        self.weights = np.random.normal(size = feat_cnt+1) # initialize weights with bias

        for _ in range(self.max_iter):
            logit = X @ self.weights
            f = lambda x: sigmoid(x)
            prob = f(logit)
            
            # compute loss
            loss = self.loss_fn(y,prob,smple_cnt)
            
            # compute gradient
            gradient = self.gradient_fn(X,y,prob,smple_cnt)

            # update weights
            self.update_gradient(gradient)

        return


    def predict(self,X):
        smple_cnt = X.shape[0]
        X = np.concatenate((np.ones((smple_cnt,1)),X),axis = 1) # append column of 1 for bias

        logit = X @ self.weights
        f = lambda x: sigmoid(x)
        prob = f(logit)

        preds = 1 * (prob > self.cutoff_prob)

        return  preds

    def score(self,X,y):

        return np.sum(self.predict(X) == y) / (X.shape[0])

    def loss_fn(self,y,prob,smple_cnt):
        base = (-1/smple_cnt) * np.sum(y * np.log(prob) + (1-y) * np.log(1-prob))
        reg_term = None

        if self.penalty == 'elastic':
            reg_term =  (1-self.l1_ratio)*(self.weights.T @ self.weights) + self.l1_ratio*np.sum(np.abs(self.weights))
        elif self.penalty == 'l2':
            reg_term =  self.weights.T @ self.weights
        elif self.penalty == 'l1':
            reg_term = np.sum(np.abs(self.weights))
        else: # no regularizer
            reg_term = 0
        
        return base + (1/smple_cnt) * (1/self.C) * reg_term

    def gradient_fn(self,X,y,prob,smple_cnt):
        base = (-1/smple_cnt) * np.sum(((y * (1-prob))[:, None] * X - ((1-y)*prob)[:, None] * X),axis=0)
        reg_term = None

        if self.penalty == 'elastic':
            reg_term =  (1-self.l1_ratio)*2*self.weights + self.l1_ratio*np.sign(self.weights)
        elif self.penalty == 'l2':
            reg_term =  2 * self.weights
        elif self.penalty == 'l1':
            reg_term = np.sign(self.weights)
        else: # no regularizer
            reg_term = 0
        
        return base + (1/smple_cnt) * (1/self.C) * reg_term

    def update_gradient(self,gradient):
        self.weights = self.weights - self.lr_rate * gradient
        return
    
