import numpy as np

class Perceptron:

    def __init__(
        self,
        strategy = 'batch', # or 'sgd'
        max_iter = 1000) -> None:
        
        self.weights = None
        self.strategy = strategy
        self.max_iter = max_iter
        self.adjust_y = False # need {1,-1} label set for perceptron to work

    def fit(self,X,y):
        if 0 in np.unique(y):
            self.adjust_y = True
            y= (y*2)-1 

        feat_cnt = X.shape[1]
        smple_cnt = X.shape[0]        
        X = np.concatenate((np.ones((smple_cnt,1)),X),axis = 1) # append column of 1 for bias
        self.weights = np.random.normal(size = feat_cnt+1) # initialize weights with bias

        if self.strategy == 'batch':
        
            for _ in range(self.max_iter):
                wTx = X @ self.weights
                ywTx = y * wTx
                
                # compute loss
                loss = self.loss_fn(-ywTx,smple_cnt)
            
                # compute gradient
                gradient = self.gradient_fn(X,y,-ywTx,smple_cnt)

                # update weights
                self.update_gradient(gradient)

        elif self.strategy == 'sgd':
            
            for _ in range(self.max_iter):
                
                for idx in range(smple_cnt):

                    if y[idx]*np.dot(self.weights,X[idx,:]) < 0: # misclassification
                        self.weights = self.weights + y[idx]*X[idx,:]


        else:
            raise NotImplementedError('invalid strategy type')

        return


    def predict(self,X):
        smple_cnt = X.shape[0]
        X = np.concatenate((np.ones((smple_cnt,1)),X),axis = 1) # append column of 1 for bias

        test_wTx = X @ self.weights
        
        preds = 1 * (test_wTx > 0)

        if self.adjust_y == False:
            preds= (preds*2)-1
        
        return preds

    def score(self,X,y):
        
        return np.sum(self.predict(X) == y) / (X.shape[0])

    def loss_fn(self,neg_ywTx,smple_cnt):

        f = lambda x: max(0,x)
        base = (1/smple_cnt) * np.sum(list(map(f,neg_ywTx)))
        
        return base

    def gradient_fn(self,X,y,neg_ywTx,smple_cnt):
        base = (-1/smple_cnt) * np.sum((y*(neg_ywTx>0)).reshape(-1, 1) * X,axis=0)
        
        return base

    def update_gradient(self,gradient):
        self.weights = self.weights - gradient
        return
