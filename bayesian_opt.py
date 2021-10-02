import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import random

class BOpt:
    def __init__(self,gpr,x,f):
        self.gpr = gpr # gaussian process object
        self.x = x # interval for fnc (hyperparam configs)
        self.f = f # function whose max is needed
        self.train_x = []
        self.train_y = None

    def update_gp(self):
        self.gpr.fit(self.train_x,self.train_y)
        return
    
    def plot_gp(self):
        post_mean, post_var = self.gpr.predict(self.x,return_std=True)

        lower = post_mean - 2*post_var
        upper = post_mean + 2*post_var

        plt.clf()
        plt.plot(self.x,post_mean)
        plt.fill_between(self.x.squeeze(),lower,upper,alpha = 0.3,color = 'darkorange')
        plt.plot(self.train_x,self.train_y,'r*',)
        return

class ThompsonSampling(BOpt):
    def __init__(self,gpr,x,f):
        super().__init__(gpr,x,f)
        
    def query_new_point(self):
        sampled_y = self.gpr.sample_y(self.x,random_state = np.random.randint(0,10))
        
        self.train_x = np.expand_dims(np.append(self.train_x,self.x[sampled_y.argmax()]),axis=1)
        self.train_y = self.f(self.train_x).ravel()
        return

class GP_UCB(BOpt):
    def __init__(self,gpr,x,f,beta):
        super().__init__(gpr,x,f)
        self.beta = beta

    def query_new_point(self):
        post_mean, post_var = self.gpr.predict(self.x,return_std=True)
        upper = post_mean + self.beta * post_var

        self.train_x = np.expand_dims(np.append(self.train_x,self.x[upper.argmax()]),axis=1)
        self.train_y = self.f(self.train_x).ravel()
        return