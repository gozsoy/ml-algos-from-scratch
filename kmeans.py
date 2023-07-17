import random
import numpy as np


class KM:

    def __init__(self, k, tol):
        # number of centroids
        self.k = k
        # centroid locations
        self.centroids = []
        # tolerance for stopping
        self.tol = tol

    # compute Lp distance between two points
    def _lp_distance(self, v1, v2, p=2):

        temp_v = [(abs(v1_i - v2_i))**p for v1_i, v2_i in zip(v1, v2)]

        return sum(temp_v) ** (1/p)

    # initialize cluster means by selecting a random point as initial centroid
    def _init_centroids(self):

        for _ in range(self.k):

            idx = random.randint(0, len(self.X)-1)

            self.centroids.append(self.X_y[idx][0])
        
        return
    
    # find closest centroid to given point x
    def _assign_point(self, x):
        
        assigned_c = np.argmin(list(map(lambda c: self._lp_distance(x, c), 
                                        self.centroids)))
        
        return assigned_c
    
    # compute new cluster means
    def _compute_new_means(self):
        
        current_diff = 0

        for cluster_idx in range(self.k):

            old_mean = self.centroids[cluster_idx]

            new_mean = np.mean(list(map(lambda x_y: x_y[0],
                                        (filter(lambda x_y: 
                                                x_y[1] == cluster_idx,
                                                self.X_y)))), axis=0)

            self.centroids[cluster_idx] = new_mean

            current_diff += self._lp_distance(new_mean, old_mean)
        
        return current_diff/self.k

    # main function
    def fit(self, X):

        # original data matrix
        self.X = X
        
        # prepare data-cluster assignment pairs
        self.X_y = [[i, j] for i, j in zip(self.X, [-1] * len(self.X))]

        # initialize centroids randomly
        self._init_centroids()

        while True:

            # for each data point, find closest centroid
            for idx in range(len(self.X_y)):

                temp_x = self.X_y[idx][0]

                self.X_y[idx][1] = self._assign_point(temp_x)

            # update means
            mean_center_diff = self._compute_new_means()
            
            # check if difference big enough
            if self.tol > mean_center_diff:
                print(f'mean_center_diff: {mean_center_diff}.')
                break
            else:
                print(f'mean_center_diff: {mean_center_diff}.')
        
        return