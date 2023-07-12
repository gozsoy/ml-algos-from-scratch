import heapq
from collections import Counter


class KNN:

    def __init__(self, n_neighbors, p, X_train, y_train):
        self.n_neighbors = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    # compute Lp distance between two points
    def _lp_distance(self, v1, v2, p):

        temp_v = [(abs(v1_i - v2_i))**p for v1_i, v2_i in zip(v1, v2)]

        return sum(temp_v) ** (1/p)

    # function for predicting only 1 sample
    def _predict(self, x_test):

        heap = []

        for temp_x, temp_y in zip(self.X_train, self.y_train):

            temp_dist = self._lp_distance(temp_x, x_test, self.p)

            if len(heap) < self.n_neighbors:
                heapq.heappush(heap, (-temp_dist, temp_y))
            elif -heap[0][0] > temp_dist:
                heapq.heappushpop(heap, (-temp_dist, temp_y))
        
        closest_labels = list(map(lambda x: x[1], heap))

        return Counter(closest_labels).most_common(1)[0][0]

    # function for predicting whole test dataset
    def predict(self, X_test):

        predicted_labels = list(map(lambda x: self._predict(x), X_test))
        
        return predicted_labels
    