import numpy as np 
from sklearn.neighbors import KNeighborsClassifier

class KNNConfig():
    def __init__(self, neighbors, X_train, Y_train, X_test, Y_test):
        self.neigh = KNeighborsClassifier(n_neighbors=neighbors)
        self.x_train = X_train
        self.y_train = Y_train
        self.x_test = X_test
        self.y_test = Y_test

    def KNN_training(self):
        self.neigh.fit(self.x_train, np.ravel(self.y_train))
        return self.neigh
    
    def KNN_predict(self):
        y_predict_knn = self.neigh.predict(self.x_test)
        return y_predict_knn
    
