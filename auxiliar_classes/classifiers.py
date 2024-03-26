import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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

class SVMConfig():
    def __init__(self, X_train, Y_train, X_test, Y_test, kernel, C=None, gamma=None):
        if kernel == 'linear':
            self.svm_model = SVC(kernel=kernel)
        if kernel == 'rbf':
            self.svm_model = SVC(kernel=kernel, C=C, gamma=gamma)
        
        self.x_train = X_train
        self.y_train = Y_train
        self.x_test = X_test
        self.y_test = Y_test

    def SVM_training(self):
        self.svm_model.fit(self.x_train, np.ravel(self.y_train))
        return self.svm_model
    
    def SVM_predict(self):
        y_predict_svm = self.svm_model.predict(self.x_test)
        return y_predict_svm

    
    
