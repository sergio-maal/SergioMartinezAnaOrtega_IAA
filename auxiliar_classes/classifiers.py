import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

class KNNConfig():
    """Clase para configurar un modelo con k-NN"""
    def __init__(self, neighbors, X_train, Y_train, X_test, Y_test):
        """Se define el modelo y los datos de entrenamiento y test"""
        self.neigh = KNeighborsClassifier(n_neighbors=neighbors)
        self.x_train = X_train
        self.y_train = Y_train
        self.x_test = X_test
        self.y_test = Y_test

    def KNN_training(self):
        """Se ejecuta el entrenamiento del modelo"""
        self.neigh.fit(self.x_train, np.ravel(self.y_train))
        return self.neigh
    
    def KNN_predict(self):
        """Se predicen las clases para un conjunto de entrada de test"""
        y_predict_knn = self.neigh.predict(self.x_test)
        return y_predict_knn

class SVMConfig():
    """Clase para configurar un modelo con k-NN"""
    def __init__(self, X_train, Y_train, X_test, Y_test, kernel, C=None, gamma=None):
        """Se define el modelo y los datos de entrenamiento y test"""
        if kernel == 'linear':
            self.svm_model = SVC(kernel=kernel)
        if kernel == 'rbf':
            self.svm_model = SVC(kernel=kernel, C=C, gamma=gamma)
        
        self.x_train = X_train
        self.y_train = Y_train
        self.x_test = X_test
        self.y_test = Y_test

    def SVM_training(self):
        """Se ejecuta el entrenamiento del modelo"""
        self.svm_model.fit(self.x_train, np.ravel(self.y_train))
        return self.svm_model
    
    def SVM_predict(self):
        """Se predicen las clases para un conjunto de entrada de test"""
        y_predict_svm = self.svm_model.predict(self.x_test)
        return y_predict_svm

    
    
