import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from auxiliar_classes.classifiers import KNNConfig
from auxiliar_classes.process_data import ShowResuls

class KNNExperiment(ShowResuls):
    def __init__(self, X, Y, iterations, text, directory):
        self.x = X
        self.y = Y
        self.iterations = iterations
        self.directory = directory
        df_orig=pd.read_csv(str(directory)+'train.csv', sep=',', na_values=[" "])
        self.df = df_orig.dropna(axis=0)
        self.text = text
        self.create_img_directories()

    def create_img_directories(self):
        names = ['KNN_img', 'KNN_img/scores', 'KNN_img/times', 'KNN_img/confusion_matrix', 'KNN_img/average_scores', 'KNN_img/average_times']
        for name in names:
            directory_path = os.path.join(self.directory, name)
            if not os.path.isdir(directory_path):
                os.mkdir(directory_path)

    def search_best_K(self, max_k):
        all_scores =  []
        all_predict_times = []
        all_train_times = []
        neighbors=list(range(1, max_k+1))
        for n in range(self.iterations):
            print("KNN ITERATION: ", n)
            X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, random_state = n+1)
            scores=[]
            times_predict=[]
            times_train = []
            predictions_KNN=[]
            for k in range(max_k):
                knn_model = KNNConfig(k+1, X_train, y_train, X_test, y_test)
                ini_train = time.time() 
                neigh = knn_model.KNN_training()
                registred_time_train =(time.time() - ini_train)
                times_train.append(registred_time_train)
                score=neigh.score(X_test, y_test)
                scores.append(score)
                ini_predict = time.time() 
                y_predict_KNN= knn_model.KNN_predict()
                registred_time_predict=(time.time() - ini_predict)
                times_predict.append(registred_time_predict)
                predictions_KNN.append(y_predict_KNN)

            all_scores.append(scores)
            all_train_times.append(times_train)
            all_predict_times.append(times_predict) 
        
            self.save_data(neighbors, scores, "Score", "k", "% success", str(self.directory)+'KNN_img/scores/KNN_study_scores'+str(self.text), n)
            self.save_data(neighbors, times_train, "Training time", "k", "ms", str(self.directory)+'KNN_img/times/KNN_study_times_train'+str(self.text), n)
            self.save_data(neighbors, times_predict, "Predicton time", "k", "ms", str(self.directory)+'KNN_img/times/KNN_study_times_predict'+str(self.text), n)
            
            print(f"KNN Máximo: {max(scores)}, indice: {scores.index(max(scores))+1}")
            print(f"KNN Tiempo de entrenamiento con mejor score para el caso {n}: {times_train[scores.index(max(scores))]*1000:.3f} ms ")
            print(f"KNN Tiempo de prediccion con mejor score para el caso {n}: {times_predict[scores.index(max(scores))]*1000:.3f} ms ")
            self.create_confusion_matrix(y_test, predictions_KNN[scores.index(max(scores))],'KNN_img/confusion_matrix/KNN_confusion_matrix_'+str(self.text), n)
        
        self.save_average_data(neighbors, all_scores, "Average score", "k", "% success", str(self.directory)+'KNN_img/average_scores/KNN_average_scores'+str(self.text))
        self.save_average_data(neighbors, all_train_times, "Average trainign time", "k", "ms", str(self.directory)+'KNN_img/average_times/KNN_average_times_train'+str(self.text))
        self.save_average_data(neighbors, all_predict_times, "Average predict time", "k", "ms", str(self.directory)+'KNN_img/average_times/KNN_average_times_predict'+str(self.text))

    def KNN_experiment(self, k_neighbors):
        scores = []
        times_predict = []
        times_train = []

        for n in range(self.iterations):
            print("ITERATION: ", n)
            X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, random_state = n+1)
            knn_model = KNNConfig(k_neighbors, X_train, y_train, X_test, y_test)
            ini_train = time.time()
            neigh = knn_model.KNN_training()
            registred_time_train=(time.time() - ini_train)
            times_train.append(registred_time_train)
            score=neigh.score(X_test, y_test)
            scores.append(score)
            ini_predict = time.time() 
            y_predict_KNN= knn_model.KNN_predict()
            registred_time_predict=(time.time() - ini_predict)
            times_predict.append(registred_time_predict)
            self.create_confusion_matrix(y_test, y_predict_KNN,'KNN_img/confusion_matrix/KNN_Final_confusion_matrix_'+str(self.text), n)


        print(f"KNN Media scores {self.text}: {np.mean(scores)}")
        print(f"KNN Desv. típica scores {self.text}: {np.std(scores)}")
        print(f"KNN Media tiempo entrenamient {self.text}: {np.mean(times_train)*1000} ms")
        print(f"KNN Desv. típica tiempo entrenamiento {self.text}: {np.std(times_train)*1000} ms")
        print(f"KNN Media tiempo predicción {self.text}: {np.mean(times_predict)*1000} ms")
        print(f"KNN Desv. típica tiempo predicción {self.text}: {np.std(times_predict)*1000} ms")
        
 




