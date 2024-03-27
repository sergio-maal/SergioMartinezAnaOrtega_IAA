import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from auxiliar_classes.classifiers import SVMConfig
from auxiliar_classes.process_data import ShowResuls
from functools import reduce
import os

class SVMExperiment(ShowResuls):
    """Clase para el lanzamiento de un experimento con SVM"""
    def __init__(self, X, Y, iterations, text, directory):
        """Se definen los parámetros del experimento (datos, iteraciones, etc)"""
        self.x = X
        self.y = Y
        self.iterations = iterations
        self.directory = directory
        self.text = text
        self.create_img_directories()

    def create_img_directories(self):
        """Crea los directorios para almacenar los resultados si no existen"""
        names = ['SVM_img', 'SVM_img/scores', 'SVM_img/times', 'SVM_img/confusion_matrix', 'SVM_img/average_scores', 'SVM_img/average_times']
        for name in names:
            directory_path = os.path.join(self.directory, name)
            if not os.path.isdir(directory_path):
                os.mkdir(directory_path)

    def search_best_params(self, svm_params):
        """"Experimento para encontrar los mejores parámetros entre un conjunto especificado"""
        kernels = svm_params['kernels']
        c_values = svm_params['rbf_c']
        gamma_values = svm_params['rbf_gamma']
        # Bucle que itera entre los kernels especificados, que pueden ser linear o rbf
        for kernel in kernels:
            if kernel == 'linear':
                all_linear_scores =  []
                all_linear_predict_times = []
                all_linear_train_times = []
                # Bucle que realiza el número de iteraciones del experimento con linear kernel
                for n in range(self.iterations):
                    print("SVM Linear model. ITERATION: ", n)
                    # Separación de los datos en train y test
                    X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, random_state = n+1)
                    # Creación del modelo
                    svm_model = SVMConfig(X_train, y_train, X_test, y_test, kernel)
                    # Entrenamiento y registro del tiempo
                    ini_train = time.time()
                    svm_results = svm_model.SVM_training()
                    registred_time_train =(time.time() - ini_train)
                    all_linear_train_times.append(registred_time_train*1000)
                    score = svm_results.score(X_test, y_test)
                    all_linear_scores.append(score)
                    # Predicción de clases en el test y refistro del tiempo 
                    ini_predict = time.time() 
                    y_predict_SVM = svm_model.SVM_predict()
                    registred_time_predict=(time.time() - ini_predict)
                    all_linear_predict_times.append(registred_time_predict*1000)
                    # Se crea la matriz de confusión de cada iteración
                    self.create_confusion_matrix(y_test, y_predict_SVM,'SVM_img/confusion_matrix/SVM_Linear_confusion_matrix_'+str(self.text), n)
                    # Se imprime por pantalla el score y tiempos de entrnamiento y predicción de cada iteración
                    print(f"Score: {score} Training time: {registred_time_train} ms Predict time: {registred_time_predict} ms")
                
                # Se imprime por pantalla la media y desv.típica de los scores y tiempos de entrenamiento y predicción
                print(f"SVM Linear model. Media de las scores {self.text}: {np.average(all_linear_scores)}")
                print(f"SVM Linear model. Desviación típica de las scores {self.text}: {np.std(all_linear_scores)}")
                print(f"SVM Linear model. Media del tiempo de entrenamiento {self.text}: {np.average(all_linear_train_times)} ms")
                print(f"SVM Linear model. Desviación típica del tiempo de entrenamiento {self.text}: {np.std(all_linear_train_times)} ms")
                print(f"SVM Linear model. Media del tiempo de predicción {self.text}: {np.average(all_linear_predict_times)} ms")
                print(f"SVM Linear model. Desviación típica del tiempo de predicción {self.text}: {np.std(all_linear_predict_times)} ms")

            elif kernel == 'rbf':
                all_scores =  []
                all_predict_times = []
                all_train_times = []
                all_best_index = []
                params = []
                # Bucle que realiza el número de iteraciones del experimento con linear rbf
                for n in range(self.iterations):
                    print("SVM RBF model. ITERATION: ", n)
                    X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, random_state = n+1)
                    rbf_scores = []
                    rbf_times_predict = []
                    rbf_times_train = []
                    predictions_SVM = []
                    # Bucles que recorren los distintos valores de c y gamma especificados
                    for c in c_values:
                        for gamma in gamma_values:
                            # Creación del modelo
                            svm_model = SVMConfig(X_train, y_train, X_test, y_test, kernel, c, gamma)
                            # Entrenamiento y registro del tiempo
                            ini_train = time.time()
                            svm_results = svm_model.SVM_training()
                            registred_time_train =(time.time() - ini_train)
                            rbf_times_train.append(registred_time_train*1000)
                            score = svm_results.score(X_test, y_test)
                            rbf_scores.append(score)
                            # Predicción de clases en el test y refistro del tiempo 
                            ini_predict = time.time() 
                            y_predict_SVM = svm_model.SVM_predict()
                            registred_time_predict=(time.time() - ini_predict)
                            rbf_times_predict.append(registred_time_predict*1000)
                            predictions_SVM.append(y_predict_SVM)
                            if n==0:
                                params.append(str([c,gamma]))


                    all_scores.append(rbf_scores)
                    all_train_times.append(rbf_times_train)
                    all_predict_times.append(rbf_times_predict)
                    
                    # Se recogen los parámetros, tiempos e índices para las scores que superen el 90%
                    # Se busca únicamente comparar los parámetros que aporten buenos resultados
                    best_scores = [score for score in rbf_scores if score >= 0.9]
                    best_scores_index = [index for index, score in enumerate(rbf_scores) if score >= 0.9]
                    best_params = [params[index] for index in best_scores_index]
                    best_training_times = [rbf_times_train[index] for index in best_scores_index]
                    best_predict_times = [rbf_times_predict[index] for index in best_scores_index]

                    all_best_index.append(best_scores_index)

                    # Se crean las gráficas con los mejores scores (>90%) y tiempos de entrenamiento y predicción en cada iteración
                    self.save_data(best_params, best_scores, "Best scores (>90%)", "params (c,gamma)", "% success", str(self.directory)+'SVM_img/scores/SVM_RBF_best_scores'+str(self.text), n)
                    self.save_data(best_params, best_training_times, "Training time in best scores", "params (c,gamma)", "ms", str(self.directory)+'SVM_img/times/SVM_RBF_best_score_times_train'+str(self.text), n)
                    self.save_data(best_params, best_predict_times, "Predicton time in best scores", "params (c,gamma)", "ms", str(self.directory)+'SVM_img/times/SVM_RBF_best_score_times_predict'+str(self.text), n)
                    
                    # Se imprimen por pantalla los parámetros y tiempos del mejor score, así como los parámetros y tiempos de los que superen el 90% de score
                    print(f"SVM Scores > 90% para el caso {n}: {best_scores}")
                    print(f"SVM Tiempos de entrenamiento para los mejores scores en {n}: {best_training_times} ms")
                    print(f"SVM Tiempos de predicción para los mejores scores en {n}: {best_predict_times} ms")
                    print(f"Parámetros de los mejores scores en {n}: {best_scores_index}", )
                    print(f"SVM Máximo para el caso {n}: {max(rbf_scores)}, Params: {params[rbf_scores.index(max(rbf_scores))]}")
                    print(f"SVM Tiempo de entrenamiento con mejor score para el caso {n}: {rbf_times_train[rbf_scores.index(max(rbf_scores))]:.3f} ms ")
                    print(f"SVM Tiempo de prediccion con mejor score para el caso {n}: {rbf_times_predict[rbf_scores.index(max(rbf_scores))]:.3f} ms ")
                    self.create_confusion_matrix(y_test, predictions_SVM[rbf_scores.index(max(rbf_scores))],'SVM_img/confusion_matrix/SVM_RBF_confusion_matrix_'+str(self.text), n)

                # Para el análisis final, únicamente retenemos los parámetros que superan el 90% de score en todos las iteraciones, así como sus tiempos
                common_best_index = sorted(list(reduce(set.intersection, (set(x) for x in  all_best_index))))
                common_best_scores = [[scores[index] for index in common_best_index] for scores in all_scores]
                common_best_training_times = [[times[index] for index in common_best_index] for times in all_train_times]
                common_best_predict_times = [[times[index] for index in common_best_index] for times in all_predict_times]
                common_best_params = [params[index] for index, score in enumerate(all_scores[0]) if score in common_best_scores[0]]

                # Se crean las gráficas con la media y desviación típica del score y los tiempos de entrnamiento y predicción para los mejores parámetros comunes a todas las iteraciones
                print("Parámetros comunes en todas las iteraciones: ", common_best_params)
                self.save_average_data(common_best_params, common_best_scores, "Average best scores", "params (c,gamma)", "% success", str(self.directory)+'SVM_img/average_scores/SVM_RBF_average_best_scores'+str(self.text))
                self.save_average_data(common_best_params, common_best_training_times, "Average best training time", "params (c,gamma)", "ms", str(self.directory)+'SVM_img/average_times/SVM_RBF_average_best_times_train'+str(self.text))
                self.save_average_data(common_best_params, common_best_predict_times, "Average best predict time", "params (c,gamma)", "ms", str(self.directory)+'SVM_img/average_times/SVM_RBF_average_best_times_predict'+str(self.text))

            else:
                # Si se utiliza un kernel distinto a linear o rbf, no existe implementación
                raise NotImplementedError
            
    def SVM_experiment(self, final_svm_params):
        """Experimento final con parámetros fijos"""
        kernel = final_svm_params['kernel']
        c = final_svm_params['c']
        gamma = final_svm_params['gamma']
        scores = []
        times_predict = []
        times_train = []

        # Bucle que realiza el número de iteraciones del experimento
        for n in range(self.iterations):
            print("ITERATION: ", n)
            # Separación de los datos en train y test
            X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, random_state = n+1)
            # Creación del modelo
            if kernel == 'linear':
                svm_model = SVMConfig(X_train, y_train, X_test, y_test, kernel)
            elif kernel == 'rbf':
                svm_model = SVMConfig(X_train, y_train, X_test, y_test, kernel, c, gamma)
            else:
                # Si se utiliza un kernel distinto a linear o rbf, no existe implementación
                raise NotImplementedError
            
            # Entrenamiento y registro del tiempo
            ini_train = time.time()
            svm_results = svm_model.SVM_training()
            registred_time_train=(time.time() - ini_train)
            times_train.append(registred_time_train*1000)
            score=svm_results.score(X_test, y_test)
            scores.append(score)
            # Predicción de clases en el test y refistro del tiempo
            ini_predict = time.time() 
            y_predict_SVM= svm_model.SVM_predict()
            registred_time_predict=(time.time() - ini_predict)
            times_predict.append(registred_time_predict*1000)
            # Se crea la matriz de confusión de cada iteración
            self.create_confusion_matrix(y_test, y_predict_SVM,'SVM_img/confusion_matrix/SVM_Final_confusion_matrix_'+str(self.text), n)

        # Se imprime por pantalla la media y desv. típica de los scores y tiempos de entrenamiento y predicción
        print(f"SVM Media scores {self.text}: {np.mean(scores)}")
        print(f"SVM Desv. típica scores {self.text}: {np.std(scores)}")
        print(f"SVM Media tiempo entrenamient {self.text}: {np.mean(times_train)} ms")
        print(f"SVM Desv. típica tiempo entrenamiento {self.text}: {np.std(times_train)} ms")
        print(f"SVM Media tiempo predicción {self.text}: {np.mean(times_predict)} ms")
        print(f"SVM Desv. típica tiempo predicción {self.text}: {np.std(times_predict)} ms")
        
 


