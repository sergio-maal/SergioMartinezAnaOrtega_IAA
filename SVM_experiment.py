import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from auxiliar_classes.classifiers import SVMConfig
from auxiliar_classes.process_data import ShowResuls
from functools import reduce

class SVMExperiment(ShowResuls):
    def __init__(self, X, Y, iterations, text, directory):
        self.x = X
        self.y = Y
        self.iterations = iterations
        self.directory = directory
        df_orig=pd.read_csv(str(directory)+'train.csv', sep=',', na_values=[" "])
        self.df = df_orig.dropna(axis=0)
        self.text = text

    def search_best_params(self, svm_params):
        kernels = svm_params['kernels']
        c_values = svm_params['rbf_c']
        gamma_values = svm_params['rbf_gamma']
        all_scores =  []
        all_predict_times = []
        all_train_times = []
        #neighbors=list(range(1, max_k+1))
        for kernel in kernels:
            if kernel == 'linear':
                all_linear_scores =  []
                all_linear_predict_times = []
                all_linear_train_times = []
                for n in range(self.iterations):
                    print("SVM Linear model. ITERATION: ", n)
                    X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, random_state = n+1)
                    svm_model = SVMConfig(X_train, y_train, X_test, y_test, kernel)
                    ini_train = time.time()
                    svm_results = svm_model.SVM_training()
                    registred_time_train =(time.time() - ini_train)
                    all_linear_train_times.append(registred_time_train)
                    score = svm_results.score(X_test, y_test)
                    all_linear_scores.append(score) 
                    ini_predict = time.time() 
                    y_predict_SVM = svm_model.SVM_predict()
                    registred_time_predict=(time.time() - ini_predict)
                    all_linear_predict_times.append(registred_time_predict)
                    self.create_confusion_matrix(y_test, y_predict_SVM,'SVM_img/confusion_matrix/SVM_Linear_confusion_matrix_'+str(self.text), n)
                    print(f"Score: {score} Training time: {registred_time_train*1000} ms Predict time: {registred_time_predict*1000} ms")
                
                print(f"SVM Linear model. Media de las scores {self.text}: {np.average(all_linear_scores)}")
                print(f"SVM Linear model. Desviación típica de las scores {self.text}: {np.std(all_linear_scores)}")
                print(f"SVM Linear model. Media del tiempo de entrenamiento {self.text}: {np.average(all_linear_train_times)}")
                print(f"SVM Linear model. Desviación típica del tiempo de entrenamiento {self.text}: {np.std(all_linear_train_times)}")
                print(f"SVM Linear model. Media del tiempo de predicción {self.text}: {np.average(all_linear_predict_times)}")
                print(f"SVM Linear model. Desviación típica del tiempo de predicción {self.text}: {np.std(all_linear_predict_times)}")

            elif kernel == 'rbf':
                all_scores =  []
                all_predict_times = []
                all_train_times = []
                all_best_index = []
                params = []
                for n in range(self.iterations):
                    print("SVM RBF model. ITERATION: ", n)
                    X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, random_state = n+1)
                    rbf_scores = []
                    rbf_times_predict = []
                    rbf_times_train = []
                    predictions_SVM = []
                    for c in c_values:
                        for gamma in gamma_values:
                            svm_model = SVMConfig(X_train, y_train, X_test, y_test, kernel, c, gamma)
                            ini_train = time.time()
                            svm_results = svm_model.SVM_training()
                            registred_time_train =(time.time() - ini_train)
                            rbf_times_train.append(registred_time_train)
                            score = svm_results.score(X_test, y_test)
                            rbf_scores.append(score) 
                            ini_predict = time.time() 
                            y_predict_SVM = svm_model.SVM_predict()
                            registred_time_predict=(time.time() - ini_predict)
                            rbf_times_predict.append(registred_time_predict)
                            predictions_SVM.append(y_predict_SVM)
                            if n==0:
                                params.append(str([c,gamma]))

                    best_scores = [score for score in rbf_scores if score >= 0.9]
                    best_scores_index = [index for index, score in enumerate(rbf_scores) if score >= 0.9]
                    best_params = [params[index] for index in best_scores_index]
                    best_training_times = [rbf_times_train[index]*1000 for index in best_scores_index]
                    best_predict_times = [rbf_times_predict[index]*1000 for index in best_scores_index]
            
                    all_scores.append(rbf_scores)
                    all_train_times.append(rbf_times_train)
                    all_predict_times.append(rbf_times_predict)
                    all_best_index.append(best_scores_index)

                    self.save_data(best_params, best_scores, "Best scores (>90%)", "params (c,gamma)", "% success", str(self.directory)+'SVM_img/scores/SVM_best_scores'+str(self.text), n)
                    self.save_data(best_params, best_training_times, "Training time in best scores", "params (c,gamma)", "ms", str(self.directory)+'SVM_img/times/SVM_best_score_times_train'+str(self.text), n)
                    self.save_data(best_params, best_predict_times, "Predicton time in best scores", "params (c,gamma)", "ms", str(self.directory)+'SVM_img/times/SVM_best_score_times_predict'+str(self.text), n)

                    print(f"SVM Máximo: {max(rbf_scores)}, Params: {params[rbf_scores.index(max(rbf_scores))]}")
                    print(f"SVM Tiempo de entrenamiento con mejor score para el caso {n}: {rbf_times_train[rbf_scores.index(max(rbf_scores))]*1000:.3f} ms ")
                    print(f"SVM Tiempo de prediccion con mejor score para el caso {n}: {rbf_times_predict[rbf_scores.index(max(rbf_scores))]*1000:.3f} ms ")
                    self.create_confusion_matrix(y_test, predictions_SVM[rbf_scores.index(max(rbf_scores))],'SVM_img/confusion_matrix/SVM_confusion_matrix_'+str(self.text), n)

                common_best_index = sorted(list(reduce(set.intersection, (set(x) for x in  all_best_index))))
                common_best_scores = [[scores[index] for index in common_best_index] for scores in all_scores]
                common_best_training_times = [[times[index] for index in common_best_index] for times in all_train_times]
                print(all_train_times)
                common_best_predict_times = [[times[index] for index in common_best_index] for times in all_predict_times]
                common_best_params = [params[index] for index, score in enumerate(all_scores[0]) if score in common_best_scores[0]]
                print(best_params)


                self.save_average_data(common_best_params, common_best_scores, "Average best scores", "params (c,gamma)", "% success", str(self.directory)+'SVM_img/average_scores/SVM_average_best_scores'+str(self.text))
                self.save_average_data(common_best_params, common_best_training_times, "Average best training time", "params (c,gamma)", "ms", str(self.directory)+'SVM_img/average_times/SVM_average_best_times_train'+str(self.text))
                self.save_average_data(common_best_params, common_best_predict_times, "Average best predict time", "params (c,gamma)", "ms", str(self.directory)+'SVM_img/average_times/SVM_average_best_times_predict'+str(self.text))

            else:
                raise NotImplementedError
            
    def SVM_experiment(self, final_svm_params):
        kernel = final_svm_params['kernel']
        c = final_svm_params['c']
        gamma = final_svm_params['k']

        scores = []
        times_predict = []
        times_train = []

        for n in range(self.iterations):
            print("ITERATION: ", n)
            X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, random_state = n+1)
            if kernel == 'linear':
                svm_model = SVMConfig(X_train, y_train, X_test, y_test, kernel)
            elif kernel == 'rbf':
                svm_model = SVMConfig(X_train, y_train, X_test, y_test, kernel)
            else:
                raise NotImplementedError
            
            ini_train = time.time()
            svm_results = svm_model.SVM_training()
            registred_time_train=(time.time() - ini_train)
            times_train.append(registred_time_train)
            score=svm_results.score(X_test, y_test)
            scores.append(score)
            ini_predict = time.time() 
            y_predict_SVM= svm_model.SVM_predict()
            registred_time_predict=(time.time() - ini_predict)
            times_predict.append(registred_time_predict)

            self.create_confusion_matrix(y_test, y_predict_SVM,'KNN_img/confusion_matrix/KNN_Final_confusion_matrix_'+str(self.text), n)


        print(f"SVM Media scores {self.text}: {np.mean(scores)}")
        print(f"SVM Desv. típica scores {self.text}: {np.std(scores)}")
        print(f"SVM Media tiempo entrenamient {self.text}: {np.mean(times_train)*1000} ms")
        print(f"SVM Desv. típica tiempo entrenamiento {self.text}: {np.std(times_train)*1000} ms")
        print(f"SVM Media tiempo predicción {self.text}: {np.mean(times_predict)*1000} ms")
        print(f"SVM Desv. típica tiempo predicción {self.text}: {np.std(times_predict)*1000} ms")
        
 


