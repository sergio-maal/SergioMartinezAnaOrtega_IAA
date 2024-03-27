from auxiliar_classes.process_data import ProcessDataset
from KNN_experiment import KNNExperiment
from SVM_experiment import SVMExperiment
import pandas as pd

class Run_Experiments(ProcessDataset):
    """
    Clase para el lanzamiento de los experimentos con KNN y SVM
    Hereda de ProcessDataset para realizar el procesamiento del conjunto de datos
    """
    def __init__(self, knn_params, svm_params, balanced_data = False , apply_PCA = False, PCA_components = 15, iterations = 5, directory = ''):
        """Se definen atributos del experimento (parámetros de SVM y KNN, si se usa PCA, nº de iteraciones, etc) y se lee el dataset"""
        self.balanced_data = balanced_data
        self.knn_params = knn_params
        self.apply_PCA = apply_PCA
        self.PCA_components = PCA_components
        self.text = '_no_PCA_' if not apply_PCA else '_PCA_'
        df_orig = pd.read_csv(str(directory)+'train.csv', sep=',', na_values=[" "])
        self.df = df_orig.dropna(axis=0)
        # El self.directory se usa para almacenar datos del experimento en una ruta específica, si se desea.
        self.directory = directory
        self.iterations = iterations
        self.svm_params = svm_params

    def run(self, show_info = True):
        """Se procesa el dataset y se lanzan los experimentos con KNN y SVM"""

        # División del dataset en features (X) y labels (Y)
        features = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi']
        labels = ['price_range']
        X_readed, Y_readed = self.divide_dataset(self.df, features, labels)
        
        # Si show_info es True, se muestran el número de items de cada clase
        # En este caso, el dataset esta balanceado con 500 ítems por clase
        if show_info:
            self.print_items_per_class(Y_readed)

        # Balanceo del datset, si fuese necesario
        if self.balanced_data:
            X_readed, Y_readed = self.balance_data(X_readed, Y_readed, random_seed=1)

        # Normalización de los datos de entrada (X)
        X_norm = self.normalization(X_readed)

        # Si show info es True, se muestra la varianza de cada feature y la varianza acumulada
        if show_info:
            self.print_features_variance(X_norm)

        # Se aplica PCA, si se desea
        if self.apply_PCA:
            X_norm, loss = self.PCA_projection(X_norm, self.PCA_components)
            print("PCA LOSS: ", loss)

        # Ejecución de los experimentos
        # Se realiza primero una búsqueda de parámetros y después se realiza un experimento final con los mejores    
        knn_experiment = KNNExperiment(X_norm, Y_readed, self.iterations, self.text, self.directory)
        knn_experiment.search_best_K(self.knn_params['max_k'])
        knn_experiment.KNN_experiment(self.knn_params['final_k'])

        svm_experiment = SVMExperiment(X_norm, Y_readed, self.iterations, self.text, self.directory)
        svm_experiment.search_best_params(self.svm_params['search_best_params'])
        svm_experiment.SVM_experiment(self.svm_params['final_params'])
        


def main():
    """Ejecución principal"""

    # Se definen los parámetros del experimento
    PCA_parameters = [False, True]
    knn_params = {'max_k':300, 'final_k':175}
    svm_params = {'search_best_params':{'kernels':['linear', 'rbf'], 'rbf_c':[0.001,0.01,0.1,1,10,100,1000], 'rbf_gamma':[0.001,0.01,0.1,1,10,100,1000]}, 'final_params':{'kernel':'linear', 'c':None, 'gamma':None}}

    # Se lanzan los experimentos llamando a la clase Run_Experiments
    for param in PCA_parameters:
        experiments = Run_Experiments(knn_params, svm_params, balanced_data=False, apply_PCA=param, PCA_components=15, iterations=5)
        experiments.run(show_info=True)

if __name__ == '__main__':
    main()