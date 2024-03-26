from auxiliar_classes.process_data import ProcessDataset
from KNN_experiment import KNNExperiment
from SVM_experiment import SVMExperiment
import pandas as pd

class Run_Experiments(ProcessDataset):
    def __init__(self, knn_params, svm_params, balanced_data = False , apply_PCA = False, PCA_components = 15, iterations = 5, directory = ''):
        self.balanced_data = balanced_data
        self.knn_params = knn_params
        self.apply_PCA = apply_PCA
        self.PCA_components = PCA_components
        self.text = '_no_PCA_' if not apply_PCA else '_PCA_'
        df_orig=pd.read_csv(str(directory)+'train.csv', sep=',', na_values=[" "])
        self.df = df_orig.dropna(axis=0)
        self.directory = directory
        self.iterations = iterations
        self.svm_params = svm_params

    def run(self, show_info = True):
        features = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi']
        labels = ['price_range']
        X_readed, Y_readed = self.divide_dataset(self.df, features, labels)
        
        if show_info:
            self.print_items_per_class(Y_readed)

        if self.balanced_data:
            X_readed, Y_readed = self.balance_data(X_readed, Y_readed, random_seed=1)

        X_norm = self.normalization(X_readed)

        if show_info:
            self.print_class_variance(X_norm)

        if self.apply_PCA:
            X_norm, loss = self.PCA_projection(X_norm, self.PCA_components)
            print("PCA LOSS: ", loss)

        knn_experiment = KNNExperiment(X_norm, Y_readed, self.iterations, self.text, self.directory)
        #knn_experiment.search_best_K(self.knn_params['max_k'])
        #knn_experiment.KNN_experiment(self.knn_params['final_k'])

        svm_experiment = SVMExperiment(X_norm, Y_readed, self.iterations, self.text, self.directory)
        svm_experiment.search_best_params(self.svm_params)
        


def main():
    PCA_parameters = [False, True]
    knn_params = {'max_k':300, 'final_k':175}
    svm_params = {'kernels':['linear', 'rbf'], 'rbf_c':[0.001,0.01,0.1,1,10,100,1000], 'rbf_gamma':[0.001,0.01,0.1,1,10,100,1000]}

    for param in PCA_parameters:
        experiments = Run_Experiments(knn_params, svm_params, balanced_data=False, apply_PCA=param, PCA_components=15, iterations=5)
        #show_info en True permite la visualización del número de ítems por clase y de la varianza de cada componente
        experiments.run(show_info=False)

if __name__ == '__main__':
    main()