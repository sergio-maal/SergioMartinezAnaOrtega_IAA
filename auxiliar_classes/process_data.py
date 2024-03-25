from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

class ProcessDataset():

    @staticmethod
    def divide_dataset(dataset, features, labels):
        feature_df = dataset[features]
        labels_df = dataset[labels]
        X = np.asarray(feature_df)
        Y = np.asarray(labels_df)
        return X, Y
    
    @staticmethod
    def print_items_per_class(Y):
        unique, counts = np.unique(Y, return_counts=True)
        print("ITEMS PER CLASS: ", dict(zip(unique, counts)))
        return unique, counts
    
    @staticmethod
    def normalization(X):
        scaler = preprocessing.StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        return X_scaled
    
    @staticmethod
    def balance_data(X, Y, random_seed = 4):
        sm = SMOTE(random_state=random_seed)
        x_resampled, y_resampled = sm.fit_resample(X, Y)
        return x_resampled, y_resampled
    
    @staticmethod
    def PCA_projection(X, n_components):
        mypca = PCA(n_components=n_components)
        mypca.fit(X)
        values_proj = mypca.transform(X)
        X_projected = mypca.inverse_transform(values_proj)
        loss = ((X - X_projected) ** 2).mean()
        return values_proj, loss
    
    @staticmethod
    def print_class_variance(X):
        mypca = PCA()
        mypca.fit(X)
        print("\n Varianza que aporta cada componente:")
        variance = mypca.explained_variance_ratio_
        print(variance)

        print("\n Varianza acumulada:")
        acumvar = []
        for i in range(len(mypca.explained_variance_ratio_)):
            if i==0:
                acumvar.append(variance[i])
            else:
                acumvar.append(variance[i] + acumvar[i-1])

        for i in range(len(acumvar)):
            print(f" {(i+1):2} componentes: {acumvar[i]} ")


class ShowResuls():
    @staticmethod
    def save_data(X, Y, title, labelx, labely, dir_name, n):
        plt.plot(X, Y)
        plt.grid(True, alpha = 0.2)
        plt.title(title, fontsize=15)
        plt.xlabel(labelx, fontsize=12)
        plt.ylabel(labely, fontsize=12)
        plt.savefig(str(dir_name)+'_'+str(n)+'.jpg')
        plt.close() 

    @staticmethod
    def save_average_data(X, Y, title, labelx, labely, dir_name):
        average = np.mean(Y, axis=0)
        std = np.std(Y, axis=0)
        plt.plot(X, average)
        plt.fill_between(X, average+std, average-std, color = 'lightblue', alpha = 0.5)
        plt.grid(True, alpha = 0.2)
        plt.title(title, fontsize=15)
        plt.xlabel(labelx, fontsize=12)
        plt.ylabel(labely, fontsize=12)
        plt.savefig(str(dir_name)+ '.jpg')
        plt.close()

    
    @staticmethod
    def create_confusion_matrix(y_true, y_pred, dir_name, n):
        cm_kNN = confusion_matrix(y_true, y_pred)
        disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_kNN, display_labels=['Clase 0','Clase 1', 'Clase 2', 'Clase 3'])
        disp_knn.plot()
        plt.title("Matriz de confusión k-NN")
        plt.savefig(str(dir_name)+'_'+str(n)+'.jpg')
        plt.close()