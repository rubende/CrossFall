
import pickle
import Auxiliary.preprocessingData as Data

with open("../Raw/dataset_05_fullWindow_sub.pkl", 'rb') as f:  # Python 3: open(..., 'rb')
    [data_all, labels_all, sub_all] = pickle.load(f)

print("K-fold")

X_train, y_train, subjects_train, X_test, y_test, subjects_test = Data.k_fold(data_all, labels_all, sub_all, 10)

print("Normalizamos")
X_train = Data.dataNormalizationScaler(X_train)
X_test = Data.dataNormalizationScaler(X_test)

print("DataAugmentation")
X_train, y_train, subjects_train = Data.dataAugmentation(X_train, y_train, subjects_train)

print("Save")
Data.saveData(X_train, y_train, subjects_train, X_test, y_test, subjects_test, "data_5_fullWindow/normalizadoEstandarizado_subjects")

quit()
