
import pickle
import Auxiliary.preprocessingData as Data
import Auxiliary.createRaw as Raw

[data_all, labels_all, subjects_all] = Raw.createDataset_11("../Raw/BaseDatos_011_SisFall_Colombia.mat")

print("K-fold")

X_train, y_train, subjects_train, X_test, y_test, subjects_test = Data.k_fold(data_all, labels_all, subjects_all, 10)

print("Normalizamos")
X_train = Data.dataNormalizationScaler_11(X_train)
X_test = Data.dataNormalizationScaler_11(X_test)

print("DataAugmentation")
[X_train, y_train , subjects_all] = Data.dataAugmentation_11(X_train, y_train, subjects_train)

print("Save")
Data.saveData(X_train, y_train, subjects_train, X_test, y_test, subjects_test, "data_11/normalizadoEstandarizado")

quit()
