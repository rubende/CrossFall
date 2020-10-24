import Auxiliary.preprocessingData as Data
import numpy as np
from keras.utils import np_utils
from scipy.spatial.distance import cdist

X_train, y_train, subjects_train, X_test, y_test, subjects_test = Data.loadData("data_15/predictores_normalizadoEstandarizado_acc5")

sum_acc = 0
sum_sensitivity = 0
sum_specifity = 0
sum_maa = 0

for i in range(len(X_train)):
    temp = np.linalg.norm(X_train[i], axis = 2)
    temp = np.expand_dims(temp, axis=3)
    temp_train = X_train[i]/temp

    temp = np.linalg.norm(X_test[i], axis = 2)
    temp = np.expand_dims(temp, axis=3)
    temp_test = X_test[i]/temp
    temp_test = np.squeeze(temp_test)

    train_labels_fall = np.unique(np.where(y_train[i] == np_utils.to_categorical(1, 2))[0])
    train_labels_adl = np.unique(np.where(y_train[i] == np_utils.to_categorical(0, 2))[0])

    mean_fall = np.squeeze(np.mean(temp_train[train_labels_fall], axis = 0))
    mean_adl = np.squeeze(np.mean(temp_train[train_labels_adl], axis = 0))

    means = np.asarray([mean_adl, mean_fall])

    sqd_fall = cdist(means, temp_test, 'seuclidean')
    minimo = np.argmin(sqd_fall, axis=0)
    min_cat = np_utils.to_categorical(minimo, 2)
    TP = sum(min_cat[:,1] * y_test[i][:,1])
    TN = sum(min_cat[:,0] * y_test[i][:,0])
    FP = sum((1 - y_test[i][:, 1]) * min_cat[:, 1])
    FN = sum(y_test[i][:, 1] * (1 - min_cat[:, 1]))

    accuracy =  (TN + TP)/(TN+TP+FN+FP)
    sensitivity = TP/(TP+FN)
    specifity = TN/(TN+FP)
    maa = (sensitivity + specifity)/2

    print("Fold " + str(i))
    print("acc:" + str(accuracy))
    print("sensitivity:" + str(sensitivity))
    print("specifity:" + str(specifity))
    print("maa:" + str(maa))
    print("------------------------------")
    sum_acc += accuracy
    sum_sensitivity += sensitivity
    sum_specifity += specifity
    sum_maa += maa

print("Media")
print("acc:" + str(sum_acc/len(X_train)))
print("sensitivity:" + str(sum_sensitivity/len(X_train)))
print("specifity:" + str(sum_specifity/len(X_train)))
print("maa:" + str(sum_maa/len(X_train)))
print("------------------------------")
quit()