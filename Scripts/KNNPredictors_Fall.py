import Auxiliary.preprocessingData as Data
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

X_train, y_train, subjects_train, X_test, y_test, subjects_test = Data.loadData("data_11/predictores_normalizadoEstandarizado_subject")

sum_acc = 0
sum_sensitivity = 0
sum_specifity = 0
sum_maa = 0

#y_train = subjects_train
#y_test = subjects_test

for i in range(len(X_train)):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(np.squeeze(X_train[i]), y_train[i])

    y_pred = model.predict(np.squeeze(X_test[i]))
    TP = sum(y_pred[:, 1] * y_test[i][:, 1])
    TN = sum(y_pred[:, 0] * y_test[i][:, 0])
    FP = sum((1 - y_test[i][:, 1]) * y_pred[:, 1])
    FN = sum(y_test[i][:, 1] * (1 - y_pred[:, 1]))

    accuracy = (TN + TP) / (TN + TP + FN + FP)
    sensitivity = TP / (TP + FN)
    specifity = TN / (TN + FP)
    maa = (sensitivity + specifity) / 2


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