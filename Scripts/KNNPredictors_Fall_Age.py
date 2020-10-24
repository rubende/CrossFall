import Auxiliary.preprocessingData as Data
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

X_train, y_train, subjects_train, ages_train, X_test, y_test, subjects_test, ages_test = Data.loadData_age("data_11/predictores_normalizadoEstandarizado_subjects_age_split")

sum_acc = 0
sum_sensitivity = 0
sum_specifity = 0
sum_maa = 0

#y_train = subjects_train
#y_test = subjects_test

ages_test = np.asarray(ages_test)
X_test = np.asarray(X_test)

old = ages_test >= 60
#old = ages_test < 60


for i in range(len(X_train)):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(np.squeeze(X_train[i]), y_train[i])



    y_pred = model.predict(np.squeeze(X_test[i]))
    y_pred2 = y_pred[old[i]]
    y_test2 = y_test[i][old[i]]
    TP = sum(y_pred2[:, 1] * y_test2[:, 1])
    TN = sum(y_pred2[:, 0] * y_test2[:, 0])
    FP = sum((1 - y_test2[:, 1]) * y_pred2[:, 1])
    FN = sum(y_test2[:, 1] * (1 - y_pred2[:, 1]))

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