import Auxiliary.preprocessingData as Data
import numpy as np
from keras.utils import np_utils
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier

X_train, y_train, subjects_train, X_test, y_test, subjects_test = Data.loadData("data_11/predictores_normalizadoEstandarizado_subject")

sum_acc = 0


#y_train = subjects_train
#y_test = subjects_test

for i in range(len(X_train)):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(np.squeeze(X_train[i]), subjects_train[i])

    # Filter by class
    temp = np.unique(np.where(y_test[i] == np_utils.to_categorical(0, 2)))
    data = X_test[i][temp]
    labels = subjects_test[i][temp]

    y_pred = model.predict(np.squeeze(data))
    accuracy = (np.argmax(y_pred, axis =1) == np.argmax(labels, axis =1)).mean()



    print("Fold " + str(i))
    print("acc:" + str(accuracy))

    print("------------------------------")
    sum_acc += accuracy

print("Media")
print("acc:" + str(sum_acc/len(X_train)))
print("------------------------------")
quit()