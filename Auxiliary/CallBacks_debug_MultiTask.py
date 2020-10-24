import keras
from keras import backend as K
import pickle
import Auxiliary.Metrics as Metrics
import numpy as np
from keras.utils import np_utils

class HistoryDebug(keras.callbacks.Callback):

    def __init__(self, X_train, y_train_1, y_train_2, X_val, y_val_1, y_val_2, index_folder, path_file_last):
        self.X_train = X_train
        self.y_train_1 = y_train_1
        self.y_train_2 = y_train_2
        self.X_val = X_val
        self.y_val_1 = y_val_1
        self.y_val_2 = y_val_2
        self.index_folder = index_folder
        self.path_file_last = path_file_last

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        # lr = float(K.get_value(self.model.optimizer.lr))
        # print(" epoch={:02d}, lr={:.5f}".format( epoch, lr ))
        pass

    def on_train_end(self, logs={}):
        #y_pred = self.model.predict([self.X_train, self.X_train[:, :, 125:176, :]], verbose=0)
        #y_pred = y_pred[0]
        [y_pred_1, y_pred_2] = self.model.predict(self.X_train, verbose=0)

        score_fall = Metrics.acc_class(self.y_train_1, y_pred_1, 1)
        score_adl = Metrics.acc_class(self.y_train_1, y_pred_1, 0)
        score_maa = Metrics.maa(self.y_train_1, y_pred_1)
        score_acc = Metrics.acc_callback(self.y_train_1, y_pred_1)
        sensitivity = Metrics.sensitivity(self.y_train_1, y_pred_1)
        specificity = Metrics.specificity(self.y_train_1, y_pred_1)
        score_acc_id = Metrics.acc_callback(self.y_train_2, y_pred_2)
        # score_acc = K.print_tensor(score_acc, 'score_acc= ')
        # score_acc = K.eval(score_acc)

        #y_val_pred = self.model.predict([self.X_val, self.X_val[:, :, 125:176, :]], verbose=0)
        #y_val_pred = y_val_pred[0]
        [y_val_pred_1, y_val_pred_2] = self.model.predict(self.X_val, verbose=0)

        val_score_fall = Metrics.acc_class(self.y_val_1, y_val_pred_1, 1)
        val_score_adl = Metrics.acc_class(self.y_val_1, y_val_pred_1, 0)
        val_score_maa = Metrics.maa(self.y_val_1, y_val_pred_1)
        val_score_acc = Metrics.acc_callback(self.y_val_1, y_val_pred_1)
        val_sensitivity = Metrics.sensitivity(self.y_val_1, y_val_pred_1)
        val_specificity = Metrics.specificity(self.y_val_1, y_val_pred_1)
        val_f1 = Metrics.f1(self.y_val_1, y_val_pred_1)
        val_f1_tf = Metrics.f1_internet(self.y_val_1, y_val_pred_1)
        val_f1_tf2 = Metrics.f1_internet3(self.y_val_1, y_val_pred_1)
        val_score_acc_id = Metrics.acc_callback(self.y_val_2, y_val_pred_2)
        # val_score_acc = K.print_tensor(val_score_acc, 'val_score_acc= ')
        # val_score_acc = K.eval(val_score_acc)

        number_adl = len(np.unique(np.where(self.y_train_1 == np_utils.to_categorical(0, 2))[0]))
        number_fall = len(np.unique(np.where(self.y_train_1 == np_utils.to_categorical(1, 2))[0]))

        with open(self.path_file_last + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([K.eval(score_acc), K.eval(score_fall), K.eval(score_adl),
                                                    K.eval(score_maa), K.eval(sensitivity), K.eval(specificity),
                                                    K.eval(score_acc_id), K.eval(val_score_acc), K.eval(val_score_fall),
                                                    K.eval(val_score_adl), K.eval(val_score_maa),
                                                    K.eval(val_sensitivity), K.eval(val_specificity),
                                                    K.eval(val_score_acc_id), K.eval(val_f1),
                                                    K.eval(val_f1_tf), K.eval(val_f1_tf2),
                                                    number_adl, number_fall], f)
        # print("\n End-Epoch - train score_acc: %.6f \n  - train score_fall: %.6f \n - train score_adl: "
        #      "%.6f \n - train score_maa: %.6f \n - val score_acc: %.6f \n - val score_fall: %.6f \n - val score_adl: "
        #      "%.6f \n - val score_maa: %.6f \n" % (K.eval(score_acc), K.eval(score_fall), K.eval(score_adl),
        #                                            K.eval(score_maa), K.eval(val_score_acc),
        #                                            K.eval(val_score_fall), K.eval(val_score_adl),
        #                                            K.eval(val_score_maa)))