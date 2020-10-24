import Redes.Red_Ini as Interface
import Auxiliary.preprocessingData as Data
import Auxiliary.GPUtil as GPU
import numpy as np
from keras.utils import np_utils

import os

from threading import Thread
import pickle


import tensorflow as tf


###################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3" # To force tensorflow to only see one GPU.
# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
###################################


def launch_cnn(index_data, index_gpu, graph, name_path, X_train, y_train, X_test, y_test):

    with graph.as_default():
        sess = tf.Session(config=config, graph=graph)
        with tf.device('/gpu:'+str(index_gpu)):
            with sess.as_default():


                print("y_train_external: " + str(len(y_train[index_data])))
                model = Interface.Red.build((1, 301, 3), 2, number_convolutional_layers=4, first_number_filters=256, dropout=0.5)
                history, model = Interface.Red.train(model, index_data, name_path + '_all_'+str(index_data), X_train[index_data], y_train[index_data], X_test[index_data],
                                         y_test[index_data], noise=1, l2_noise=1, weight_decay_noise=0.00001, class_weight = {0 : 0.2, 1 : 0.8},  stepped = True)
                #, class_weight = {0 : 0.4, 1 : 0.6}

                model.save_weights(name_path + str(index_data) + "_1.h5")

                model2 = Interface.Red.build((1, 301, 3), 2, number_convolutional_layers=4, first_number_filters=256, dropout=0.5)
                history2, model2 = Interface.Red.train(model2, index_data, name_path + '_all_'+str(index_data), X_train[index_data], y_train[index_data], X_test[index_data],
                                         y_test[index_data], noise=1, l2_noise=1, weight_decay_noise=0.00001,  stepped = True)

                model2.save_weights(name_path + str(index_data) + "_2.h5")


                #with open(name_path + str(index_data) + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                #    pickle.dump([history.history], f)
                # Interface.Red.plot_info(history)

                t1 = model.predict(X_test[index_data], batch_size=128)
                t2 = model2.predict(X_test[index_data], batch_size=128)
                t3 = t1 * t2

                pred_1 = np.argmax(t1, axis=1)
                pred_2 = np.argmax(t2, axis=1)
                pred_3 = np.argmax(t3, axis=1)

                labels = np.argmax(y_test[index_data], axis=1)


                equals_1 = np.equal(pred_1, labels)
                equals_1 = [int(elem) for elem in equals_1]
                equals_1 = np.array(equals_1)
                not_equals_1 = 1 - equals_1

                equals_2 = np.equal(pred_2, labels)
                equals_2 = [int(elem) for elem in equals_2]
                equals_2 = np.array(equals_2)
                not_equals_2 = 1 - equals_2

                equals_fusion = np.equal(pred_3, labels)
                equals_fusion = [int(elem) for elem in equals_fusion]
                equals_fusion = np.array(equals_fusion)
                not_equals_fusion = 1 - equals_fusion

                index_fall = np.squeeze(np.argwhere(labels == 1))
                index_adl = np.squeeze(np.argwhere(labels == 0))

                TP_1 = np.sum(equals_1[index_fall])
                FN_1 = np.sum(not_equals_1[index_fall])
                TN_1 = np.sum(equals_1[index_adl])
                FP_1 = np.sum(not_equals_1[index_adl])

                TP_2 = np.sum(equals_2[index_fall])
                FN_2 = np.sum(not_equals_2[index_fall])
                TN_2 = np.sum(equals_2[index_adl])
                FP_2 = np.sum(not_equals_2[index_adl])

                TP_fusion = np.sum(equals_fusion[index_fall])
                FN_fusion = np.sum(not_equals_fusion[index_fall])
                TN_fusion = np.sum(equals_fusion[index_adl])
                FP_fusion = np.sum(not_equals_fusion[index_adl])


                sensitivity_1 = TP_1/(TP_1+FN_1)
                specificity_1 = TN_1/(TN_1+FP_1)
                maa_1 = (sensitivity_1 + specificity_1)/2
                acc_1 = np.sum(equals_1)/len(labels)

                sensitivity_2 = TP_2/(TP_2+FN_2)
                specificity_2 = TN_2/(TN_2+FP_2)
                maa_2 = (sensitivity_2 + specificity_2)/2
                acc_2 = np.sum(equals_2)/len(labels)

                sensitivity_fusion = TP_fusion/(TP_fusion+FN_fusion)
                specificity_fusion = TN_fusion/(TN_fusion+FP_fusion)
                maa_fusion = (sensitivity_fusion + specificity_fusion)/2
                acc_fusion = np.sum(equals_fusion)/len(labels)

                with open(name_path + str(index_data) + '_late.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump([acc_1, sensitivity_1, specificity_1, maa_1, acc_2, sensitivity_2, specificity_2, maa_2,
                                 acc_fusion, sensitivity_fusion, specificity_fusion, maa_fusion], f)




X_train, y_train, subjects_train, X_test, y_test, subjects_test = Data.loadData("data_5_fullWindow/normalizadoEstandarizado")
name_path = "../Results/result_05_fullWindow/late_fusion_4_capas/escalado/con_weight_0.8__sin_weight/"

try:
    # Create target Directory
    os.makedirs(name_path)
    print("Directory " , name_path ,  " Created ")
except FileExistsError:
    print("Directory " , name_path ,  " already exists")


name_file = "result"
number_folders = 10
number_gpus = 4
#number_folders = 1
#number_gpus = 1

temp_number_folder = 0
while temp_number_folder < number_folders:
    threads = []
    for i in range(number_gpus):
        if temp_number_folder < number_folders:
            graph = tf.Graph()
            t = Thread(target=launch_cnn, args=(temp_number_folder, i ,graph, name_path + name_file, X_train, y_train, X_test, y_test))
            temp_number_folder = temp_number_folder + 1
            threads.append(t)

    # Start all threads
    for x in threads:
        x.start()

    # Wait for all of them to finish
    for x in threads:
        x.join()


quit()
