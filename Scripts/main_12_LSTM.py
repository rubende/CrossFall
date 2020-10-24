import Redes.Red_LSTM as Interface
import Auxiliary.preprocessingData as Data
import Auxiliary.GPUtil as GPU
import numpy as np
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
                model = Interface.Red.build((1, 1, 50, 3), 2, number_convolutional_layers=4, first_number_filters=256, dropout=0.5)
                history, model = Interface.Red.train(model, index_data, name_path + '_all_'+str(index_data), X_train[index_data], y_train[index_data], X_test[index_data],
                                         y_test[index_data], noise=1, l2_noise=1, weight_decay_noise=0.00001,  stepped = True)
                #, class_weight = {0 : 0.4, 1 : 0.6}

                with open(name_path + str(index_data) + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump([history.history], f)
                # Interface.Red.plot_info(history)



X_train, y_train, subjects_train, X_test, y_test, subjects_test = Data.loadData("data_12_fullWindow/normalizadoEstandarizado")
name_path = "../Results/result_12/lstm_4_capas/escalado/sin_weight/"

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


for i in range(number_folders):
    X_train[i] = div_signal(X_train[i])
    X_test[i] = div_signal(X_test[i])




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




