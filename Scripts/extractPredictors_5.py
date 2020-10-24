import Redes.Red_LSTM as Interface
import Auxiliary.preprocessingData as Data
import Auxiliary.noiseGradients as Noise
import Auxiliary.Metrics as Metrics
import Auxiliary.createRaw as Raw

import numpy as np
import keras
import os
from threading import Thread
import pickle

import telebot  # Importamos las librería

import tensorflow as tf


###################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # To force tensorflow to only see one GPU.
# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
###################################


number_folds = 10
predictors = []
labels = []
name_path = "../Results/result_05_fullWindow/lstm_4_capas/escalado/con_weight_0.67/"

def div_signal(signal):
    temp = signal
    temp1 = temp[:, :, 0:50, :]
    temp2 = temp[:, :, 50:100, :]
    temp3 = temp[:, :, 100:150, :]
    temp4 = temp[:, :, 150:200, :]
    temp5 = temp[:, :, 200:250, :]
    temp6 = temp[:, :, 250:300, :]

    temp_f = []

    for i in range(len(temp)):
        temp_f.append(np.concatenate([temp1[i], temp2[i], temp3[i], temp4[i], temp5[i], temp6[i]]))

    temp_f = np.array(temp_f)
    temp_f = np.expand_dims(temp_f, axis=2)
    return temp_f


X_train, y_train, subjects_train, X_test, y_test, subjects_test = Data.loadData("data_5_fullWindow/normalizadoEstandarizado_subjects")

number_folders = 10
for i in range(number_folders):
    X_train[i] = div_signal(X_train[i])
    X_test[i] = div_signal(X_test[i])


graph = tf.Graph()
with graph.as_default():
    sess = tf.Session(config=config, graph=graph)
    with tf.device('/gpu:0'):
        with sess.as_default():
            model = Interface.Red.build((None, 1, 50, 3), 2, number_convolutional_layers=4, first_number_filters=256,
                                        dropout=0.5)
            optimizer_noise = Noise.add_gradient_noise(keras.optimizers.Adam)
            model.load_weights(name_path + 'result' + str(0) + '_1.h5')
            model.layers.pop()
            model.compile(loss='categorical_crossentropy',
                               optimizer=optimizer_noise(noise_eta=0.01, noise_gamma=0.55,
                                                         weight_decay=0.00001, bpe=1),
                               metrics=['accuracy', Metrics.acc_fall, Metrics.acc_adl, Metrics.maa,
                                        Metrics.acc_callback, Metrics.sensitivity, Metrics.specificity])

            for i in range(len(X_test)):
                [_, X_train[i]] = model.predict(X_train[i], batch_size=128)
                [_, X_test[i]] = model.predict(X_test[i], batch_size=128)
                X_train[i] = np.expand_dims(X_train[i], axis= 1)
                X_train[i] = np.expand_dims(X_train[i], axis=3)
                X_test[i] = np.expand_dims(X_test[i], axis=1)
                X_test[i] = np.expand_dims(X_test[i], axis=3)

with open("../Data/data_5/"+'predictores_normalizadoEstandarizado_subjects.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([X_train, y_train, subjects_train, X_test, y_test, subjects_test], f)

quit()