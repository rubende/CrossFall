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
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3" # To force tensorflow to only see one GPU.
# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
###################################


number_folds = 5
predictors = []
labels = []
name_path = "../Results/result_05_fullWindow/lstm_4_capas/escalado/con_weight_0.67/"




[data_all, labels_all, subjects_all] = Raw.createDataset_12("../Raw/BaseDatos_012_Milan.mat")

#train = []
#for i in range(len(data_all)):
#    train.append(data_all[i][0])
#train = np.asarray(train)
#print(train.shape)
#train = np.expand_dims(train, axis=1)
data_all = Data.dataNormalizationScalerWithoutFold(data_all)
[data_all, labels_all, subjects_all] = Data.dataAugmentationWithoutFold(data_all, labels_all, subjects_all)

data_all = np.squeeze(data_all)
n_data_all = []
for i in range(len(data_all)):
    temp = data_all[i, 0:50]
    n_data_all.append(np.expand_dims(np.asarray(temp), axis=1))
data_all = np.asarray(n_data_all)
data_all = np.swapaxes(data_all,1,2)
data_all = np.expand_dims(data_all, axis=1)

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

            for i in range(len(data_all)):
                predictors.append(model.predict(np.expand_dims(data_all[i],0), batch_size=128))

with open("../Data/data_12/"+'predictores_normalizadoEstandarizado.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([data_all, labels_all, subjects_all, predictors], f)

quit()