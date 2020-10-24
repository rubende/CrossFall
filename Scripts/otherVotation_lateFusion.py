
import Redes.Red_Ini as Interface
import Auxiliary.noiseGradients as Noise
import Auxiliary.Metrics as Metrics
import Auxiliary.preprocessingData as Data
import numpy as np
import keras
import os
import pickle
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # To force tensorflow to only see one GPU.
# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True




name_path = "../Results/result_05_fullWindow/late_fusion_4_capas/escalado/con_weight_0.85__sin_weight/"
X_train, y_train, subjects_train, X_test, y_test, subjects_test = Data.loadData("data_5_fullWindow/normalizadoEstandarizado")

number_folds = 10

# ------------------------

t1 = []
t2 = []
labels = []

file = open(name_path+"otherVotation.txt","w")

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session(config=config, graph=graph)
    with tf.device('/gpu:0'):
        with sess.as_default():
            model1 = Interface.Red.build((1, 301, 3), 2, number_convolutional_layers=4, first_number_filters=256,
                                         dropout=0.5)
            optimizer_noise = Noise.add_gradient_noise(keras.optimizers.Adam)
            model1.compile(loss='categorical_crossentropy', optimizer=optimizer_noise(noise_eta=0.01, noise_gamma=0.55,
                                                                                      weight_decay=0.00001, bpe=1),
                           metrics=['accuracy', Metrics.acc_fall, Metrics.acc_adl, Metrics.maa,
                                    Metrics.acc_callback, Metrics.sensitivity, Metrics.specificity])


            model2 = Interface.Red.build((1, 301, 3), 2, number_convolutional_layers=4, first_number_filters=256,
                                         dropout=0.5)
            optimizer_noise = Noise.add_gradient_noise(keras.optimizers.Adam)
            model2.compile(loss='categorical_crossentropy', optimizer=optimizer_noise(noise_eta=0.01, noise_gamma=0.55,
                                                                                      weight_decay=0.00001, bpe=1),
                           metrics=['accuracy', Metrics.acc_fall, Metrics.acc_adl, Metrics.maa,
                                    Metrics.acc_callback, Metrics.sensitivity, Metrics.specificity])

            for i in range(number_folds):
                model1.load_weights(name_path + 'result' + str(i) + '_1.h5')
                model2.load_weights(name_path + 'result' + str(i) + '_2.h5')
                t1.append(model1.predict(X_test[i], batch_size=128))
                t2.append(model2.predict(X_test[i], batch_size=128))
                labels.append(np.argmax(y_test[i], axis=1))

            Matrix = [[0 for x in range(9)] for y in range(9)]
            for j in np.arange(0.1, 1.0, 0.1):
                for k in np.arange(0.1, 1.0, 0.1):
                    # file.write("Para " + str(j) + " y " + str(k) + ":\n")
                    all_acc = 0
                    all_sensitivity = 0
                    all_specificity = 0
                    all_maa = 0
                    for i in range(number_folds):

                        pred_1 = t1[i] * j
                        pred_2 = t2[i] * k
                        pred_3 = np.argmax(pred_1 + pred_2, axis=1)

                        equals_fusion = np.equal(pred_3, labels[i])
                        equals_fusion = [int(elem) for elem in equals_fusion]
                        equals_fusion = np.array(equals_fusion)
                        not_equals_fusion = 1 - equals_fusion

                        index_fall = np.squeeze(np.argwhere(labels[i] == 1))
                        index_adl = np.squeeze(np.argwhere(labels[i] == 0))

                        TP_fusion = np.sum(equals_fusion[index_fall])
                        FN_fusion = np.sum(not_equals_fusion[index_fall])
                        TN_fusion = np.sum(equals_fusion[index_adl])
                        FP_fusion = np.sum(not_equals_fusion[index_adl])

                        sensitivity_fusion = TP_fusion / (TP_fusion + FN_fusion)
                        specificity_fusion = TN_fusion / (TN_fusion + FP_fusion)
                        maa_fusion = (sensitivity_fusion + specificity_fusion) / 2
                        acc_fusion = np.sum(equals_fusion) / len(labels[i])

                        # file.write("Para batch " + str(i) + ":\n")
                        # file.write("acc: " + str(acc_fusion) + "\n")
                        all_acc = all_acc + acc_fusion
                        # file.write("sensitivity: " + str(sensitivity_fusion) + "\n")
                        all_sensitivity = all_sensitivity + sensitivity_fusion
                        # file.write("specificity: " + str(specificity_fusion) + "\n")
                        all_specificity = all_specificity + specificity_fusion
                        # file.write("maa: " + str(maa_fusion) + "\n")
                        all_maa = all_maa + maa_fusion

                    # file.write("Media:" + "\n")
                    # file.write("acc_fusion: " + str(all_acc / number_folds) + "\n")
                    # file.write("sensitivity_fusion: " + str(all_sensitivity / number_folds) + "\n")
                    # file.write("specificity_fusion: " + str(all_specificity / number_folds) + "\n")
                    # file.write("maa_fusion: " + str(all_maa / number_folds) + "\n")
                    # file.write("######################################\n")

                    Matrix[int(j*10)-1][int(k*10)-1] = [all_acc / number_folds, all_sensitivity / number_folds, all_specificity / number_folds, all_maa / number_folds]


with open(name_path+'otherVotation.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(Matrix, f)

# file.close()
quit()

