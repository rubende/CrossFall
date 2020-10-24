import keras
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD, Adagrad, Adadelta, Adam
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, TimeDistributed, LSTM
from numpy.random import seed
import Auxiliary.CallBacks_debug_MultiTask as Debug
import Auxiliary.noiseGradients as Noise
import Auxiliary.Metrics as Metrics

import keras.backend as K

import numpy as np
from keras.utils import np_utils
import math
import tensorflow as tf

class Red:
    @staticmethod
    def build(input_shape, num_classes_1, num_classes_2, number_convolutional_layers, first_number_filters, weight_decay=1e-4,
              dropout=0, L2_norm=None):

        if number_convolutional_layers < 1:
            print("ERROR: Number of convolutional layers must be greater than 0")

        if L2_norm != None:
            L2_norm = regularizers.l2(weight_decay)

        filter_size = 5

        model_input = Input(shape=input_shape)
        print(tf.shape(model_input))
        output = TimeDistributed(Conv2D(first_number_filters, kernel_size=(1, filter_size), kernel_regularizer=L2_norm, \
                        activation='relu', input_shape=input_shape))(model_input)
        output = TimeDistributed(BatchNormalization())(output)
        output = TimeDistributed(MaxPooling2D(pool_size=(1, 2)))(output)
        print(tf.shape(output))
        filter_size = filter_size - 1

        for i in range(1, number_convolutional_layers):
            first_number_filters = first_number_filters * 2
            output = TimeDistributed(Conv2D(first_number_filters, kernel_size=(1, filter_size), kernel_regularizer=L2_norm, \
                            activation='relu'))(output)
            output = TimeDistributed(BatchNormalization())(output)

            if i != number_convolutional_layers - 1:
                output = TimeDistributed(MaxPooling2D(pool_size=(1, 2)))(output)

            filter_size = filter_size - 1
            # model.summary()


        output = TimeDistributed(AveragePooling2D(pool_size=(1, output.shape[3].value)))(output)
        output = TimeDistributed(Flatten())(output)

        output = LSTM(128)(output)
        output_t = Dropout(dropout, name = "predictores")(output)

        #output = Dropout(dropout)(output)

        output_1 = Dense(num_classes_1, activation='softmax', name='caidas')(output_t)
        output_2 = Dense(num_classes_2, activation='softmax', name='id')(output_t)


        model = Model(inputs=model_input, outputs=[output_1, output_2])
        model.summary()
        return model

    def lr_schedule(epoch, lr):
        lrate = 0.01
        if epoch > 50:
            lrate = 0.001
        return lrate

    def lr_schedule2(epoch, lr):
        lrate = 0.01
        return lrate

    def lr_schedule3(epoch, lr):
        lrate = 0.001
        return lrate

    def train(model_defined, index_folder, path_file_last, x_train, y_train_1, y_train_2, x_test, y_test_1, y_test_2, batch_size=128, ADAM=0, save_name=None, noise=0,
              l2_noise=0, weight_decay_noise=0.0001, bpe_noise=1, class_weight = None, stepped = False, loss = None):
        seed(0)

        if noise == 0:
            if ADAM == 0:
                optimizer = SGD()
            else:
                optimizer = Adam()
        else:
            if l2_noise == 0:
                weight_decay_noise = 0
                bpe = 1

            if ADAM == 0:
                optimizer_noise = Noise.add_gradient_noise(keras.optimizers.SGD)
            else:
                optimizer_noise = Noise.add_gradient_noise(keras.optimizers.Adam)

        if noise == 0:
            model_defined.compile(loss='categorical_crossentropy', optimizer=optimizer,
                                  metrics=['accuracy', Metrics.acc_fall, Metrics.acc_adl, Metrics.maa,
                                           Metrics.acc_callback, Metrics.sensitivity, Metrics.specificity], loss_weights = loss)
        else:
            model_defined.compile(loss='categorical_crossentropy',
                                  optimizer=optimizer_noise(noise_eta=0.01, noise_gamma=0.55,
                                                            weight_decay=weight_decay_noise, bpe=bpe_noise),
                                  metrics=['accuracy', Metrics.acc_fall, Metrics.acc_adl, Metrics.maa,
                                           Metrics.acc_callback, Metrics.sensitivity, Metrics.specificity], loss_weights = loss)

        debug_callback = Debug.HistoryDebug(x_train, y_train_1, y_train_2, x_test, y_test_1, y_test_2, index_folder, path_file_last)

        if not stepped:
            if ADAM == 0:
                history = model_defined.fit(x_train, [y_train_1, y_train_2],
                                            batch_size=batch_size, epochs=100,
                                            verbose=0, validation_data=(x_test, [y_test_1, y_test_2]),
                                            callbacks=[LearningRateScheduler(Red.lr_schedule),
                                                       debug_callback], class_weight=class_weight)  # Modifica el lr del optimizador?
            else:
                history = model_defined.fit(x_train, [y_train_1, y_train_2],
                                            batch_size=batch_size, epochs=100,
                                            verbose=0, validation_data=(x_test, [y_test_1, y_test_2]),
                                            callbacks=[debug_callback], class_weight=class_weight)  # Modifica el lr del optimizador?
        else:
            t = np.unique(np.where(y_train_1 == np_utils.to_categorical(1, 2))[0])
            t2 = len(np.unique(t))
            if t2 == 0:
                t2 = 1

            t3 = np.unique(np.where(y_train_1 == np_utils.to_categorical(0, 2))[0])
            np.random.shuffle(t3)
            # print(t3)
            # t3 = np.unique(t3[0])
            # print(t2)
            # print(t3)
            # t3 = t3[:len(t3)-len(t3)%t2]
            # print(len(t3))
            # t3 = np.split(t3, math.floor(len(t3)/t2))
            print("y_train: " + str(len(y_train_1)))
            print("y_train_shape: " + str(x_train.shape))
            print("t3: " + str(len(t3)))
            print("t: " + str(len(t)))

            index = 0
            learning_rate = Red.lr_schedule2
            epochs = 20
            for i in range(math.ceil(len(t3)/t2)):
                print("div: " + str(math.ceil(len(t3)/t2)))
                if i == 1:
                    epochs = 10
                elif i == 4:
                    epochs  = 5
                elif i == 6:
                    learning_rate = Red.lr_schedule3

                index_temp = np.concatenate((t3[index:index+t2], t))
                print("HOLAAAAAAAAA")
                print(x_train[index_temp].shape)

                history = model_defined.fit(x_train[index_temp], [y_train_1[index_temp], y_train_2[index_temp]],
                                        batch_size=batch_size, epochs=epochs,
                                        verbose=0, validation_data=(x_test, [y_test_1, y_test_2]),
                                        callbacks=[debug_callback, LearningRateScheduler(learning_rate)],
                                        class_weight=class_weight)  # Modifica el lr del optimizador?

                index = index + t2
                if index + t2 > len(t3):
                    t2 = len(t3) - index



        if save_name != None:
            model_defined.save(save_name + '.h5')

        return history, model_defined


    def _parse_sample_function(proto):

        # Create a dictionary describing the features.
        sample_feature_description = {
            'x': tf.FixedLenFeature([51], tf.float32),
            'y': tf.FixedLenFeature([51], tf.float32),
            'z': tf.FixedLenFeature([51], tf.float32),
            'label': tf.FixedLenFeature([1], tf.int64),
        }

        # Load one example
        parsed_features = tf.parse_single_example(proto, sample_feature_description)

        return parsed_features

    def plot_info(history):
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
