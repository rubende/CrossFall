
from keras import backend as K
import tensorflow as tf


def acc_class(y_true, y_pred, n_class):

    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    accuracy_mask = K.cast(K.equal(class_id_true, n_class), 'int32')
    sum_mask = K.sum(accuracy_mask)
    class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
    class_acc_tensor = K.sum(class_acc_tensor)
    class_acc_f = class_acc_tensor / K.maximum(sum_mask, 1)

    return class_acc_f


def acc_fall(y_true, y_pred):

    return acc_class(y_true, y_pred, 1)


def acc_adl(y_true, y_pred):

    return acc_class(y_true, y_pred, 0)


def maa(y_true, y_pred):

    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    accuracy_mask = K.cast(K.equal(class_id_true, 1), 'int32')
    sum_mask = K.sum(accuracy_mask)
    # sum_mask = K.print_tensor(sum_mask, 'sum_mask_1 = ')
    class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
    class_acc_tensor = K.sum(class_acc_tensor)
    class_acc_f = class_acc_tensor / K.maximum(sum_mask, 1)

    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    accuracy_mask = K.cast(K.equal(class_id_true, 0), 'int32')
    sum_mask = K.sum(accuracy_mask)
    # sum_mask = K.print_tensor(sum_mask, 'sum_mask_2 = ')
    class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
    class_acc_tensor = K.sum(class_acc_tensor)
    class_acc_a = class_acc_tensor / K.maximum(sum_mask, 1)

    return (class_acc_f + class_acc_a)/2


def acc_callback(y_true, y_pred):

    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    temp = K.sum(K.cast(K.equal(class_id_true, class_id_preds), 'int32'))
    # temp = K.print_tensor(temp, 'temp= ')
    shape_temp = K.shape(y_true)[0]
    # shape_temp = K.print_tensor(shape_temp, 'shape_temp= ')

    return temp/shape_temp


def maa_tensorflow(y_true, y_pred):
    return tf.metrics.mean_per_class_accuracy(y_true, y_pred, 2)[0]


def sensitivity(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    accuracy_mask = K.cast(K.equal(class_id_true, 1), 'int32')
    TP = K.sum(K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask)
    FN = K.sum(K.cast(K.not_equal(class_id_true, class_id_preds), 'int32') * accuracy_mask)

    return TP/(TP+FN)


def specificity(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    accuracy_mask = K.cast(K.equal(class_id_true, 0), 'int32')
    TN = K.sum(K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask)
    FP = K.sum(K.cast(K.not_equal(class_id_true, class_id_preds), 'int32') * accuracy_mask)

    return TN/(TN+FP)

def f1(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    accuracy_mask = K.cast(K.equal(class_id_true, 1), 'int32')
    TP = K.sum(K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask)
    FN = K.sum(K.cast(K.not_equal(class_id_true, class_id_preds), 'int32') * accuracy_mask)

    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    accuracy_mask = K.cast(K.equal(class_id_true, 0), 'int32')
    TN = K.sum(K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask)
    FP = K.sum(K.cast(K.not_equal(class_id_true, class_id_preds), 'int32') * accuracy_mask)

    precision = TP/(TP + FP)
    recall = TP/(TP + FN)

    f1 = 2*((precision*recall)/(precision+recall))
    return f1

def f1_tensorflow(y_true, y_pred):
    y_true = K.cast(y_true,  'int32')
    y_pred = K.cast(y_pred, 'int32')
    return tf.contrib.metrics.f1_score(y_true, y_pred, 2)[0]

def f1_internet(y_true, y_pred):

    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    #f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    #return K.mean(f1)
    return f1

def f1_internet3(y_true, y_pred):

    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
    return f1

def f1_internet2(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())


    return f1_val