from numpy.random import seed

import scipy.io


from keras.utils import np_utils

import numpy as np
import pickle
import scipy as sc


def createDataset_12(path):
    seed(0)
    sample = []
    labels = []
    subject = []
    mat = scipy.io.loadmat(path)
    for i in range(mat['muestras']['Experiment_ID'].size):
        sample.append(mat['muestras'].item(i)[18][:, 1:4])
        subject.append(np_utils.to_categorical(int(mat['muestras'].item(i)[2][0][-1]), 30))

        label = mat['muestras'].item(i)[7]
        filter_label = lambda label: 1 if label == "Fall" else 0
        label = filter_label(label)

        labels.append(np_utils.to_categorical(label, 2))

    sample = np.expand_dims(sample, 1)

    return sample, np.array(labels), np.array(subject)


def createDataset_11(path):
    seed(0)
    sample = []
    labels = []
    subject = []
    ages = []
    mat = scipy.io.loadmat(path)
    for i in range(mat['muestras']['Experiment_ID'].size):
        sample.append(mat['muestras'].item(i)[18][:, 1:4])
        subject.append(np_utils.to_categorical(int(mat['muestras'].item(i)[2][0][-1]), 30))

        label = mat['muestras'].item(i)[7]
        age = mat['muestras'].item(i)[3]
        filter_label = lambda label: 1 if label == "Fall" else 0
        label = filter_label(label)

        labels.append(np_utils.to_categorical(label, 2))
        ages.append(age)

    sample = np.expand_dims(sample, 1)

    return sample, np.array(labels), np.array(subject), np.array(ages)

def createDataset_15(path):
    seed(0)
    sample = []
    labels = []
    subject = []
    mat = scipy.io.loadmat(path)
    for i in range(mat['muestras']['Experiment_ID'].size):
        if np.any(mat['muestras'].item(i)[23][:, 1:4]):
            sample.append(mat['muestras'].item(i)[23][:, 1:4])
            subject.append(np_utils.to_categorical(int(mat['muestras'].item(i)[2][0][-1]), 30))

            label = mat['muestras'].item(i)[7]
            filter_label = lambda label: 1 if label == "Fall" else 0
            label = filter_label(label)

            labels.append(np_utils.to_categorical(label, 2))

    sample = np.expand_dims(sample, 1)

    return sample, np.array(labels), np.array(subject)

def createDataset_07(path):
    seed(0)
    sample = []
    labels = []
    subject = []
    mat = scipy.io.loadmat(path)
    for i in range(mat['muestras']['Experiment_ID'].size):
        if np.any(mat['muestras'].item(i)[19][:, 1:4]):
            sample.append(mat['muestras'].item(i)[19][:, 1:4])
            subject.append(np_utils.to_categorical(int(mat['muestras'].item(i)[2][0][-1]), 30))

            label = mat['muestras'].item(i)[7]
            filter_label = lambda label: 1 if label == "Fall" else 0
            label = filter_label(label)

            labels.append(np_utils.to_categorical(label, 2))

    sample = np.expand_dims(sample, 1)

    return sample, np.array(labels), np.array(subject)


def createDataset_03(path):
    seed(0)
    sample = []
    labels = []
    subject = []
    mat = scipy.io.loadmat(path)
    for i in range(mat['muestras']['Experiment_ID'].size):
        if np.any(mat['muestras'].item(i)[18][:, 1:4]):
            sample.append(mat['muestras'].item(i)[18][:, 1:4])
            subject.append(np_utils.to_categorical(int(mat['muestras'].item(i)[2][0][-1]), 30))

            label = mat['muestras'].item(i)[7]
            filter_label = lambda label: 1 if label == "Fall" else 0
            label = filter_label(label)

            labels.append(np_utils.to_categorical(label, 2))

    sample = np.expand_dims(sample, 1)

    return sample, np.array(labels), np.array(subject)

def createDataset_05(path):
    data_adl = getAllDataAsListNew('adl')
    data_adl = data_adl[:, :, 125:176]
    data_adl = np.stack(data_adl, 2)
    data_adl = np.expand_dims(data_adl, 1)
    labels_adl = [np_utils.to_categorical(0, 2)] * len(data_adl)

    data_fall = getAllDataAsListNew('fall')
    data_fall = data_fall[:, :, 125:176]
    data_fall = np.stack(data_fall, 2)
    data_fall = np.expand_dims(data_fall, 1)
    labels_fall = [np_utils.to_categorical(1, 2)] * len(data_fall)

    data_all = np.concatenate((data_adl, data_fall))
    print(data_all.shape)
    labels_all = np.concatenate((labels_adl, labels_fall))
    print(labels_all.shape)

    with open("dataset_05" + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([data_all, labels_all], f)


def getAllDataAsListNew(kind):
    """
    Obtains data of all people together as a list (each member for a given person)
    Each entry is an array. We use the data in vectorial form to get only the total acceleration
    kind='fall' or 'adl'
    position='pocket' or 'hbag'
    Some combinations are not implemented yet
    Returns the list of data. Each element of the list is an array, in which each row is a temporal sequence
    of acceleration values
    """
    if(kind=='fall'):
        falldum=sc.loadtxt('data201307/person0/fallProcessedVector/0fallPV.dat')
        fall0=[falldum[0::3], falldum[1::3], falldum[2::3]]
        ###
        falldum=sc.loadtxt('data201307/person1/fallProcessedVector/1fallPV.dat')
        fall1=[falldum[0::3], falldum[1::3], falldum[2::3]]
        ###
        falldum=sc.loadtxt('data201307/person2/fallProcessedVector/2fallPV.dat')
        fall2=[falldum[0::3], falldum[1::3], falldum[2::3]]
        ###
        falldum=sc.loadtxt('data201307/person3/fallProcessedVector/3fallPV.dat')
        fall3=[falldum[0::3], falldum[1::3], falldum[2::3]]
        ###
        falldum=sc.loadtxt('data201307/person4/fallProcessedVector/4fallPV.dat')
        fall4=[falldum[0::3], falldum[1::3], falldum[2::3]]
        ###
        falldum=sc.loadtxt('data201307/person5/fallProcessedVector/5fallPV.dat')
        fall5=[falldum[0::3], falldum[1::3], falldum[2::3]]
        ###
        falldum=sc.loadtxt('data201307/person6/fallProcessedVector/6fallPV.dat')
        fall6=[falldum[0::3], falldum[1::3], falldum[2::3]]
        ###
        falldum=sc.loadtxt('data201307/person7/fallProcessedVector/7fallPV.dat')
        fall7=[falldum[0::3], falldum[1::3], falldum[2::3]]
        ###
        falldum=sc.loadtxt('data201307/person8/fallProcessedVector/8fallPV.dat')
        fall8=[falldum[0::3], falldum[1::3], falldum[2::3]]
        ###
        falldum=sc.loadtxt('data201307/person9/fallProcessedVector/9fallPV.dat')
        fall9=[falldum[0::3], falldum[1::3], falldum[2::3]]
        ###
        return np.concatenate([fall0, fall1, fall2, fall3, fall4, fall5, fall6, fall7, fall8, fall9], 1)
        ####################
    elif(kind=='adl'):
        adldum=sc.loadtxt('data201307/person0/adlProcessedVector/0adlPV.dat')
        adl0=[adldum[0::3], adldum[1::3], adldum[2::3]]
        ###
        adldum=sc.loadtxt('data201307/person1/adlProcessedVector/1adlPV.dat')
        adl1=[adldum[0::3], adldum[1::3], adldum[2::3]]
        ###
        adldum=sc.loadtxt('data201307/person2/adlProcessedVector/2adlPV.dat')
        adl2=[adldum[0::3], adldum[1::3], adldum[2::3]]
        ###
        adldum=sc.loadtxt('data201307/person3/adlProcessedVector/3adlPV.dat')
        adl3=[adldum[0::3], adldum[1::3], adldum[2::3]]
        ###
        adldum=sc.loadtxt('data201307/person4/adlProcessedVector/4adlPV.dat')
        adl4=[adldum[0::3], adldum[1::3], adldum[2::3]]
        ####
        adldum=sc.loadtxt('data201307/person5/adlProcessedVector/5adlPV.dat')
        adl5=[adldum[0::3], adldum[1::3], adldum[2::3]]
        ###
        adldum=sc.loadtxt('data201307/person6/adlProcessedVector/6adlPV.dat')
        adl6=[adldum[0::3], adldum[1::3], adldum[2::3]]
        ###
        adldum=sc.loadtxt('data201307/person7/adlProcessedVector/7adlPV.dat')
        adl7=[adldum[0::3], adldum[1::3], adldum[2::3]]
        ###
        adldum=sc.loadtxt('data201307/person8/adlProcessedVector/8adlPV.dat')
        adl8=[adldum[0::3], adldum[1::3], adldum[2::3]]
        ###
        adldum=sc.loadtxt('data201307/person9/adlProcessedVector/9adlPV.dat')
        adl9=[adldum[0::3], adldum[1::3], adldum[2::3]]
        ###
        return np.concatenate([adl0, adl1, adl2, adl3, adl4, adl5, adl6, adl7, adl8, adl9], 1)
    else:
        return ()
        ########################

