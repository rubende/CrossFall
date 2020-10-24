
from scipy import signal


import numpy as np

from sklearn.model_selection import KFold

import pickle



def dataAugmentation(samples, labels, subjects):
    temp_sample = []
    temp_label = []
    temp_subject = []

    for i in range(len(samples)):
        sample = samples[i]
        label = labels[i]
        subject = subjects[i]
        sample_aug = []
        label_aug = []

        noiseSigma = 0.01 * sample;
        aug1 = noiseSigma * np.random.randn(sample.shape[0], sample.shape[1], sample.shape[2], sample.shape[3]);
        aug1 = sample + aug1

        aug2 = (1.1 - 0.7) * np.random.randn(1, 1) + 0.7;
        aug2 = sample * aug2;

        ns = len(sample)
        aug3 = signal.resample(sample, ns * 10);

        n = np.random.uniform(low=-5, high=5, size=(ns))
        # n = np.random.randn(1,ns)
        n = np.rint(n)

        # n = np.round(n)
        n[0] = 1;
        idx = [x * 10 for x in range(ns)]
        idx = idx + n
        aug3 = aug3[idx.astype(int)];

        sample_aug = sample
        sample_aug = np.append(sample_aug, aug1, 0)
        sample_aug = np.append(sample_aug, aug2, 0)
        sample_aug = np.append(sample_aug, aug3, 0)

        label_aug = label
        label_aug = np.append(label_aug, label, 0)
        label_aug = np.append(label_aug, label, 0)
        label_aug = np.append(label_aug, label, 0)

        subject_aug = subject
        subject_aug = np.append(subject_aug, subject, 0)
        subject_aug = np.append(subject_aug, subject, 0)
        subject_aug = np.append(subject_aug, subject, 0)

        temp_sample.append(np.array(sample_aug))
        temp_label.append(np.array(label_aug))
        temp_subject.append(np.array(subject_aug))

    return temp_sample, temp_label, temp_subject


def dataAugmentationWithoutFold(samples, labels, subjects):
    temp_sample = []
    temp_label = []
    temp_subject = []

    noiseSigma = 0.01 * samples;
    aug1 = noiseSigma * np.random.randn(samples.shape[0], samples.shape[1], samples.shape[2], samples.shape[3]);
    aug1 = samples + aug1

    aug2 = (1.1 - 0.7) * np.random.randn(1, 1) + 0.7;
    aug2 = samples * aug2;

    ns = len(samples)
    aug3 = signal.resample(samples, ns * 10);

    n = np.random.uniform(low=-5, high=5, size=(ns))
    # n = np.random.randn(1,ns)
    n = np.rint(n)

    # n = np.round(n)
    n[0] = 1;
    idx = [x * 10 for x in range(ns)]
    idx = idx + n
    aug3 = aug3[idx.astype(int)];

    sample_aug = samples
    sample_aug = np.append(sample_aug, aug1, 0)
    sample_aug = np.append(sample_aug, aug2, 0)
    sample_aug = np.append(sample_aug, aug3, 0)

    label_aug = labels
    label_aug = np.append(label_aug, labels, 0)
    label_aug = np.append(label_aug, labels, 0)
    label_aug = np.append(label_aug, labels, 0)

    subjects_aug = subjects
    subjects_aug = np.append(subjects_aug, subjects, 0)
    subjects_aug = np.append(subjects_aug, subjects, 0)
    subjects_aug = np.append(subjects_aug, subjects, 0)

    temp_sample.append(np.array(sample_aug))
    temp_label.append(np.array(label_aug))
    temp_subject.append(np.array(subjects_aug))

    return temp_sample, temp_label, temp_subject


def dataAugmentation_11(samples, labels, subjects, ages):
    temp_sample = []
    temp_label = []
    temp_subject = []
    temp_ages = []

    for i in range(len(samples)):
        sample = samples[i][0]
        label = labels[i]
        subject = subjects[i]
        age = ages[i]


        noiseSigma = 0.01 * sample
        aug1 = noiseSigma * np.random.randn(sample.shape[0], sample.shape[1])
        aug1 = sample + aug1

        aug2 = (1.1 - 0.7) * np.random.randn(1, 1) + 0.7
        aug2 = sample * aug2

        ns = len(sample)
        aug3 = signal.resample(sample, ns * 10)

        n = np.random.uniform(low=-5, high=5, size=(ns))
        # n = np.random.randn(1,ns)
        n = np.rint(n)

        # n = np.round(n)
        n[0] = 1;
        idx = [x * 10 for x in range(ns)]
        idx = idx + n
        aug3 = aug3[idx.astype(int)];

        temp_sample.append(np.array(sample))
        temp_sample.append(np.array(aug1))
        temp_sample.append(np.array(aug2))
        temp_sample.append(np.array(aug3))

        temp_label.append(np.array(label))
        temp_label.append(np.array(label))
        temp_label.append(np.array(label))
        temp_label.append(np.array(label))

        temp_subject.append(subject)
        temp_subject.append(subject)
        temp_subject.append(subject)
        temp_subject.append(subject)

        temp_ages.append(age)
        temp_ages.append(age)
        temp_ages.append(age)
        temp_ages.append(age)



    return temp_sample, temp_label, temp_subject, temp_ages


def dataAugmentationType(samples, labels, type):
    temp_sample = []
    temp_label = []

    for i in range(len(samples)):
        sample = samples[i]
        label = labels[i]
        print(label)
        sample_aug = []
        label_aug = []

        noiseSigma = 0.01 * sample
        aug1 = noiseSigma * np.random.randn(sample.shape[0], sample.shape[1], sample.shape[2], sample.shape[3]);
        aug1 = sample + aug1

        aug2 = (1.1 - 0.7) * np.random.randn(1, 1) + 0.7;
        aug2 = sample * aug2;

        ns = len(sample)
        aug3 = signal.resample(sample, ns * 10);

        n = np.random.uniform(low=-5, high=5, size=(ns))
        # n = np.random.randn(1,ns)
        n = np.rint(n)

        # n = np.round(n)
        n[0] = 1;
        idx = [x * 10 for x in range(ns)]
        idx = idx + n
        aug3 = aug3[idx.astype(int)];

        sample_aug = sample
        sample_aug = np.append(sample_aug, aug1, 0)
        sample_aug = np.append(sample_aug, aug2, 0)
        sample_aug = np.append(sample_aug, aug3, 0)

        label_aug = label
        label_aug = np.append(label_aug, label, 0)
        label_aug = np.append(label_aug, label, 0)
        label_aug = np.append(label_aug, label, 0)

        temp_sample.append(np.array(sample_aug))
        temp_label.append(np.array(label_aug))

    return temp_sample, temp_label


def k_fold(sample, labels, subject, split):
    kf = KFold(n_splits=split, random_state=0, shuffle=True)
    kf.get_n_splits(sample)

    X_train = []
    y_train = []
    subjects_train = []
    X_test = []
    y_test = []
    subjects_test = []

    for t1, t2 in kf.split(sample):
        X_train.append(sample[t1])
        X_test.append(sample[t2])
        y_train.append(labels[t1])
        y_test.append(labels[t2])

        subjects_train.append(subject[t1])
        subjects_test.append(subject[t2])

    return X_train, y_train, subjects_train, X_test, y_test, subjects_test


def dataNormalization(samples):
    for i in range(len(samples)):
        t = np.mean(samples[i], 0)
        t = np.mean(t, 1)
        samples[i] = samples[i] - t
    return samples


def dataNormalizationSample(samples):
    for i in range(len(samples)):
        t = np.mean(samples[i], 0)
        samples[i] = samples[i] - t
    return samples


def dataNormalizationScaler(samples):
    for i in range(len(samples)):
        mean = np.mean(samples[i], 0)
        mean = np.mean(mean, 1)

        std = np.std(samples[i],0)
        std = np.std(std, 1)

        samples[i] = (samples[i] - mean)/std
    return samples

def dataNormalizationScalerWithoutFold(samples):
    mean = np.mean(samples, 0)
    mean = np.mean(mean, 1)

    std = np.std(samples,0)
    std = np.std(std, 1)

    samples = (samples - mean)/std
    return samples


def dataNormalizationScaler_11(samples):
    for i in range(len(samples)):
        mean = []
        std = []
        for j in range(len(samples[i][0])):
            mean.append(np.mean(samples[i][0][j], 0))
            std.append(np.std(samples[i][0][j], 0))

        mean = np.mean(mean, 0)
        std = np.std(std, 0)
        for j in range(len(samples[i][0])):
            samples[i][0][j] = (samples[i][0][j] - mean)/std

    return samples


def dataNormalizationScalerSample(samples):
    for i in range(len(samples)):
        mean = np.mean(samples[i], 0)

        std = np.std(samples[i],0)

        samples[i] = (samples[i] - mean)/std
    return samples




def saveData(X_train, y_train, subjects_train, X_test, y_test, subjects_test, name):
    with open("../Data/" + name + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([X_train, y_train, subjects_train, X_test, y_test, subjects_test], f)


def loadData(name):
    with open("../Data/" + name + '.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        X_train, y_train, subjects_train, X_test, y_test, subjects_test = pickle.load(f)

    return X_train, y_train, subjects_train, X_test, y_test, subjects_test


def loadData_age(name):
    with open("../Data/" + name + '.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        X_train, y_train, subjects_train, ages_train, X_test, y_test, subjects_test, ages_test = pickle.load(f)

    return X_train, y_train, subjects_train, ages_train, X_test, y_test, subjects_test, ages_test