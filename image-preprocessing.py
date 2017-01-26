import scipy.io
from keras.datasets import cifar100
import numpy as np

svhn_train = scipy.io.loadmat('datasets/train_32x32.mat')['X']
svhn_test = scipy.io.loadmat('datasets/test_32x32.mat')['X']
(cifar_train, cifar_train_labels), (cifar_test, cifar_test_labels) = cifar100.load_data(label_mode='fine')

svhn_train = np.transpose(svhn_train, (3,0,1,2))
svhn_test = np.transpose(svhn_test, (3,0,1,2))

# as all the svhn data just has a single label (street numbers),
# we give that a number 100 to fit with the 0-99 from cifar100
svhn_train_labels = np.ndarray(shape=(len(svhn_train), 1), dtype='uint8')
svhn_train_labels.fill(100)
svhn_test_labels = np.ndarray(shape=(len(svhn_test), 1), dtype='uint8')
svhn_test_labels.fill(100)

train = np.concatenate((svhn_train, cifar_train), axis=0)
test = np.concatenate((svhn_test, cifar_test), axis=0)

train_labels = np.concatenate((svhn_train_labels, cifar_train_labels), axis=0)
test_labels = np.concatenate((svhn_test_labels, cifar_test_labels), axis=0)

np.save('datasets/train.npy', train)
np.save('datasets/test.npy', test)
np.save('datasets/train_labels.npy', train_labels)
np.save('datasets/test_labels.npy', test_labels)