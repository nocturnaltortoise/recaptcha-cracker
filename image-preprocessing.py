import scipy.io
import keras
from keras.datasets import cifar100
import numpy as np
import skimage.io
import glob
import os
from PIL import Image
from sklearn.cross_validation import train_test_split


def resize_images(path):
    for infile in glob.glob(path):
        file, ext = os.path.splitext(infile)
        image = Image.open(infile)
        image = image.resize((32, 32))
        image.save(file + "_32x32" + ext)


def colour_images(path):
    for infile in glob.glob(path):
        file, ext = os.path.splitext(infile)
        image = Image.open(infile)
        if image.mode != "RGB":
            image = image.convert("RGB")
            image.save(file + ext)


def create_labels_from_directories(path, num_images):
    num_dirs = len(os.listdir(path))
    labels = np.zeros(shape=(num_images,), dtype='uint8')
    file_nums_list = []
    for i, directory in enumerate(os.listdir(path)):
        num_files = len([name for name in os.listdir(os.path.join(path, directory)) if os.path.isfile(os.path.join(path, directory, name)) and "_32x32" in name])

        # dir_labels = np.ndarray(shape=(num_files,), dtype='uint8')
        # dir_labels.fill(i)
        # print(dir_labels.shape, dir_labels)
        if i == 0:
            labels[:num_files] = i
            file_nums_list.append(num_files)
            # print(labels)
        else:
            start = np.sum(file_nums_list)
            file_nums_list.append(num_files)
            end = np.sum(file_nums_list)
            labels[start:end] = i
            print(i)
            print(labels[:start])
            # print(labels)
    total_current_labels = np.sum(file_nums_list)
    print(len(np.unique(labels)))
    return labels, total_current_labels, num_dirs


# resize_images('datasets/selected_images/**/*')
# print("colouring images")
# colour_images('datasets/selected_images/**/*')

print("loading main image set")
outdoor_images = skimage.io.imread_collection('datasets/selected_images/**/*_32x32.jpg')
print(outdoor_images[0].shape)

print("making labels")
labels, total_current_labels, current_num_classes = create_labels_from_directories('datasets/selected_images', len(outdoor_images))
print(len(np.unique(labels)))

print("loading svhn")
svhn_train = skimage.io.imread_collection('datasets/svhn-train/*_32x32.png')
svhn_test = skimage.io.imread_collection('datasets/svhn-test/*_32x32.png')
svhn_train_labels = np.ndarray(shape=(len(svhn_train),), dtype='uint8')
svhn_train_labels.fill(current_num_classes + 1)
svhn_test_labels = np.ndarray(shape=(len(svhn_test),), dtype='uint8')
svhn_test_labels.fill(current_num_classes + 1)
print(svhn_train[0].shape)

print("loading traffic signs")
traffic_signs_train = skimage.io.imread_collection('datasets/trafficsigns-train/**/*_32x32.ppm')
traffic_signs_test = skimage.io.imread_collection('datasets/trafficsigns-test/*_32x32.ppm')
traffic_signs_train_labels = np.ndarray(shape=(len(traffic_signs_train),), dtype='uint8')
traffic_signs_train_labels.fill(current_num_classes + 2)
traffic_signs_test_labels = np.ndarray(shape=(len(traffic_signs_test),), dtype='uint8')
traffic_signs_test_labels.fill(current_num_classes + 2)
print(traffic_signs_train[0].shape)

print("splitting train and test sets")
train, test, labels_train, labels_test = train_test_split(outdoor_images, labels, test_size=0.2, random_state=123412)

print("concatenating")
train = np.concatenate((train, svhn_train, traffic_signs_train), axis=0)
labels_train = np.concatenate((labels_train, svhn_train_labels, traffic_signs_train_labels), axis=0)
test = np.concatenate((test, svhn_test, traffic_signs_test), axis=0)
test_labels = np.concatenate((labels_test, svhn_test_labels, traffic_signs_test_labels), axis=0)

print("creating validation sets")
train, validation, labels_train, labels_validation = train_test_split(train, labels_train, test_size=0.2, random_state=34565)

print(train.shape, test.shape, validation.shape, labels_train.shape, labels_test.shape, labels_validation.shape)
print("saving files")
np.save('datasets/train.npy', train)
np.save('datasets/train_labels.npy', labels_train)
np.save('datasets/test.npy', test)
np.save('datasets/test_labels.npy', labels_test)
np.save('datasets/validation.npy', validation)
np.save('datasets/validation_labels.npy', labels_validation)
