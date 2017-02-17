import keras
from keras.datasets import cifar100
import numpy as np
import skimage.io
import glob
import os
from PIL import Image
from sklearn.cross_validation import train_test_split
import math
import re


def resize_images(path):
    for infile in glob.glob(path):
        if os.path.isfile(infile):
            filename, ext = os.path.splitext(infile)
            if "_32x32" not in filename:
                print("resizing: {0}".format(filename + ext))
                image = Image.open(infile)
                image = image.resize((110, 110))
                image.save(filename + "_110x110" + ext)


def colour_images(path):
    for infile in glob.glob(path):
        if os.path.isfile(infile):
            filename, ext = os.path.splitext(infile)
            if "_110x110" in filename:
                print("opening: {0}".format(filename + ext))
                image = Image.open(infile)
                if image.mode != "RGB":
                    print("image not RGB, colouring")
                    image = image.convert("RGB")
                    image.save(filename + ext)


def create_labels_from_directories(paths, num_images):
    labels = np.zeros(shape=(num_images,), dtype='uint8')
    file_nums_list = []
    for j, path in enumerate(paths):
        for i, directory in enumerate(os.listdir(path)):
            num_files = len([name for name in os.listdir(os.path.join(path, directory)) if os.path.isfile(os.path.join(path, directory, name)) and "_110x110" in name])

            # dir_labels = np.ndarray(shape=(num_files,), dtype='uint8')
            # dir_labels.fill(i)
            # print(dir_labels.shape, dir_labels)
            if j == 0:
                labels[:num_files] = j

            if i == 0:
                file_nums_list.append(num_files)
                # print(labels)
            else:
                start = np.sum(file_nums_list)
                file_nums_list.append(num_files)
                end = np.sum(file_nums_list)
                labels[start:end] = i
                # print(labels)
    total_current_labels = np.sum(file_nums_list)
    current_num_classes = len(np.unique(labels))
    return labels, total_current_labels, current_num_classes

# print("resizing images")
# resize_images('../../datasets/svhn-train/*.png')
# resize_images('../../datasets/svhn-test/*.png')
# resize_images('../../datasets/trafficsigns-train/**/*.ppm')
# resize_images('../../datasets/trafficsigns-test/*.ppm')
# print("colouring images")
# colour_images('../../datasets/svhn-train/*.png')
# colour_images('../../datasets/svhn-test/*.png')
# colour_images('../../datasets/trafficsigns-train/**/*.ppm')
# colour_images('../../datasets/trafficsigns-test/*.ppm')


def chunk_image_data(desired_num_chunks):
    print("loading data")
    places2 = skimage.io.imread_collection('../../Downloads/train_places365/**/**/*_110x110.jpg')
    print("creating labels")
    labels, total_current_labels, num_classes = create_labels_from_directories(glob.glob('../../Downloads/train_places365/**'), len(places2))
    print(labels.shape, num_classes)

    places2_length = len(places2)
    print(places2_length)
    if places2_length % 2 == 0:
        step = 2
    else:
        step = 1
    data_length_factors = [x for x in range(1, int(math.sqrt(places2_length))+1, step) if places2_length % x == 0]
    print(data_length_factors)
    desired_chunk_divisor = desired_num_chunks
    chunk_divisor = math.inf
    previous_difference = math.inf
    for factor in data_length_factors:
        if abs(factor - desired_chunk_divisor) < previous_difference:
            chunk_divisor = factor
        previous_difference = abs(factor - desired_chunk_divisor)

    print(chunk_divisor)
    # find factors of the length, and then find the closest factor to a divisor that will chunk the data into small enough
    # chunks - this way we get an exact divisor of the dataset, rather than missing out images because of truncation.

    chunk_size = int(len(places2) / chunk_divisor)

    # completed_chunks = []
    # for file in os.listdir('.'):
    #     if os.path.isfile(file) and "chunk_" in file:
    #         chunk_number_pos = re.search(r'[0-9]', file).start()
    #         completed_chunk_num = file[chunk_number_pos:chunk_number_pos+1]
    #     if completed_chunk_num not in completed_chunks:
    #         completed_chunks.append(completed_chunk_num)

    # find the chunk files we've already finished,


    chunk_number = 0
    print("Beginning chunking with chunk size: {0}".format(chunk_size))
    for i in range(0, len(places2), chunk_size):
        print("Chunk number: {0}".format(chunk_number))
        print("Slicing image data.")
        chunk = places2[i:i+chunk_size].concatenate()
        print(chunk.shape)
        print("Slicing labels.")
        chunk_labels = labels[i:i+chunk_size]
        print(chunk_labels.shape)
        print("Splitting train and test sets.")
        train, test, train_labels, test_labels = train_test_split(chunk, chunk_labels, test_size=0.1, random_state=43)
        print("Splitting train and validation sets.")
        train, validation, train_labels, validation_labels = train_test_split(train, train_labels, test_size=0.1, random_state=234)

        print("Saving train, test and validation.")
        np.save("chunk_{0}_train.npy".format(chunk_number), train)
        np.save("chunk_{0}_test.npy".format(chunk_number), test)
        np.save("chunk_{0}_validation.npy".format(chunk_number), validation)

        print("Saving labels.")
        np.save("chunk_{0}_train_labels.npy".format(chunk_number), train_labels)
        np.save("chunk_{0}_test_labels.npy".format(chunk_number), test_labels)
        np.save("chunk_{0}_validation_labels.npy".format(chunk_number), validation_labels)
        # train test split, with labels
        # train validation split, with labels
        # save files, using chunk number
        print(train.shape, test.shape, validation.shape, train_labels.shape, test_labels.shape, validation_labels.shape)
        chunk_number += 1

chunk_image_data(137)

# print("loading main image set")
# outdoor_images = skimage.io.imread_collection('datasets/selected_images/**/*_32x32.jpg')
# print(outdoor_images[0].shape)
#
# print("making labels")
# labels, total_current_labels, current_num_classes = create_labels_from_directories('datasets/selected_images', len(outdoor_images))
# print(len(np.unique(labels)))
#
# print("loading svhn")
# svhn_train = skimage.io.imread_collection('datasets/svhn-train/*_32x32.png')
# svhn_test = skimage.io.imread_collection('datasets/svhn-test/*_32x32.png')
# svhn_train_labels = np.ndarray(shape=(len(svhn_train),), dtype='uint8')
# svhn_train_labels.fill(current_num_classes + 1)
# svhn_test_labels = np.ndarray(shape=(len(svhn_test),), dtype='uint8')
# svhn_test_labels.fill(current_num_classes + 1)
# print(svhn_train[0].shape)
#
# print("loading traffic signs")
# traffic_signs_train = skimage.io.imread_collection('datasets/trafficsigns-train/**/*_32x32.ppm')
# traffic_signs_test = skimage.io.imread_collection('datasets/trafficsigns-test/*_32x32.ppm')
# traffic_signs_train_labels = np.ndarray(shape=(len(traffic_signs_train),), dtype='uint8')
# traffic_signs_train_labels.fill(current_num_classes + 2)
# traffic_signs_test_labels = np.ndarray(shape=(len(traffic_signs_test),), dtype='uint8')
# traffic_signs_test_labels.fill(current_num_classes + 2)
# print(traffic_signs_train[0].shape)
#
# print("concatenating")
# data = np.concatenate((outdoor_images, svhn_train, traffic_signs_train, svhn_test, traffic_signs_test), axis=0)
# labels = np.concatenate((labels, svhn_train_labels, traffic_signs_train_labels, svhn_test_labels, traffic_signs_test_labels), axis=0)
#
# print("splitting train and test sets")
# train, test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=123412)
#
# print("creating validation sets")
# train, validation, labels_train, labels_validation = train_test_split(train, labels_train, test_size=0.2, random_state=34565)
#
# print(train.shape, test.shape, validation.shape, labels_train.shape, labels_test.shape, labels_validation.shape)
# print("saving files")
# np.save('datasets/train.npy', train)
# np.save('datasets/train_labels.npy', labels_train)
# np.save('datasets/test.npy', test)
# np.save('datasets/test_labels.npy', labels_test)
# np.save('datasets/validation.npy', validation)
# np.save('datasets/validation_labels.npy', labels_validation)
