import scipy.io
import keras
from keras.datasets import cifar100
import numpy as np
import skimage.io
import glob
import os.path
from PIL import Image


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

# print("resizing traffic signs")
# resize_images('datasets/trafficsigns-test/*.ppm')
# resize_images('datasets/trafficsigns-train/**/*.ppm')
# print("resizing svhn")
# resize_images('datasets/svhn-train/*.png')
# resize_images('datasets/svhn-test/*.png')
# print("resizing grass")
# resize_images('datasets/grass/*.JPEG')
# print("resizing storefront")
# resize_images('datasets/storefront/*.JPEG')
# colour_images('datasets/storefront/*_32x32.JPEG')
#
print("imread svhn")
svhn_train = skimage.io.imread_collection('datasets/svhn-train/*_32x32.png')
svhn_test = skimage.io.imread_collection('datasets/svhn-test/*_32x32.png')

print("imread traffic")
traffic_signs_train = skimage.io.imread_collection('datasets/trafficsigns-train/**/*_32x32.ppm')
traffic_signs_test = skimage.io.imread_collection('datasets/trafficsigns-test/*_32x32.ppm')

print("imread grass")
grass = skimage.io.imread_collection('datasets/grass/*_32x32.JPEG')
train_length = int(len(grass)*0.8)
grass_train = grass[:train_length]
grass_test = grass[train_length:]

print("imread store front")
store_fronts = skimage.io.imread_collection('datasets/storefront/*_32x32.JPEG')
train_length = int(len(store_fronts)*0.8)
store_fronts_train = store_fronts[:train_length]
store_fronts_test = store_fronts[train_length:]
#
# print("concatenating")
# train = np.concatenate((svhn_train, traffic_signs_train, grass_train, store_fronts_train), axis=0)
# test = np.concatenate((svhn_test, traffic_signs_test, grass_test, store_fronts_test), axis=0)
# print(train.shape, test.shape)
#
# np.save('datasets/train.npy', train)
# np.save('datasets/test.npy', test)

# keras.backend.set_image_dim_ordering('th')
# svhn_train = scipy.io.loadmat('datasets/train_32x32.mat')['X']
# svhn_test = scipy.io.loadmat('datasets/test_32x32.mat')['X']
# (cifar_train, cifar_train_labels), (cifar_test, cifar_test_labels) = cifar100.load_data(label_mode='fine')
#
# svhn_train = np.transpose(svhn_train, (3,2,0,1))
# svhn_test = np.transpose(svhn_test, (3,2,0,1))
#
# # as all the svhn data just has a single label (street numbers),
# # we give that a number 100 to fit with the 0-99 from cifar100
svhn_train_labels = np.ndarray(shape=(len(svhn_train), 1), dtype='uint8')
svhn_train_labels.fill(0)
svhn_test_labels = np.ndarray(shape=(len(svhn_test), 1), dtype='uint8')
svhn_test_labels.fill(0)

traffic_train_labels = np.ndarray(shape=(len(traffic_signs_train), 1), dtype='uint8')
traffic_train_labels.fill(1)
traffic_test_labels = np.ndarray(shape=(len(traffic_signs_test), 1), dtype='uint8')
traffic_test_labels.fill(1)

grass_train_labels = np.ndarray(shape=(len(grass_train), 1), dtype='uint8')
grass_train_labels.fill(2)
grass_test_labels = np.ndarray(shape=(len(grass_test), 1), dtype='uint8')
grass_test_labels.fill(2)

store_fronts_train_labels = np.ndarray(shape=(len(store_fronts_train), 1), dtype='uint8')
store_fronts_train_labels.fill(3)
store_fronts_test_labels = np.ndarray(shape=(len(store_fronts_test), 1), dtype='uint8')
store_fronts_test_labels.fill(3)
# 
train_labels = np.concatenate((svhn_train_labels, traffic_train_labels, grass_train_labels, store_fronts_train_labels), axis=0)
test_labels = np.concatenate((svhn_test_labels, traffic_test_labels, grass_test_labels, store_fronts_test_labels), axis=0)
# 
# #
# # train = np.concatenate((svhn_train, cifar_train), axis=0)
# # test = np.concatenate((svhn_test, cifar_test), axis=0)
# #
# # train_labels = np.concatenate((svhn_train_labels, cifar_train_labels), axis=0)
# # test_labels = np.concatenate((svhn_test_labels, cifar_test_labels), axis=0)
# #
# #
np.save('datasets/train_labels.npy', train_labels)
np.save('datasets/test_labels.npy', test_labels)