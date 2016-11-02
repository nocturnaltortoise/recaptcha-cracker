import pickle
import numpy as np
import keras
from keras.utils import np_utils
from keras.datasets import cifar100

# all this dataset loading is all cool and stuff, but it's flat, and reshaping is irritating, plus keras has a properly shaped

# def load_datasets(files):
#     datasets = {}
#     for file in files:
#         with open(file, 'rb') as f:
#             dict = pickle.load(f, encoding='bytes')
#             if 'meta' not in datasets:
#                 datasets['meta'] = dict
#             elif 'train' not in datasets:
#                 datasets['train'] = dict
#             else:
#                 datasets['test'] = dict
#             # filenames should be passed as a list meta, train, test
#     return datasets
#
# datasets = load_datasets(['cifar-100-python/meta', 'cifar-100-python/train', 'cifar-100-python/test'])
# print("train keys: ", datasets['train'].keys())
# print("train: ", datasets['train'][b'coarse_labels'])
# # coarse labels are superclasses (like fish, trees) fine labels are classes like trout, oak
# # coarse and fine labels in train and test are numbers referring to their indices in the meta dict
# # numbering the labels like this in the train and test sets also allows us to use a one-hot encoding of those labels,
# # and then decode it later
#
# # print("meta fine labels: ", len(datasets['meta'][b'fine_label_names']))
# for label in datasets['meta'][b'coarse_label_names']:
#     print(label)
# print("test: ", datasets['test'])

keras.backend.set_image_dim_ordering('th')
seed = 7
np.random.seed(seed)

# train_data = datasets['train'][b'data']
# train_data_2d = np.empty((train_data.shape[0], 32, 32, 3))
# print("reshaping train data")
# for image in train_data:
#     np.append(train_data_2d, image.reshape(32, 32, 3))
#
#
# train_labels = datasets['train'][b'fine_labels']
# test_data = datasets['test'][b'data'].reshape(32, 32, 3)
# test_data_2d = np.empty((test_data.shape[0], 32, 32, 3))
# print("reshaping test data")
# for image in test_data:
#     np.append(test_data_2d, image.reshape(32, 32, 3))
#
# test_labels = datasets['test'][b'fine_labels']
#
# print("train data: ", train_data_2d[0])

(train_data, train_labels), (test_data, test_labels) = cifar100.load_data(label_mode='fine')

normalised_train_data = train_data[:5000].astype('float32') / 255.0
normalised_test_data = test_data[:1000].astype('float32') / 255.0

one_hot_train_labels = np_utils.to_categorical(train_labels[:5000])
one_hot_test_labels = np_utils.to_categorical(test_labels[:1000])

num_classes = one_hot_test_labels.shape[1]

model = keras.models.Sequential()
model.add(keras.layers.Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=keras.constraints.maxnorm(3)))
# layer for filtering 2d images
# input shape - 32x32 images, each with three channels (i.e. RGB)
model.add(keras.layers.Dropout(0.2))  # 20% dropout
model.add(keras.layers.Convolution2D(32, 3, 3, border_mode='same', activation='relu', W_constraint=keras.constraints.maxnorm(3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
# think this is some kind of feature reduction layer
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu', W_constraint=keras.constraints.maxnorm(3))) # fully connected layer
model.add(keras.layers.Dropout(0.5)) # 50% dropout
model.add(keras.layers.Dense(num_classes, activation='softmax'))
# softmax layer as output so we can get probability distribution of classes

epochs = 25
learning_rate = 0.01
decay = learning_rate / epochs
sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

model.fit(normalised_train_data, one_hot_train_labels, validation_data=(normalised_test_data, one_hot_test_labels), nb_epoch=epochs, batch_size=32)
scores = model.evaluate(normalised_test_data, one_hot_test_labels, verbose=0)

model_yaml = model.to_yaml()

with open('initial-conv-net.yaml', 'w') as f:
    f.write(model_yaml)

model.save_weights('initial-conv-net-weights.h5')
